"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
void_topology.py
=================
Doc §2.3 ("Пустота как часть поля, а не её отсутствие"): topology of the
field's LOW-field region.

Design note (2026-09, discussion that shaped this module): a void region is
not treated here as a standalone unit that combines with critical points the
way a new letter would. The working analogy is the white space enclosed by a
stroke in an old script -- not itself a separate sign, but what completes the
legibility of the sign around it. Concretely that means two things stay
structurally separate in this module:

  1. `void_topology_profile` computes region topology from the field ALONE --
     no critical point is consulted -- exactly as §2.3 asks: connectivity of
     the low-field region (Betti_0), through-tunnels (Betti_1), and enclosed
     cavities (Betti_2), at one or more explicit thresholds. Threshold choice
     is a representational choice (doc §20), not a constant this module
     picks on the caller's behalf.

  2. `link_critical_points_to_voids` is a SEPARATE, purely geometric join
     between an already-extracted critical-point list and these regions.
     Keeping it separate is what makes "does this void's shape complete a
     surrounding critical-point pattern, or combine with it more freely"
     into a testable distributional question (doc §6.1 method) rather than
     an answer baked into how the data happens to be stored.

Kept out of `critical_points.py` for the same reason: a void region has no
single position and no Hessian. Forcing it into a point-shaped descriptor
would silently answer doc §20.2's "what counts as a candidate" rather than
leaving it open.

Connectivity model: regions and their Betti numbers are computed with
GUDHI's cubical complex on the field's own voxel grid (`gudhi.CubicalComplex`),
the standard construction for cubical homology -- not a voxel-touching
heuristic. Validated here against three hand-built cases with known answers
(a shell enclosing a cavity, a torus-shaped tunnel through a solid block, and
several disconnected low-field pockets) before use on real fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dataclass_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gudhi
except ImportError as _exc:  # pragma: no cover - environment-dependent
    gudhi = None
    _GUDHI_IMPORT_ERROR = _exc
else:
    _GUDHI_IMPORT_ERROR = None

from scipy import ndimage

from .critical_points import CriticalPoint
from .schema import ConfidenceBlock


def _require_gudhi() -> None:
    if gudhi is None:  # pragma: no cover - environment-dependent
        raise ImportError(
            "void_topology requires the 'gudhi' package (pip install gudhi "
            f"--break-system-packages) to compute Betti numbers; import "
            f"failed with: {_GUDHI_IMPORT_ERROR}"
        )


@dataclass
class VoidRegion:
    """One connected component of a field's low-field region at one
    threshold. Not a point: `centroid` and `principal_extents` are a cheap
    shape summary, not a claim that the region is well-described by them --
    see `VoidTopologySnapshot.notes`.
    """

    region_id: str
    molecule_id: str
    threshold: float
    field_kind: str
    level_of_theory: str
    environment: str
    voxel_count: int
    volume: float  # physical volume, field's native coordinate units^3
    centroid: np.ndarray  # Cartesian, field's native units
    principal_extents: Tuple[float, float, float]  # sqrt of PCA eigenvalues of
    # voxel positions about the centroid -- an elongated (tunnel-like) region
    # has one extent much larger than the other two; a blob-like cavity has
    # three comparable extents. Not a substitute for `betti_1`/`betti_2`
    # below, which is what actually distinguishes a tunnel from a cavity.
    betti_0_local: int  # always 1 for a single connected component by
    # construction; kept for a uniform field name with the snapshot-level
    # aggregate rather than a surprising omission.
    betti_1_local: int  # independent through-tunnels within THIS region alone
    betti_2_local: int  # enclosed cavities within THIS region alone
    boundary_voxel_positions: np.ndarray  # Cartesian centers of this region's
    # surface voxels only (mask minus its binary erosion) -- small even when
    # `voxel_count` is large, and enough for `link_critical_points_to_voids`
    # to test proximity without re-loading the full 3D mask.

    def to_json_safe(self) -> Dict:
        return {
            "region_id": self.region_id,
            "molecule_id": self.molecule_id,
            "threshold": float(self.threshold),
            "field_kind": self.field_kind,
            "level_of_theory": self.level_of_theory,
            "environment": self.environment,
            "voxel_count": int(self.voxel_count),
            "volume": float(self.volume),
            "centroid": np.asarray(self.centroid, dtype=float).tolist(),
            "principal_extents": [float(x) for x in self.principal_extents],
            "betti_1_local": int(self.betti_1_local),
            "betti_2_local": int(self.betti_2_local),
            "boundary_voxel_count": int(len(self.boundary_voxel_positions)),
        }


@dataclass
class VoidTopologySnapshot:
    """Region topology of one field at one threshold. §2.3's three
    Betti-like quantities are `betti_0`/`betti_1`/`betti_2` below, computed
    over the WHOLE low-field space (the direct sum over its connected
    components -- consistent with, not merely alongside, `regions`, each of
    which additionally carries its own local Betti_1/Betti_2)."""

    molecule_id: str
    threshold: float
    threshold_rule: str  # e.g. "quantile:0.05" -- how the threshold was
    # chosen, so a corpus-wide comparison never silently mixes an absolute
    # cutoff from one molecule with a quantile-derived one from another.
    connectivity_model: str
    betti_0: int
    betti_1: int
    betti_2: int
    regions: List[VoidRegion]
    confidence: ConfidenceBlock


def _betti_numbers_of_mask(mask: np.ndarray) -> Tuple[int, int, int]:
    """Betti_0/1/2 of the True region of a 3D boolean mask, via a binary
    cubical complex. `mask=True` marks membership in the region whose
    topology is wanted. See module docstring for validation cases."""
    _require_gudhi()
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    if not mask.any():
        return (0, 0, 0)
    values = np.where(mask, 0.0, 1.0).astype(np.float64)
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=values)
    complex_.compute_persistence()
    betti = [0, 0, 0]
    for dim, (birth, death) in complex_.persistence():
        if dim > 2:
            continue
        if birth <= 0.0 and death > 0.0:
            betti[dim] += 1
    return (betti[0], betti[1], betti[2])


def _sublevel_betti_profile(
    grid: np.ndarray,
    thresholds: Sequence[float],
) -> List[Tuple[int, int, int]]:
    """Compute low-field Betti numbers at many cutoffs in one filtration.

    The former implementation rebuilt a binary cubical complex and computed
    persistence independently at every threshold.  A cubical filtration of
    the original scalar field contains the same sublevel-set answers for all
    requested thresholds, so pay the persistence cost once per molecule.
    Strict births and inclusive deaths match the existing mask definition
    ``grid < threshold`` at values exactly on a cutoff.
    """
    if len(thresholds) == 0:
        return []
    _require_gudhi()
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("sublevel Betti profile requires a finite 3-D scalar field")
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=values)
    complex_.compute_persistence()
    intervals = complex_.persistence()
    profile: List[Tuple[int, int, int]] = []
    for threshold in thresholds:
        betti = [0, 0, 0]
        cutoff = float(threshold)
        for dimension, (birth, death) in intervals:
            if dimension <= 2 and birth < cutoff <= death:
                betti[dimension] += 1
        profile.append((betti[0], betti[1], betti[2]))
    return profile


def _voxel_to_cartesian(indices: np.ndarray, origin: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.asarray(origin, dtype=float) + np.asarray(indices, dtype=float) @ np.asarray(vectors, dtype=float)


def _default_thresholds(grid: np.ndarray, quantiles: Sequence[float]) -> List[Tuple[float, str]]:
    finite = grid[np.isfinite(grid)]
    return [
        (float(np.quantile(finite, q)), f"quantile:{q:g}")
        for q in quantiles
    ]


def void_topology_profile(
    field,
    *,
    molecule_id: str,
    quantile_thresholds: Sequence[float] = (0.02, 0.05, 0.10),
    explicit_thresholds: Optional[Sequence[float]] = None,
    min_region_voxels: int = 4,
) -> List[VoidTopologySnapshot]:
    """One `VoidTopologySnapshot` per threshold. `field` must expose
    `.grid`, `.origin`, `.vectors` (any `ScalarField` backed by a sampled
    grid, e.g. `CubeFileField`) -- this is a region computation and has no
    meaning for an unsampled analytic field.

    `quantile_thresholds` are quantiles of the field's OWN value
    distribution over the sampled box, not absolute density cutoffs --
    portable across molecules of very different scale, at the cost of
    needing `threshold_rule` recorded alongside every result so two
    snapshots are only compared when they used the same rule (same
    discipline as `analysis_signature`, doc §15/§20.5). Pass
    `explicit_thresholds` instead (or as well) for an absolute cutoff, e.g.
    a fixed density isovalue meaningful across a whole homogeneous corpus.

    `min_region_voxels` drops components too small to be more than grid
    noise from `regions` (they still count toward the snapshot's aggregate
    `betti_0`, since dropping them from the SUM would misstate the actual
    topology of the low-field space -- only the per-region breakdown omits
    them).
    """
    grid = getattr(field, "grid", None)
    origin = getattr(field, "origin", None)
    vectors = getattr(field, "vectors", None)
    if grid is None or origin is None or vectors is None:
        raise ValueError(
            "void_topology_profile needs a grid-backed field (.grid/.origin/"
            ".vectors); sample the field onto a grid first"
        )
    grid = np.asarray(grid)
    vectors = np.asarray(vectors, dtype=float)
    voxel_volume = abs(float(np.linalg.det(vectors)))
    field_kind = getattr(field, "field_kind", "unspecified")
    level_of_theory = getattr(field, "level_of_theory", "unspecified")
    environment = getattr(field, "environment", "unspecified")

    thresholds: List[Tuple[float, str]] = list(_default_thresholds(grid, quantile_thresholds))
    for value in explicit_thresholds or ():
        thresholds.append((float(value), "explicit"))

    snapshots: List[VoidTopologySnapshot] = []
    aggregate_betti_profile = _sublevel_betti_profile(
        grid, [threshold for threshold, _rule in thresholds],
    )
    for (threshold, rule), (betti_0, betti_1, betti_2) in zip(thresholds, aggregate_betti_profile):
        mask = grid < threshold

        labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        # `labeled == component_index` over the entire molecule-sized grid for
        # every component makes this loop O(grid_voxels * components).  A
        # low-field threshold can produce many tiny islands, so the old path
        # repeatedly allocated full 3-D boolean arrays even though regions
        # below `min_region_voxels` are discarded.  Count once and obtain each
        # retained component's tight crop from SciPy's single label scan.
        component_sizes = np.bincount(labeled.ravel(), minlength=n_components + 1)
        component_slices = ndimage.find_objects(labeled, max_label=n_components)
        regions: List[VoidRegion] = []
        for component_index, slices in enumerate(component_slices, start=1):
            voxel_count = int(component_sizes[component_index])
            if voxel_count < min_region_voxels:
                continue
            if slices is None:
                continue
            local_labels = labeled[slices]
            component_mask = local_labels == component_index
            local_indices = np.argwhere(component_mask)
            starts = np.asarray([axis_slice.start for axis_slice in slices], dtype=int)
            indices = local_indices + starts
            centroid_idx = indices.mean(axis=0)
            centroid = _voxel_to_cartesian(centroid_idx, origin, vectors)
            centered = _voxel_to_cartesian(indices, origin, vectors) - centroid
            # Principal extents: sqrt of eigenvalues of the voxel-position
            # covariance -- a cheap shape summary, not a homology substitute.
            if len(indices) >= 2:
                cov = np.cov(centered, rowvar=False)
                eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
                extents = tuple(float(np.sqrt(v)) for v in sorted(eigvals, reverse=True))
            else:
                extents = (0.0, 0.0, 0.0)

            # Crop to this component's own bounding box (padded by 1 voxel)
            # before the per-region Betti call -- cheap even when the whole
            # grid is large, since a void region is normally a small
            # fraction of it.
            lo = indices.min(axis=0)
            hi = indices.max(axis=0) + 1
            pad = 1
            lo_p = np.clip(lo - pad, 0, None)
            hi_p = np.minimum(hi + pad, np.array(mask.shape))
            padded_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lo_p, hi_p))
            local_mask = labeled[padded_slices] == component_index
            _, local_b1, local_b2 = _betti_numbers_of_mask(local_mask)

            boundary_mask = component_mask & ~ndimage.binary_erosion(component_mask)
            boundary_indices = np.argwhere(boundary_mask) + starts
            boundary_positions = _voxel_to_cartesian(boundary_indices, origin, vectors)

            regions.append(VoidRegion(
                region_id=f"{molecule_id}_void_t{rule}_{component_index:04d}",
                molecule_id=molecule_id,
                threshold=threshold,
                field_kind=field_kind,
                level_of_theory=level_of_theory,
                environment=environment,
                voxel_count=voxel_count,
                volume=voxel_count * voxel_volume,
                centroid=centroid,
                principal_extents=extents,
                betti_0_local=1,
                betti_1_local=local_b1,
                betti_2_local=local_b2,
                boundary_voxel_positions=boundary_positions,
            ))

        snapshots.append(VoidTopologySnapshot(
            molecule_id=molecule_id,
            threshold=threshold,
            threshold_rule=rule,
            connectivity_model="gudhi_cubical_complex",
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            regions=regions,
            confidence=ConfidenceBlock(
                n=1, method="analytic_CI", score=1.0, corpus_version="n/a",
            ),
        ))
    return snapshots


def link_critical_points_to_voids(
    critical_points: Sequence[CriticalPoint],
    regions: Sequence[VoidRegion],
    *,
    boundary_distance: float,
) -> Dict[str, List[str]]:
    """Purely geometric join, kept separate from `void_topology_profile` on
    purpose (see module docstring): maps each region's `region_id` to the
    field_type-tagged ids of critical points within `boundary_distance` of
    that region's *surface* voxels -- not its centroid, so an elongated
    tunnel-shaped region correctly links to points along its whole rim
    rather than only near its (possibly empty) geometric center.

    This says only that a critical point sits near a region's boundary; it
    does not decide whether that proximity is functionally meaningful --
    that is a distributional question for the corpus, not for this join.
    """
    links: Dict[str, List[str]] = {region.region_id: [] for region in regions}
    if not critical_points or not regions:
        return links
    cp_positions = np.stack([np.asarray(cp.position, dtype=float) for cp in critical_points])
    cp_labels = [
        f"{cp.field_type}@{','.join(f'{x:.4f}' for x in cp.position)}"
        for cp in critical_points
    ]
    for region in regions:
        if len(region.boundary_voxel_positions) == 0:
            continue
        # Nearest boundary-voxel distance per critical point; a full
        # cdist is fine at this scale (boundary voxels of one region and
        # a molecule's handful of critical points), and exact.
        deltas = cp_positions[:, None, :] - region.boundary_voxel_positions[None, :, :]
        min_dist = np.linalg.norm(deltas, axis=2).min(axis=1)
        for label, dist in zip(cp_labels, min_dist):
            if dist <= boundary_distance:
                links[region.region_id].append(label)
    return links


def write_void_snapshots(
    void_archive_dir,
    molecule_id: str,
    snapshots: Sequence[VoidTopologySnapshot],
    *,
    links: Optional[Dict[str, List[str]]] = None,
) -> "Path":
    """One `{molecule_id}.json` per molecule under `void_archive_dir`
    (`ArtifactPaths.void_archive_dir` -- see its docstring for why this is
    its own directory rather than folded into `letter_archive_dir`).
    Written from the same extraction worker that finds this molecule's
    critical points (see `esm_master_analytics.py`'s
    `_extract_record_worker`), so it exists exactly once a molecule has
    been processed at all -- not behind a separate, optional step.

    `links` (from `link_critical_points_to_voids`, region_id -> critical-
    point labels) is stored alongside when the caller already computed it,
    purely as a convenience so a later reader does not need both files to
    see the join; the join itself is still computed independently of this
    write, per the module's design (§2.3 discussion: a void completes a
    surrounding letter's legibility, it is not folded into being one)."""
    from pathlib import Path as _Path
    import json as _json
    import os as _os

    directory = _Path(void_archive_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{molecule_id}.json"
    payload = {
        "molecule_id": molecule_id,
        "snapshots": [
            {
                "threshold": snap.threshold,
                "threshold_rule": snap.threshold_rule,
                "connectivity_model": snap.connectivity_model,
                "betti_0": snap.betti_0,
                "betti_1": snap.betti_1,
                "betti_2": snap.betti_2,
                "regions": [region.to_json_safe() for region in snap.regions],
            }
            for snap in snapshots
        ],
        "boundary_links": links or {},
    }
    tmp_path = path.with_suffix(f".{_os.getpid()}.tmp")
    tmp_path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    return path

