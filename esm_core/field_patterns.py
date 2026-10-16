"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Field-native, multi-scale candidate regions for the ESM alphabet search.

This module deliberately does not import or inspect atoms, atomic basins, or
critical points.  It proposes *regions of the sampled scalar field* by
tracking connected components of both superlevel and sublevel sets over a
small, recorded set of within-field quantiles.  A proposal may contain many
critical points, and a proposal is never required to be centred on one.
Bounded complement components of superlevel regions are also proposed, so an
interior low-field cavity is not lost merely because the same low-field
sublevel connects to the padded grid exterior.

The output is a candidate archive, not a claim that each component is a
letter.  Nested components provide provisional part/whole evidence.  Reuse
of a small descriptor among several children of one larger component is
reported as a possible primitive-reuse signal; it does not by itself decide
whether the parent is a letter, syllable, or word.

The discretized connected-component construction is a practical proposal
family, not a completeness theorem.  Quantile, connectivity, clipping, and
size controls are recorded so that alternative proposal families can be
compared without silently changing the meaning of a result.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import ndimage


FIELD_PATTERN_SCHEMA = "esm-field-region-candidates/v2"
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _sample_component_flat_indices(
    mask: np.ndarray,
    voxel_count: int,
    sample_limit: int,
    seed_material: str,
) -> np.ndarray:
    """Return exact members for small masks and a bounded uniform sample otherwise."""
    flat_mask = mask.reshape(-1)
    if voxel_count <= sample_limit:
        return np.flatnonzero(flat_mask).astype(np.int64, copy=False)
    occupancy = voxel_count / max(int(flat_mask.size), 1)
    draw_count = int(np.ceil(2.5 * sample_limit / max(occupancy, 1e-12)))
    draw_count = min(int(flat_mask.size), max(2 * sample_limit, draw_count), 1_000_000)
    seed = int.from_bytes(hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest(), "big")
    rng = np.random.default_rng(seed)
    selected = np.unique(rng.integers(0, flat_mask.size, size=draw_count, dtype=np.int64))
    selected = selected[flat_mask[selected]]
    if len(selected) > sample_limit:
        # `selected` is sorted by np.unique; regular spacing through that sort
        # would correlate the sample with flat-grid position. Randomly subsample
        # instead so rotations/translations do not inherit that ordering bias.
        selected = rng.choice(selected, size=sample_limit, replace=False)
    return selected.astype(np.int64, copy=False)


@dataclass
class FieldRegionCandidate:
    candidate_id: str
    molecule_id: str
    polarity: str
    region_family: str
    quantile: float
    threshold: float
    centroid: Tuple[float, float, float]
    voxel_count: int
    physical_volume: float
    index_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    principal_extents: Tuple[float, float, float]
    raw_field_quantiles: Tuple[float, ...]
    invariant_descriptor: Tuple[float, ...]
    descriptor_sample_count: int
    descriptor_is_sampled: bool
    touches_grid_boundary: bool
    # Kept only during extraction to build cross-scale containment links.
    _flat_voxel_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64), repr=False)

    def to_json_safe(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("_flat_voxel_indices", None)
        payload["centroid"] = [float(x) for x in self.centroid]
        payload["index_bounds"] = [[int(a), int(b)] for a, b in self.index_bounds]
        payload["principal_extents"] = [float(x) for x in self.principal_extents]
        payload["raw_field_quantiles"] = [float(x) for x in self.raw_field_quantiles]
        payload["invariant_descriptor"] = [float(x) for x in self.invariant_descriptor]
        return payload


@dataclass(frozen=True)
class FieldPartWholeRelation:
    child_candidate_id: str
    parent_candidate_id: str
    relation_kind: str
    # For nested components this is the sampled fraction of child voxels in
    # the parent; for a bounded void it is the fraction of adjacent shell
    # voxels belonging to the selected enclosing component.
    relation_fraction: float
    scale_ratio: float
    relation_sample_count: int
    relation_is_sampled: bool


@dataclass
class FieldPatternBatch:
    molecule_id: str
    field_kind: str
    level_of_theory: str
    environment: str
    grid_shape: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    grid_vectors: Tuple[Tuple[float, float, float], ...]
    thresholds: Dict[str, List[Dict[str, float]]]
    candidates: List[FieldRegionCandidate]
    part_whole_relations: List[FieldPartWholeRelation]
    repeated_part_groups: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    method: str = "field_quantile_excursions_and_bounded_voids_no_cp_or_atom_seeds_v2"

    def to_json_safe(self) -> Dict[str, Any]:
        return {
            "schema": FIELD_PATTERN_SCHEMA,
            "method": self.method,
            "molecule_id": self.molecule_id,
            "field_kind": self.field_kind,
            "level_of_theory": self.level_of_theory,
            "environment": self.environment,
            "grid_shape": list(self.grid_shape),
            "origin": list(self.origin),
            "grid_vectors": [list(row) for row in self.grid_vectors],
            "thresholds": self.thresholds,
            "candidate_count": len(self.candidates),
            "part_whole_relation_count": len(self.part_whole_relations),
            "repeated_part_group_count": len(self.repeated_part_groups),
            "candidates": [candidate.to_json_safe() for candidate in self.candidates],
            "part_whole_relations": [asdict(item) for item in self.part_whole_relations],
            "repeated_part_groups": self.repeated_part_groups,
            "candidate_role_hypotheses": self.diagnostics.get(
                "candidate_role_hypotheses", []
            ),
            "diagnostics": self.diagnostics,
        }


def _component_descriptor(
    grid: np.ndarray,
    sample_indices: np.ndarray,
    exact_index_center: np.ndarray,
    origin: np.ndarray,
    vectors: np.ndarray,
    robust_center: float,
    robust_scale: float,
    polarity: str,
) -> Tuple[np.ndarray, Tuple[float, ...], Tuple[float, float, float], Tuple[float, float, float]]:
    """Rotation/translation-invariant summary of one voxel component.

    It preserves shape, radial occupancy, and the interior field-value
    distribution separately.  It is intentionally a coarse candidate
    fingerprint: no finite set of moments is asserted to uniquely identify a
    3-D field region.
    """
    centered_index = sample_indices - exact_index_center
    covariance_index = centered_index.T @ centered_index / max(len(sample_indices), 1)
    # A row of `vectors` maps an index-axis displacement into Cartesian space.
    covariance_cartesian = vectors.T @ covariance_index @ vectors
    eigvals = np.maximum(np.linalg.eigvalsh(covariance_cartesian), 0.0)
    extents = np.sqrt(eigvals)
    mean_extent = max(float(np.mean(extents)), 1e-12)
    shape_ratios = extents / mean_extent
    voxel_scale = max(abs(float(np.linalg.det(vectors))) ** (1.0 / 3.0), 1e-12)
    log_scale = float(np.log(max(mean_extent / voxel_scale, 1e-12)))

    cartesian = np.asarray(origin, dtype=float) + sample_indices @ vectors
    center_cartesian = np.asarray(origin, dtype=float) + exact_index_center @ vectors
    radii = np.linalg.norm(cartesian - center_cartesian, axis=1)
    max_radius = max(float(np.quantile(radii, 0.99)), voxel_scale)
    radial_hist, _ = np.histogram(
        np.clip(radii / max_radius, 0.0, 1.0),
        bins=12,
        range=(0.0, 1.0),
    )
    radial_hist = radial_hist.astype(float)
    radial_hist /= max(float(radial_hist.sum()), 1.0)

    sampled_values = grid[tuple(sample_indices.T)].astype(float, copy=False)
    q_levels = np.linspace(0.05, 0.95, 12)
    raw_quantiles = tuple(float(x) for x in np.quantile(sampled_values, q_levels))
    normalized_quantiles = (np.asarray(raw_quantiles) - robust_center) / robust_scale
    polarity_feature = 1.0 if polarity == "superlevel" else -1.0
    descriptor = np.concatenate((
        shape_ratios,
        np.asarray([log_scale]),
        radial_hist,
        normalized_quantiles,
        np.asarray([polarity_feature]),
    ))
    return (
        descriptor,
        raw_quantiles,
        tuple(float(x) for x in center_cartesian),
        tuple(float(x) for x in extents),
    )


def _similarity_distance(a: FieldRegionCandidate, b: FieldRegionCandidate) -> float:
    """A provisional, block-balanced distance used only for repeated-part flags."""
    va = np.asarray(a.invariant_descriptor, dtype=float)
    vb = np.asarray(b.invariant_descriptor, dtype=float)
    # Blocks: normalized extents, log scale, radial occupancy, normalized
    # field quantiles, polarity. Equal block weight avoids the 12-bin value
    # profile drowning the small shape descriptor merely by having more axes.
    shape = float(np.linalg.norm(va[0:3] - vb[0:3]) / np.sqrt(3.0))
    scale = float(abs(va[3] - vb[3]))
    radial = float(np.linalg.norm(va[4:16] - vb[4:16]) / np.sqrt(2.0))
    values = float(np.linalg.norm(va[16:28] - vb[16:28]) / np.sqrt(12.0))
    polarity = float(abs(va[28] - vb[28]) / 2.0)
    return float(np.sqrt((shape * shape + scale * scale + radial * radial + values * values + polarity * polarity) / 5.0))


def _repeated_child_groups(
    candidates_by_id: Mapping[str, FieldRegionCandidate],
    relations: Sequence[FieldPartWholeRelation],
    distance_threshold: float,
    min_part_scale_ratio: float,
) -> List[Dict[str, Any]]:
    children_by_parent: Dict[str, List[str]] = {}
    for relation in relations:
        if relation.scale_ratio < min_part_scale_ratio:
            continue
        children_by_parent.setdefault(relation.parent_candidate_id, []).append(
            relation.child_candidate_id
        )
    groups: List[Dict[str, Any]] = []
    for parent_id, child_ids in children_by_parent.items():
        if len(child_ids) < 2:
            continue
        clusters: List[List[str]] = []
        for child_id in child_ids:
            candidate = candidates_by_id[child_id]
            for cluster in clusters:
                representative = candidates_by_id[cluster[0]]
                if _similarity_distance(candidate, representative) <= distance_threshold:
                    cluster.append(child_id)
                    break
            else:
                clusters.append([child_id])
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            distances = [
                _similarity_distance(candidates_by_id[left], candidates_by_id[right])
                for offset, left in enumerate(cluster)
                for right in cluster[offset + 1:]
            ]
            groups.append({
                "parent_candidate_id": parent_id,
                "child_candidate_ids": cluster,
                "n_repeated_parts": len(cluster),
                "repeated_part_volume_fraction": float(
                    sum(candidates_by_id[item].voxel_count for item in cluster)
                    / max(candidates_by_id[parent_id].voxel_count, 1)
                ),
                "mean_pairwise_descriptor_distance": float(np.mean(distances)) if distances else 0.0,
                "distance_threshold_used": float(distance_threshold),
                "interpretation": "repeated_subpattern_signal_only_not_a_word_class_verdict",
            })
    return groups


def extract_field_pattern_candidates(
    field: Any,
    molecule_id: str,
    *,
    quantiles: Sequence[float] = (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.995),
    min_voxels: int = 8,
    max_regions_per_level: int = 512,
    max_grid_fraction: float = 0.98,
    include_boundary_components: bool = False,
    containment_min_fraction: float = 0.80,
    repeated_part_distance: float = 0.25,
    min_part_scale_ratio: float = 1.5,
    descriptor_sample_voxels: int = 4096,
    containment_sample_voxels: int = 8192,
) -> FieldPatternBatch:
    """Propose field regions without using atoms, basins, or critical points.

    Both high-field and low-field excursion components are evaluated.  Their
    nested containment across adjacent quantile levels supplies candidate
    part/whole relations.  Components touching the grid boundary are excluded
    by default because their extent is clipped by the finite cube, not because
    they are assumed chemically unimportant.
    """
    grid = getattr(field, "grid", None)
    origin = getattr(field, "origin", None)
    vectors = getattr(field, "vectors", None)
    if grid is None or origin is None or vectors is None:
        raise ValueError("field-pattern extraction needs a grid-backed scalar field")
    grid = np.asarray(grid)
    origin = np.asarray(origin, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if grid.ndim != 3 or not np.all(np.isfinite(grid)):
        raise ValueError("field-pattern extraction needs a finite 3-D grid")
    if origin.shape != (3,) or vectors.shape != (3, 3):
        raise ValueError("field grid geometry must be origin(3) and vectors(3,3)")
    if abs(float(np.linalg.det(vectors))) <= 1e-15:
        raise ValueError("field grid vectors must span non-zero physical volume")
    quantiles = tuple(sorted({float(q) for q in quantiles}))
    if not quantiles or any(not 0.0 < q < 1.0 for q in quantiles):
        raise ValueError("quantiles must be a non-empty sequence strictly inside (0,1)")
    if min_voxels < 1 or max_regions_per_level < 1:
        raise ValueError("min_voxels and max_regions_per_level must be positive")
    if not 0.0 < max_grid_fraction <= 1.0:
        raise ValueError("max_grid_fraction must be in (0,1]")
    if not 0.0 <= containment_min_fraction <= 1.0:
        raise ValueError("containment_min_fraction must be in [0,1]")
    if repeated_part_distance < 0.0:
        raise ValueError("repeated_part_distance must be non-negative")
    if min_part_scale_ratio < 1.0:
        raise ValueError("min_part_scale_ratio must be at least 1")
    if descriptor_sample_voxels < 1 or containment_sample_voxels < 1:
        raise ValueError("descriptor and containment sample limits must be positive")

    finite = grid.astype(np.float64, copy=False)
    all_quantiles = sorted(set(quantiles) | {0.02, 0.50, 0.98})
    all_quantile_values = np.quantile(finite, all_quantiles)
    quantile_thresholds = {
        float(q): float(value) for q, value in zip(all_quantiles, all_quantile_values)
    }
    robust_center = quantile_thresholds[0.50]
    robust_scale = quantile_thresholds[0.98] - quantile_thresholds[0.02]
    if robust_scale <= 1e-15:
        robust_scale = max(float(np.max(finite) - np.min(finite)), 1.0)

    voxel_volume = abs(float(np.linalg.det(vectors)))
    n_grid = int(grid.size)
    candidates: List[FieldRegionCandidate] = []
    relations: List[FieldPartWholeRelation] = []
    thresholds_record: Dict[str, List[Dict[str, float]]] = {"sublevel": [], "superlevel": []}
    diagnostics: Dict[str, Any] = {
        "min_voxels": int(min_voxels),
        "max_regions_per_level": int(max_regions_per_level),
        "max_grid_fraction": float(max_grid_fraction),
        "include_boundary_components": bool(include_boundary_components),
        "relation_fraction_minimum": float(containment_min_fraction),
        "repeated_part_distance": float(repeated_part_distance),
        "min_part_scale_ratio": float(min_part_scale_ratio),
        "descriptor_sample_voxels": int(descriptor_sample_voxels),
        "containment_sample_voxels": int(containment_sample_voxels),
        "dropped_too_small": 0,
        "dropped_too_large": 0,
        "dropped_boundary_clipped": 0,
        "dropped_by_per_level_cap": 0,
        "levels_processed": 0,
    }

    def process_polarity(polarity: str, q_levels: Sequence[float]) -> None:
        previous_level_candidates: List[FieldRegionCandidate] = []
        previous_threshold: float | None = None
        for q in q_levels:
            threshold = quantile_thresholds[float(q)]
            if previous_threshold is not None and np.isclose(
                threshold, previous_threshold, rtol=1e-12, atol=1e-15
            ):
                continue  # identical excursion mask; do not manufacture a scale variant
            previous_threshold = threshold
            mask = finite <= threshold if polarity == "sublevel" else finite >= threshold
            labels, n_labels = ndimage.label(mask, structure=_CONNECTIVITY_26)
            sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
            objects = ndimage.find_objects(labels, max_label=n_labels)
            eligible = []
            for component_label, slices in enumerate(objects, start=1):
                if slices is None:
                    continue
                voxel_count = int(sizes[component_label])
                if voxel_count < min_voxels:
                    diagnostics["dropped_too_small"] += 1
                    continue
                if voxel_count / n_grid > max_grid_fraction:
                    diagnostics["dropped_too_large"] += 1
                    continue
                eligible.append((voxel_count, component_label, slices))
            eligible.sort(key=lambda item: (-item[0], item[1]))
            if len(eligible) > max_regions_per_level:
                diagnostics["dropped_by_per_level_cap"] += len(eligible) - max_regions_per_level
                eligible = eligible[:max_regions_per_level]

            current_level: List[FieldRegionCandidate] = []
            label_to_candidate: Dict[int, FieldRegionCandidate] = {}
            for component_rank, (voxel_count, component_label, slices) in enumerate(eligible):
                starts = np.asarray([item.start for item in slices], dtype=np.int64)
                stops = np.asarray([item.stop for item in slices], dtype=np.int64)
                local_labels = labels[slices]
                local_mask = local_labels == component_label
                touches_boundary = bool(
                    (starts[0] == 0 and np.any(local_mask[0, :, :]))
                    or (stops[0] == grid.shape[0] and np.any(local_mask[-1, :, :]))
                    or (starts[1] == 0 and np.any(local_mask[:, 0, :]))
                    or (stops[1] == grid.shape[1] and np.any(local_mask[:, -1, :]))
                    or (starts[2] == 0 and np.any(local_mask[:, :, 0]))
                    or (stops[2] == grid.shape[2] and np.any(local_mask[:, :, -1]))
                )
                if touches_boundary and not include_boundary_components:
                    diagnostics["dropped_boundary_clipped"] += 1
                    continue
                local_shape = local_mask.shape
                descriptor_flat = _sample_component_flat_indices(
                    local_mask,
                    voxel_count,
                    descriptor_sample_voxels,
                    f"{molecule_id}|{polarity}|{q:.8f}|{component_label}|descriptor",
                )
                descriptor_sample_count = len(descriptor_flat)
                local_shape = local_mask.shape
                local_i, local_j, local_k = np.unravel_index(descriptor_flat, local_shape)
                descriptor_indices = np.column_stack((local_i, local_j, local_k)) + starts
                center_local = ndimage.center_of_mass(local_mask)
                exact_index_center = np.asarray(center_local, dtype=float) + starts

                containment_flat_local = _sample_component_flat_indices(
                    local_mask,
                    voxel_count,
                    containment_sample_voxels,
                    f"{molecule_id}|{polarity}|{q:.8f}|{component_label}|containment",
                )
                containment_sample_count = len(containment_flat_local)
                contain_i, contain_j, contain_k = np.unravel_index(
                    containment_flat_local, local_shape
                )
                contain_i = contain_i + starts[0]
                contain_j = contain_j + starts[1]
                contain_k = contain_k + starts[2]
                flat_indices = (
                    contain_i * (grid.shape[1] * grid.shape[2])
                    + contain_j * grid.shape[2]
                    + contain_k
                ).astype(np.int64, copy=False)
                descriptor, raw_quantiles, centroid, extents = _component_descriptor(
                    finite, descriptor_indices, exact_index_center, origin, vectors,
                    robust_center, robust_scale, polarity,
                )
                candidate_id = f"{molecule_id}:{polarity}:q{q:.5f}:c{component_rank:05d}"
                bounds = tuple((int(starts[i]), int(stops[i] - 1)) for i in range(3))
                candidate = FieldRegionCandidate(
                    candidate_id=candidate_id,
                    molecule_id=molecule_id,
                    polarity=polarity,
                    region_family="excursion_component",
                    quantile=float(q),
                    threshold=threshold,
                    centroid=centroid,
                    voxel_count=voxel_count,
                    physical_volume=float(voxel_count * voxel_volume),
                    index_bounds=bounds,
                    principal_extents=extents,
                    raw_field_quantiles=raw_quantiles,
                    invariant_descriptor=tuple(float(x) for x in descriptor),
                    descriptor_sample_count=int(descriptor_sample_count),
                    descriptor_is_sampled=descriptor_sample_count < voxel_count,
                    touches_grid_boundary=touches_boundary,
                    _flat_voxel_indices=flat_indices,
                )
                candidates.append(candidate)
                current_level.append(candidate)
                label_to_candidate[component_label] = candidate

            thresholds_record[polarity].append({"quantile": float(q), "threshold": threshold})
            diagnostics["levels_processed"] += 1

            # The current level is broader than the previous one by
            # construction: strict low-quantile pockets expand with q;
            # high-quantile islands expand as q moves down. Relate every
            # retained child to its strongest containing component.
            if previous_level_candidates:
                flat_labels = labels.reshape(-1)
                for child in previous_level_candidates:
                    parent_labels = flat_labels[child._flat_voxel_indices]
                    parent_labels = parent_labels[parent_labels > 0]
                    if not len(parent_labels):
                        continue
                    counts = np.bincount(parent_labels, minlength=n_labels + 1)
                    parent_label = int(np.argmax(counts))
                    overlap = int(counts[parent_label])
                    # `flat_voxel_indices` is a bounded deterministic sample
                    # for large regions; divide by its size, not the full mask.
                    fraction = overlap / max(len(child._flat_voxel_indices), 1)
                    parent = label_to_candidate.get(parent_label)
                    if (
                        parent is not None
                        and parent.candidate_id != child.candidate_id
                        and fraction >= containment_min_fraction
                    ):
                        relations.append(FieldPartWholeRelation(
                            child_candidate_id=child.candidate_id,
                            parent_candidate_id=parent.candidate_id,
                            relation_kind="nested_excursion_component",
                            relation_fraction=float(fraction),
                            scale_ratio=float(parent.voxel_count / max(child.voxel_count, 1)),
                            relation_sample_count=int(len(child._flat_voxel_indices)),
                            relation_is_sampled=(len(child._flat_voxel_indices) < child.voxel_count),
                        ))

            # A low-field minimum enclosed by a high-field shell is not an
            # ordinary sublevel component: its complement component is
            # connected to the padded exterior and was previously discarded
            # as boundary-clipped.  Record bounded complement components as
            # first-class field candidates, then link each to the surrounding
            # superlevel component by boundary adjacency.  No CP or atom seed
            # is involved; this is specifically useful for ring-like fields
            # whose interior minimum may be more informative than the maxima
            # on the ring.
            if polarity == "superlevel":
                # Reuse the threshold mask buffer; making a second full-size
                # boolean complement needlessly raises peak RAM per worker.
                np.logical_not(mask, out=mask)
                void_labels, n_void_labels = ndimage.label(
                    mask, structure=_CONNECTIVITY_26
                )
                void_sizes = np.bincount(
                    void_labels.ravel(), minlength=n_void_labels + 1
                )
                void_objects = ndimage.find_objects(
                    void_labels, max_label=n_void_labels
                )
                boundary_ids = set()
                for face in (
                    void_labels[0, :, :], void_labels[-1, :, :],
                    void_labels[:, 0, :], void_labels[:, -1, :],
                    void_labels[:, :, 0], void_labels[:, :, -1],
                ):
                    boundary_ids.update(int(x) for x in np.unique(face) if x > 0)

                eligible_voids = []
                for void_label, void_slices in enumerate(void_objects, start=1):
                    if void_slices is None or void_label in boundary_ids:
                        continue
                    void_count = int(void_sizes[void_label])
                    if void_count < min_voxels:
                        diagnostics["dropped_too_small"] += 1
                        continue
                    if void_count / n_grid > max_grid_fraction:
                        diagnostics["dropped_too_large"] += 1
                        continue
                    eligible_voids.append((void_count, void_label, void_slices))
                eligible_voids.sort(key=lambda item: (-item[0], item[1]))
                if len(eligible_voids) > max_regions_per_level:
                    diagnostics["dropped_by_per_level_cap"] += (
                        len(eligible_voids) - max_regions_per_level
                    )
                    eligible_voids = eligible_voids[:max_regions_per_level]

                for void_rank, (void_count, void_label, void_slices) in enumerate(eligible_voids):
                    starts = np.asarray([item.start for item in void_slices], dtype=np.int64)
                    stops = np.asarray([item.stop for item in void_slices], dtype=np.int64)
                    local_labels = void_labels[void_slices]
                    local_mask = local_labels == void_label
                    local_shape = local_mask.shape
                    descriptor_flat = _sample_component_flat_indices(
                        local_mask,
                        void_count,
                        descriptor_sample_voxels,
                        f"{molecule_id}|bounded_void|{q:.8f}|{void_label}|descriptor",
                    )
                    descriptor_sample_count = len(descriptor_flat)
                    local_i, local_j, local_k = np.unravel_index(
                        descriptor_flat, local_shape
                    )
                    descriptor_indices = np.column_stack((local_i, local_j, local_k)) + starts
                    center_local = ndimage.center_of_mass(local_mask)
                    exact_index_center = np.asarray(center_local, dtype=float) + starts
                    containment_flat_local = _sample_component_flat_indices(
                        local_mask,
                        void_count,
                        containment_sample_voxels,
                        f"{molecule_id}|bounded_void|{q:.8f}|{void_label}|relation",
                    )
                    containment_sample_count = len(containment_flat_local)
                    contain_i, contain_j, contain_k = np.unravel_index(
                        containment_flat_local, local_shape
                    )
                    flat_indices = (
                        (contain_i + starts[0]) * (grid.shape[1] * grid.shape[2])
                        + (contain_j + starts[1]) * grid.shape[2]
                        + (contain_k + starts[2])
                    ).astype(np.int64, copy=False)
                    descriptor, raw_quantiles, centroid, extents = _component_descriptor(
                        finite, descriptor_indices, exact_index_center, origin, vectors,
                        robust_center, robust_scale, "sublevel",
                    )
                    candidate_id = (
                        f"{molecule_id}:bounded_sublevel_void:q{q:.5f}:c{void_rank:05d}"
                    )
                    bounds = tuple(
                        (int(starts[i]), int(stops[i] - 1)) for i in range(3)
                    )
                    void_candidate = FieldRegionCandidate(
                        candidate_id=candidate_id,
                        molecule_id=molecule_id,
                        polarity="sublevel",
                        region_family="bounded_complement_void",
                        quantile=float(q),
                        threshold=threshold,
                        centroid=centroid,
                        voxel_count=void_count,
                        physical_volume=float(void_count * voxel_volume),
                        index_bounds=bounds,
                        principal_extents=extents,
                        raw_field_quantiles=raw_quantiles,
                        invariant_descriptor=tuple(float(x) for x in descriptor),
                        descriptor_sample_count=int(descriptor_sample_count),
                        descriptor_is_sampled=descriptor_sample_count < void_count,
                        touches_grid_boundary=False,
                        _flat_voxel_indices=flat_indices,
                    )
                    candidates.append(void_candidate)

                    # Identify the surrounding high-field region using the
                    # fraction of the cavity boundary adjacent to each
                    # superlevel component.  A cavity can be surrounded by a
                    # mixture of components; retain only an unambiguous link.
                    pad_starts = np.maximum(starts - 1, 0)
                    pad_stops = np.minimum(stops + 1, np.asarray(grid.shape))
                    pad_slices = tuple(
                        slice(int(pad_starts[i]), int(pad_stops[i])) for i in range(3)
                    )
                    padded_void = void_labels[pad_slices] == void_label
                    padded_neighbors = ndimage.binary_dilation(
                        padded_void, structure=_CONNECTIVITY_26
                    ) & ~padded_void
                    surrounding_labels = labels[pad_slices][padded_neighbors]
                    surrounding_labels = surrounding_labels[surrounding_labels > 0]
                    if not len(surrounding_labels):
                        continue
                    surround_counts = np.bincount(surrounding_labels)
                    parent_label = int(np.argmax(surround_counts))
                    adjacent_count = int(surround_counts[parent_label])
                    boundary_fraction = adjacent_count / max(len(surrounding_labels), 1)
                    parent = label_to_candidate.get(parent_label)
                    if parent is not None and boundary_fraction >= containment_min_fraction:
                        relations.append(FieldPartWholeRelation(
                            child_candidate_id=void_candidate.candidate_id,
                            parent_candidate_id=parent.candidate_id,
                            relation_kind="bounded_void_within_excursion",
                            relation_fraction=float(boundary_fraction),
                            scale_ratio=float(parent.voxel_count / max(void_count, 1)),
                            relation_sample_count=int(len(surrounding_labels)),
                            relation_is_sampled=False,
                        ))
            previous_level_candidates = current_level

    lower_quantiles = [q for q in quantiles if q <= 0.50]
    upper_quantiles = [q for q in quantiles if q >= 0.50]
    process_polarity("sublevel", sorted(lower_quantiles))
    process_polarity("superlevel", sorted(upper_quantiles, reverse=True))

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    repeated_groups = _repeated_child_groups(
        candidates_by_id, relations, repeated_part_distance, min_part_scale_ratio
    )
    diagnostics["candidate_count"] = len(candidates)
    diagnostics["part_whole_relation_count"] = len(relations)
    diagnostics["repeated_part_group_count"] = len(repeated_groups)
    parent_groups: Dict[str, List[Dict[str, Any]]] = {}
    child_groups: Dict[str, List[str]] = {}
    for group in repeated_groups:
        parent_groups.setdefault(group["parent_candidate_id"], []).append(group)
        for child_id in group["child_candidate_ids"]:
            child_groups.setdefault(child_id, []).append(group["parent_candidate_id"])
    role_hypotheses = []
    for candidate_id, groups in parent_groups.items():
        role_hypotheses.append({
            "candidate_id": candidate_id,
            "role_hypothesis": "possible_composite_or_higher_order_candidate",
            "supporting_repeated_part_group_count": len(groups),
            "interpretation": "structural_hypothesis_only_not_a_letter_syllable_or_word_verdict",
        })
    for candidate_id, parent_ids in child_groups.items():
        role_hypotheses.append({
            "candidate_id": candidate_id,
            "role_hypothesis": "possible_reusable_subpattern",
            "observed_in_parent_candidate_ids": sorted(set(parent_ids)),
            "interpretation": "structural_hypothesis_only_not_a_primitive_letter_verdict",
        })
    diagnostics["candidate_role_hypotheses"] = role_hypotheses
    diagnostics["bounded_complement_void_candidate_count"] = sum(
        candidate.region_family == "bounded_complement_void" for candidate in candidates
    )
    diagnostics["clipped_regions_are_not_interpreted_as_absent_patterns"] = True

    # Do not retain large per-voxel index arrays in the returned archive.
    for candidate in candidates:
        candidate._flat_voxel_indices = np.empty(0, dtype=np.int64)

    return FieldPatternBatch(
        molecule_id=molecule_id,
        field_kind=str(getattr(field, "field_kind", "unspecified")),
        level_of_theory=str(getattr(field, "level_of_theory", "unspecified")),
        environment=str(getattr(field, "environment", "unspecified")),
        grid_shape=tuple(int(x) for x in grid.shape),
        origin=tuple(float(x) for x in origin),
        grid_vectors=tuple(tuple(float(x) for x in row) for row in vectors),
        thresholds=thresholds_record,
        candidates=candidates,
        part_whole_relations=relations,
        repeated_part_groups=repeated_groups,
        diagnostics=diagnostics,
    )


def write_field_pattern_archive(
    batch: FieldPatternBatch,
    output_dir: str | Path,
    *,
    source_cube: str | Path | None = None,
    cache_signature: str = "unversioned",
) -> Path:
    """Atomically write one compressed, resumable per-molecule archive."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch.molecule_id).strip("._")
    if not safe_id:
        raise ValueError("molecule_id cannot produce an empty archive filename")
    shard = hashlib.sha256(batch.molecule_id.encode("utf-8")).hexdigest()[:2]
    shard_dir = root / shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    destination = shard_dir / f"{safe_id}.json.gz"
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    payload = batch.to_json_safe()
    payload["cache_signature"] = str(cache_signature)
    payload["source_cube"] = str(source_cube) if source_cube is not None else None
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.replace(temporary, destination)
    return destination
