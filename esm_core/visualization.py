"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Portable visualisation artefacts for concrete ESM letter realisations.

A ``LetterClass`` is an equivalence class in invariant-descriptor space; it
does not by itself have an oriented 3-D density grid.  This module therefore
exports *realisations* (a critical point in a particular molecule) together
with its class ID and original density cube.  It never averages unaligned
local grids and calls the result a class field.

The output is deliberately simple JSON plus Gaussian-cube files, so it can be
consumed by the bundled Streamlit/py3Dmol viewer, Cube Suite-like tools, or an
independent visualisation stack.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .corpus import LetterCandidate
from .fields import CubeFileField
from .schema import LetterClass


VISUALIZATION_SCHEMA_VERSION = "esm-letter-scenes/v2"

# Fixed type colours make the QTAIM topology readable before a user learns
# arbitrary class IDs. Class colours are additionally deterministic below.
TYPE_COLOURS = {
    "nuclear_attractor": "#f6bd16",
    "bond_critical_point": "#20a4f3",
    "ring_critical_point": "#b65cff",
    "cage_critical_point": "#00b894",
    "degenerate": "#8c8c8c",
    "unassigned_outlier": "#e74c3c",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def colour_for_class(class_id: str) -> str:
    """Stable high-contrast RGB colour for an individual class ID."""
    digest = hashlib.sha1(class_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.92)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def local_patch_relative_path(molecule_id: str, global_candidate_index: int) -> Path:
    """Stable bundle-relative name for an on-demand local field cube."""
    return Path("patches") / f"{_safe_file_stem(molecule_id)}__{global_candidate_index:08d}.cube"


def _nearest_atom_reference(
    position: Sequence[float], atoms: Sequence[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Return a continuous post-hoc spatial reference, never a label.

    The atom block in a cube is useful to *compare* a field realisation to a
    conventional molecular drawing.  This helper intentionally reports only
    a nearest-reference distance and atom metadata; it has no threshold and
    is not used by extraction, descriptors, or clustering.
    """
    if not atoms:
        return None
    point = np.asarray(position, dtype=float)
    coordinates = np.asarray([atom["position"] for atom in atoms], dtype=float)
    distances = np.linalg.norm(coordinates - point, axis=1)
    index = int(np.argmin(distances))
    atom = atoms[index]
    return {
        "atom_index": index,
        "atomic_number": int(atom["atomic_number"]),
        "distance": float(distances[index]),
        "coordinate_units": "cube_native_units",
        "role": "post_hoc_reference_only",
    }


def read_cube_header(cube_path: str | Path) -> Dict[str, Any]:
    """Read only atom/grid metadata, without loading the volumetric values."""
    path = Path(cube_path)
    with path.open(encoding="utf-8") as handle:
        comment_1 = handle.readline().rstrip("\n")
        comment_2 = handle.readline().rstrip("\n")
        header = handle.readline().split()
        if len(header) < 4:
            raise ValueError(f"invalid cube header in {path}")
        n_atoms = abs(int(header[0]))
        origin = [float(v) for v in header[1:4]]
        dims: List[int] = []
        vectors: List[List[float]] = []
        for _ in range(3):
            parts = handle.readline().split()
            if len(parts) < 4:
                raise ValueError(f"invalid cube grid line in {path}")
            dims.append(abs(int(parts[0])))
            vectors.append([float(v) for v in parts[1:4]])
        atoms = []
        for _ in range(n_atoms):
            parts = handle.readline().split()
            if len(parts) < 5:
                raise ValueError(f"invalid cube atom line in {path}")
            atoms.append({
                "atomic_number": int(float(parts[0])),
                "nuclear_charge": float(parts[1]),
                "position": [float(v) for v in parts[2:5]],
            })
    return {
        "comment_1": comment_1,
        "comment_2": comment_2,
        "origin": origin,
        "dims": dims,
        "vectors": vectors,
        "atoms": atoms,
        # Gaussian cube coordinates share the grid unit. PySCF writes Bohr;
        # retain native units rather than silently relabelling them Å.
        "coordinate_units": "cube_native_units",
    }


def extract_local_field_cube(
    source_cube: str | Path,
    center: Sequence[float],
    destination_cube: str | Path,
    radius: float,
) -> Dict[str, Any]:
    """Write a local cube window around one actual letter realisation.

    ``radius`` is expressed in the cube coordinate unit.  The window is a
    conservative Cartesian-radius bounding box on a possibly non-orthogonal
    cube grid.  Atom records are retained so the extracted field can be
    displayed in molecular context.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    field = CubeFileField(str(source_cube), field_kind="density")
    point = np.asarray(center, dtype=float)
    inverse_vectors = np.linalg.inv(field.vectors)
    fractional_center = (point - field.origin) @ inverse_vectors
    # Component i of delta @ inverse_vectors is dot(delta, column i), so the
    # largest possible magnitude within a radius-r ball is r*||column i||.
    index_radius = radius * np.linalg.norm(inverse_vectors, axis=0)
    low = np.maximum(0, np.floor(fractional_center - index_radius).astype(int))
    high = np.minimum(np.asarray(field.dims) - 1, np.ceil(fractional_center + index_radius).astype(int))
    if np.any(high < low):
        raise ValueError("local cube window does not intersect the source grid")

    values = field.grid[
        low[0]: high[0] + 1,
        low[1]: high[1] + 1,
        low[2]: high[2] + 1,
    ]
    origin = field.origin + low @ field.vectors
    destination = Path(destination_cube)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("ESM local letter-realisation field\n")
        handle.write(f"source={Path(source_cube).name}; center={point.tolist()}; radius={radius}\n")
        handle.write(
            f"{len(field.atomic_numbers):5d} {origin[0]: .8f} {origin[1]: .8f} {origin[2]: .8f}\n"
        )
        for size, vector in zip(values.shape, field.vectors):
            handle.write(f"{size:5d} {vector[0]: .8f} {vector[1]: .8f} {vector[2]: .8f}\n")
        for atomic_number, position in zip(field.atomic_numbers, field.atom_positions):
            handle.write(
                f"{int(atomic_number):5d} {float(atomic_number): .6f} "
                f"{position[0]: .8f} {position[1]: .8f} {position[2]: .8f}\n"
            )
        flat = values.ravel()
        for start in range(0, len(flat), 6):
            handle.write(" ".join(f"{value: .5E}" for value in flat[start:start + 6]) + "\n")

    return {
        "path": str(destination),
        "origin": origin.tolist(),
        "dims": list(values.shape),
        "vectors": field.vectors.tolist(),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "coordinate_units": "cube_native_units",
    }

def plane_through_points(
    source_cube: str | Path,
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
    *,
    half_extent: Optional[float] = None,
    n_samples: int = 96,
) -> Dict[str, Any]:
    """Trilinearly resample density on the arbitrary plane through three points.

    Unlike ``field_slices_through_point`` (grid-index-aligned, nearest-voxel),
    an atom-defined plane is generally not aligned to any grid axis, so this
    interpolates via ``CubeFileField.value`` at each sampled point instead of
    indexing the grid directly.

    The in-plane basis is built from the three points themselves: ``u_axis``
    runs from ``point_a`` to ``point_b`` (so the first two picks set the
    cross-section's horizontal direction), ``normal`` is
    ``(b-a) x (c-a)`` normalised, and ``v_axis`` completes a right-handed
    orthonormal frame. The sampling window is centred on the centroid of the
    three points, not on ``point_a``, so the result does not privilege
    whichever point was picked first.

    All inputs and outputs are in the cube's native coordinate unit (Bohr for
    a PySCF density cube) -- the same convention as
    ``field_slices_through_point`` and ``extract_local_field_cube``. Callers
    that work in Angstrom (e.g. a viewer UI) must convert before calling and
    after reading the result.

    Raises ``ValueError`` if the three points are collinear, coincide, or
    ``point_a == point_b`` (the u-axis would be undefined).
    """
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)

    edge_ab = b - a
    u_norm = float(np.linalg.norm(edge_ab))
    if u_norm < 1e-6:
        raise ValueError("the first two points coincide; they cannot set the in-plane axis")
    u_hat = edge_ab / u_norm

    normal = np.cross(edge_ab, c - a)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-6:
        raise ValueError("the three points are collinear (or coincide); they do not define a plane")
    normal_hat = normal / normal_norm
    v_hat = np.cross(normal_hat, u_hat)  # unit length: normal_hat, u_hat are already orthonormal

    centroid = (a + b + c) / 3.0

    if half_extent is None:
        pairwise = (u_norm, float(np.linalg.norm(c - a)), float(np.linalg.norm(c - b)))
        # Enough margin that all three defining atoms sit inside the frame
        # rather than right on its edge.
        half_extent = 0.65 * max(pairwise) + 1.0

    field = CubeFileField(str(source_cube), field_kind="density")
    axis = np.linspace(-float(half_extent), float(half_extent), int(n_samples))
    density = np.empty((axis.size, axis.size), dtype=float)
    for i, u in enumerate(axis):
        offset_u = centroid + u * u_hat
        for j, v in enumerate(axis):
            density[i, j] = field.value(offset_u + v * v_hat)

    return {
        "source": str(source_cube),
        "centroid": centroid.tolist(),
        "normal": normal_hat.tolist(),
        "u_axis": u_hat.tolist(),
        "v_axis": v_hat.tolist(),
        "u_coords": axis.tolist(),
        "v_coords": axis.tolist(),
        "half_extent": float(half_extent),
        # density[i, j] corresponds to (u_coords[i], v_coords[j]); kept as an
        # ndarray like field_slices_through_point's "slices", not pre-flattened
        # for JSON, since the intended consumer plots it directly.
        "density": density,
        "minimum": float(np.min(density)),
        "maximum": float(np.max(density)),
        "quantiles": {
            str(q): float(np.quantile(density, q)) for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.98)
        },
        "defining_points": {"a": a.tolist(), "b": b.tolist(), "c": c.tolist()},
        "coordinate_units": "cube_native_units",
        "note": (
            "Trilinearly interpolated density on an arbitrary plane through three chosen points. "
            "Points outside the source cube's grid extent are clamped to its boundary, matching "
            "CubeFileField.value's behaviour."
        ),
    }

def field_slices_through_point(
    source_cube: str | Path,
    center: Sequence[float],
) -> Dict[str, Any]:
    """Return three sampled density sections through a field realisation.

    These are scalar voxel values, not a geometric boundary.  They make the
    internal maxima, minima, saddle-like changes and void structure visible
    even when a 3-D isosurface happens to look like a hollow rigid shell.
    """
    field = CubeFileField(str(source_cube), field_kind="density")
    point = np.asarray(center, dtype=float)
    fractional = (point - field.origin) @ np.linalg.inv(field.vectors)
    index = np.clip(np.rint(fractional).astype(int), 0, np.asarray(field.dims) - 1)
    grid = np.asarray(field.grid, dtype=float)
    return {
        "source": str(source_cube),
        "grid_index": index.tolist(),
        "position": point.tolist(),
        "minimum": float(np.min(grid)),
        "maximum": float(np.max(grid)),
        "quantiles": {
            str(q): float(np.quantile(grid, q)) for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.98)
        },
        "slices": {
            "axis_0": grid[index[0], :, :],
            "axis_1": grid[:, index[1], :],
            "axis_2": grid[:, :, index[2]],
        },
        "coordinate_units": "cube_native_units",
        "note": "Index-aligned grid sections; use only as a view of sampled field values.",
    }


def basin_boundary_relative_path(molecule_id: str) -> Path:
    """Bundle-relative cache name for a classical basin-boundary overlay."""
    return Path("basins") / f"{_safe_file_stem(molecule_id)}__gradient_basin_boundaries.cube"


def field_component_boundary_relative_path(
    molecule_id: str,
    global_candidate_index: int,
    side: str,
    level_fraction: float,
    *,
    source_scope: str = "local",
) -> Path:
    """Cache name for a threshold-defined component around one field root."""
    if side not in {"above", "below"}:
        raise ValueError("side must be 'above' or 'below'")
    if not 0.0 < float(level_fraction) < 1.0:
        raise ValueError("level_fraction must be between 0 and 1")
    side_key = side
    scope_key = "full" if source_scope == "full" else "local"
    fraction_pct = int(round(float(level_fraction) * 100.0))
    return Path("boundaries") / (
        f"{_safe_file_stem(molecule_id)}__{int(global_candidate_index):08d}"
        f"__{scope_key}_{side_key}_{fraction_pct:02d}.cube"
    )


def _write_cube_values(
    field: CubeFileField,
    values: np.ndarray,
    origin: np.ndarray,
    destination: Path,
    *,
    comment_1: str,
    comment_2: str,
) -> None:
    """Write a cube sharing ``field``'s atom block and grid vectors."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(comment_1 + "\n")
        handle.write(comment_2 + "\n")
        handle.write(
            f"{len(field.atomic_numbers):5d} {origin[0]: .8f} {origin[1]: .8f} {origin[2]: .8f}\n"
        )
        for size, vector in zip(values.shape, field.vectors):
            handle.write(f"{size:5d} {vector[0]: .8f} {vector[1]: .8f} {vector[2]: .8f}\n")
        for atomic_number, position in zip(field.atomic_numbers, field.atom_positions):
            handle.write(
                f"{int(atomic_number):5d} {float(atomic_number): .6f} "
                f"{position[0]: .8f} {position[1]: .8f} {position[2]: .8f}\n"
            )
        flat = values.ravel()
        for start in range(0, len(flat), 6):
            handle.write(" ".join(f"{value: .5E}" for value in flat[start:start + 6]) + "\n")


def _snap_to_density_peak(
    density: np.ndarray,
    requested_index: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    """Snap a continuous attractor to a local sampled maximum, not a voxel.

    A field root usually lies between cube samples.  Rounding it straight to
    an arbitrary neighbouring voxel can make a correct attractor appear to
    be a non-maximum and starts the watershed from the wrong place.  This
    small, deterministic 3x3x3 refinement is a grid-resolution diagnostic,
    not an atom-derived correction.
    """
    requested = np.asarray(requested_index, dtype=int)
    lower = np.maximum(0, requested - 1)
    upper = np.minimum(np.asarray(density.shape) - 1, requested + 1)
    block = density[
        lower[0]:upper[0] + 1,
        lower[1]:upper[1] + 1,
        lower[2]:upper[2] + 1,
    ]
    maximum = float(np.max(block))
    local_indices = np.argwhere(block == maximum)
    # Plateau ties are resolved by closest-to-root then lexicographic index;
    # this keeps re-renders stable without pretending a plateau is a sharp
    # resolved nuclear maximum.
    absolute = local_indices + lower
    distances = np.sum((absolute - requested) ** 2, axis=1)
    snapped = absolute[int(np.argmin(distances))]
    at_grid_edge = bool(np.any(snapped == 0) or np.any(snapped == np.asarray(density.shape) - 1))
    return snapped, bool(len(local_indices) == 1 and not at_grid_edge), int(len(local_indices))


def validate_density_attractors(
    source_cube: str | Path,
    nuclear_attractor_positions: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Audit field-derived attractors before using them as basin markers.

    The audit has deliberately modest claims: it tests only whether a root is
    inside the sampled cube and is represented by a nearby local *density*
    peak on that grid.  Hessian/gradient estimates are reported as resolution
    diagnostics, never used to inject an atom position or certify QTAIM.
    """
    field = CubeFileField(str(source_cube), field_kind="density")
    density = np.asarray(field.grid, dtype=np.float64)
    inverse_vectors = np.linalg.inv(field.vectors)
    rows: List[Dict[str, Any]] = []
    for position in nuclear_attractor_positions:
        point = np.asarray(position, dtype=float)
        fractional = (point - field.origin) @ inverse_vectors
        rounded = np.rint(fractional).astype(int)
        if np.any(rounded < 0) or np.any(rounded >= np.asarray(density.shape)):
            rows.append({
                "position": point.tolist(), "inside_grid": False,
                "accepted_for_watershed": False,
            })
            continue
        snapped, strict_peak, plateau_size = _snap_to_density_peak(density, rounded)
        snapped_position = field.origin + snapped @ field.vectors
        try:
            gradient_norm = float(np.linalg.norm(field.gradient(snapped_position)))
            eigvals = np.linalg.eigvalsh(field.hessian(snapped_position))
            hessian_negative_definite = bool(np.all(eigvals < 0.0))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            gradient_norm, eigvals, hessian_negative_definite = None, None, False
        rows.append({
            "position": point.tolist(),
            "inside_grid": True,
            "rounded_grid_index": rounded.tolist(),
            "snapped_grid_index": snapped.tolist(),
            "snapped_position": np.asarray(snapped_position, dtype=float).tolist(),
            "snap_distance": float(np.linalg.norm(snapped_position - point)),
            "sampled_density": float(density[tuple(snapped)]),
            "strict_3x3x3_grid_peak": strict_peak,
            "peak_plateau_voxels": plateau_size,
            "gradient_norm_at_snapped_voxel": gradient_norm,
            "hessian_eigenvalues_at_snapped_voxel": (
                np.asarray(eigvals, dtype=float).tolist() if eigvals is not None else None
            ),
            "hessian_negative_definite_at_grid_scale": hessian_negative_definite,
            # A plateau may be an unresolved maximum; it remains usable as a
            # marker but visibly lowers confidence in the overlay.
            "accepted_for_watershed": True,
        })
    return {
        "method": "field_root_to_nearby_density_grid_peak_audit",
        "coordinate_units": "cube_native_units",
        "n_input": len(nuclear_attractor_positions),
        "n_accepted": sum(bool(row.get("accepted_for_watershed")) for row in rows),
        "attractors": rows,
        "limitations": [
            "A cube grid only resolves extrema to its finite spacing.",
            "Finite-difference Hessian signs are diagnostics at the cube scale, not a convergence proof.",
            "No atomic coordinate is used by this audit or by marker placement.",
        ],
    }


def extract_gradient_basin_boundaries(
    source_cube: str | Path,
    nuclear_attractor_positions: Sequence[Sequence[float]],
    destination_cube: str | Path,
) -> Dict[str, Any]:
    """Cache a discrete gradient-basin boundary overlay for one molecule.

    This is a *visual reference* for the classical QTAIM atom/basin picture:
    a marker-controlled topographic watershed over ``-rho`` starts from
    independently field-detected attractors (snapped to nearby grid peaks),
    and interfaces between labels are emitted as a binary cube.  It is not a
    direct integration of gradient paths and not a certified zero-flux
    surface.  The function deliberately keeps that distinction explicit
    while avoiding a misleading nearest-atom Voronoi diagram.
    """
    if len(nuclear_attractor_positions) < 2:
        raise ValueError("at least two nuclear attractors are needed for a basin-boundary overlay")
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:  # pragma: no cover - SciPy is an environment dependency
        raise RuntimeError("SciPy is required for the gradient-basin overlay") from exc

    field = CubeFileField(str(source_cube), field_kind="density")
    density = np.asarray(field.grid, dtype=np.float64)
    if density.ndim != 3 or not np.all(np.isfinite(density)):
        raise ValueError("basin overlay requires a finite 3-D density cube")

    audit = validate_density_attractors(source_cube, nuclear_attractor_positions)
    markers = np.zeros(density.shape, dtype=np.int32)
    retained_seeds = []
    for row in audit["attractors"]:
        if not row.get("accepted_for_watershed"):
            continue
        index = np.asarray(row["snapped_grid_index"], dtype=int)
        index_tuple = tuple(index.tolist())
        # Several numerical roots can resolve to the same grid maximum. They
        # are one discrete basin at the cube's current resolution, not an
        # excuse to fabricate a zero-width basin boundary between duplicates.
        if markers[index_tuple] != 0:
            continue
        markers[index_tuple] = len(retained_seeds) + 1
        retained_seeds.append({
            "position": row["position"],
            "grid_index": index.tolist(),
            "snapped_position": row["snapped_position"],
            "snap_distance": row["snap_distance"],
            "strict_3x3x3_grid_peak": row["strict_3x3x3_grid_peak"],
        })
    if len(retained_seeds) < 2:
        raise ValueError("fewer than two distinct nuclear-attractor seeds fall inside the cube grid")

    # Watershed floods minima.  `-log(rho - min(rho) + 1)` preserves density
    # order.  uint16 (rather than the former uint8) retains substantially
    # more topographic ordering between sharp cores and low-density voids;
    # the log transform only compresses dynamic range for the IFT input.
    shifted = density - float(np.min(density))
    topography = -np.log1p(np.maximum(shifted, 0.0))
    lo, hi = float(np.min(topography)), float(np.max(topography))
    if hi > lo:
        topography_u16 = np.rint((topography - lo) * (65535.0 / (hi - lo))).astype(np.uint16)
    else:
        topography_u16 = np.zeros(density.shape, dtype=np.uint16)
    basin_labels = ndimage.watershed_ift(topography_u16, markers)

    boundary = np.zeros(density.shape, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        different = basin_labels[tuple(left)] != basin_labels[tuple(right)]
        boundary[tuple(left)] |= different
        boundary[tuple(right)] |= different
    destination = Path(destination_cube)
    _write_cube_values(
        field,
        boundary.astype(np.float32),
        field.origin,
        destination,
        comment_1="ESM discrete gradient-basin boundary overlay",
        comment_2=(
            f"source={Path(source_cube).name}; seeds={len(retained_seeds)}; "
            "field-root markers snapped to density peaks; topographic watershed approximation, not a certified zero-flux surface"
        ),
    )
    return {
        "path": str(destination),
        "n_input_nuclear_attractors": len(nuclear_attractor_positions),
        "n_distinct_grid_seeds": len(retained_seeds),
        "seeds": retained_seeds,
        "boundary_voxels": int(np.count_nonzero(boundary)),
        "grid_voxels": int(boundary.size),
        "method": "marker_controlled_topographic_watershed_on_negative_log_density_uint16",
        "zero_flux_surface_certified": False,
        "attractor_validation": audit,
        "coordinate_units": "cube_native_units",
    }


def extract_field_component_boundary(
    source_cube: str | Path,
    center: Sequence[float],
    destination_cube: str | Path,
    *,
    side: str = "above",
    level_fraction: float = 0.15,
) -> Dict[str, Any]:
    """Extract a crisp, field-only boundary for the component containing a root.

    The displayed surface is the boundary of the connected superlevel set
    ``rho >= level`` (``side='above'``) or sublevel set ``rho <= level``
    (``side='below'``) that contains the selected field critical point. The
    level is placed a fraction of the contrast from the corresponding
    cube-wide extreme toward the sampled root value. No atom coordinates or labels are
    used. This is an explicit, reproducible *operational boundary*, not a
    claim that every field-letter has a unique physical border.

    If the selected component touches the cube edge, the returned diagnostic
    marks the surface as clipped: use a larger local patch or the complete
    molecular cube before interpreting its extent.
    """
    if side not in {"above", "below"}:
        raise ValueError("side must be 'above' or 'below'")
    if not 0.0 < float(level_fraction) < 1.0:
        raise ValueError("level_fraction must be between 0 and 1")
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:  # pragma: no cover - SciPy is an environment dependency
        raise RuntimeError("SciPy is required to extract a field-component boundary") from exc

    field = CubeFileField(str(source_cube), field_kind="density")
    values = np.asarray(field.grid, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("field-component boundary requires a finite 3-D cube")

    point = np.asarray(center, dtype=np.float64)
    fractional_index = (point - field.origin) @ np.linalg.inv(field.vectors)
    seed = np.rint(fractional_index).astype(int)
    if np.any(seed < 0) or np.any(seed >= np.asarray(values.shape)):
        raise ValueError("selected field root lies outside the cube grid")

    seed_value = float(values[tuple(seed)])
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if maximum <= minimum:
        raise ValueError("field cube is constant; it has no resolvable boundary")
    if side == "above":
        # A small fraction gives a broad superlevel component; a value near
        # one moves the cut toward the root and yields a tighter envelope.
        level = minimum + float(level_fraction) * (seed_value - minimum)
        component_mask = values >= level
    else:
        # Symmetric rule for a sublevel component around a local minimum.
        level = maximum - float(level_fraction) * (maximum - seed_value)
        component_mask = values <= level

    labels, _ = ndimage.label(component_mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    seed_label = int(labels[tuple(seed)])
    if seed_label == 0:
        raise ValueError("selected root voxel is not inside the requested field component")
    selected_component = labels == seed_label
    voxel_count = int(np.count_nonzero(selected_component))
    touches_edge = bool(
        np.any(selected_component[0, :, :]) or np.any(selected_component[-1, :, :])
        or np.any(selected_component[:, 0, :]) or np.any(selected_component[:, -1, :])
        or np.any(selected_component[:, :, 0]) or np.any(selected_component[:, :, -1])
    )

    destination = Path(destination_cube)
    _write_cube_values(
        field,
        selected_component.astype(np.float32),
        field.origin,
        destination,
        comment_1="ESM field-letter connected-component boundary (binary mask)",
        comment_2=(
            f"source={Path(source_cube).name}; side={side}; level={level:.10g}; "
            f"level_fraction={float(level_fraction):.6g}; seed_index={seed.tolist()}; "
            "field-only threshold component; operational boundary, not a universal physical border"
        ),
    )
    voxel_volume = abs(float(np.linalg.det(field.vectors)))
    return {
        "path": str(destination),
        "method": "connected_density_superlevel_component" if side == "above" else "connected_density_sublevel_component",
        "side": side,
        "level": float(level),
        "level_fraction": float(level_fraction),
        "seed_index": seed.tolist(),
        "sampled_seed_value": seed_value,
        "voxel_count": voxel_count,
        "volume": float(voxel_count * voxel_volume),
        "grid_voxels": int(values.size),
        "touches_grid_boundary": touches_edge,
        "boundary_is_clipped": touches_edge,
        "coordinate_units": "cube_native_units",
        "interpretation": (
            "Explicit field-only connected-component boundary at the recorded level; "
            "it is not asserted to be the unique physical boundary of a letter."
        ),
    }


def write_visualization_bundle(
    *,
    candidates: Sequence[LetterCandidate],
    classes: Sequence[LetterClass],
    labels: Sequence[Optional[str]],
    metadata_registry: Mapping[str, Mapping[str, Any]],
    cube_dir: str | Path,
    output_dir: str | Path,
    class_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    include_molecule_ids: Optional[Sequence[str]] = None,
    analysis_context: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Export one portable scene JSON per molecule and a corpus manifest.

    The caller supplies labels obtained from the same clustering decision rule
    used for the analysis.  That is essential: a visualisation must not
    quietly re-cluster a point merely to assign it a colour.
    """
    if len(candidates) != len(labels):
        raise ValueError("candidates and labels must have identical lengths")

    out_dir = Path(output_dir)
    scene_dir = out_dir / "molecules"
    scene_dir.mkdir(parents=True, exist_ok=True)
    cube_root = Path(cube_dir)
    supplied = class_records or {}
    provenance = dict(analysis_context or {})
    field_first_confirmed = (
        str(provenance.get("candidate_discovery_method", "")).startswith("blind_scalar_field_")
        and provenance.get("atomic_coordinates_used_to_discover_candidates") is False
    )
    scene_molecule_ids = set(include_molecule_ids) if include_molecule_ids is not None else None
    by_class = {item.class_id: item for item in classes}
    occurrences_by_molecule: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    class_occurrences: Counter[str] = Counter()
    class_molecules: Dict[str, set[str]] = defaultdict(set)
    class_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    class_example_molecules: Dict[str, set[str]] = defaultdict(set)

    for global_index, (candidate, class_id) in enumerate(zip(candidates, labels)):
        if scene_molecule_ids is not None and candidate.source_molecule_id not in scene_molecule_ids:
            continue
        field_type = candidate.critical_point.field_type
        assigned_class_id = class_id or "unassigned_outlier"
        class_item = by_class.get(class_id) if class_id else None
        descriptor_distance = None
        if class_item is not None:
            descriptor_distance = float(np.linalg.norm(
                candidate.descriptor.to_vector() - class_item.invariant_centroid
            ))
        occurrence = {
            "occurrence_id": f"{candidate.source_molecule_id}:{global_index}",
            "global_candidate_index": global_index,
            "class_id": assigned_class_id,
            "critical_point_type": field_type,
            "position": np.asarray(candidate.critical_point.position, dtype=float).tolist(),
            "field_value": float(candidate.critical_point.value),
            "hessian_eigenvalues": np.asarray(candidate.critical_point.eigenvalues, dtype=float).tolist(),
            "gradient_norm_at_field_root": getattr(candidate.critical_point, "gradient_norm", None),
            "field_discovery_stage": getattr(candidate.critical_point, "discovery_stage", "unknown"),
            "rank": int(candidate.critical_point.rank),
            "signature": int(candidate.critical_point.signature),
            "poincare_hopf_index": candidate.critical_point.poincare_hopf_index,
            "descriptor_distance_to_class_centroid": descriptor_distance,
            "descriptor_support": {
                "shell_radii": [float(radius) for radius in candidate.descriptor.shell_radii],
                "outer_radius": float(max(candidate.descriptor.shell_radii)),
                "coordinate_units": "cube_native_units",
                "meaning": (
                    "Radial support sampled by this local invariant descriptor; "
                    "not a basin boundary or a claim of the letter's physical volume."
                ),
            },
            "type_colour": TYPE_COLOURS.get(field_type, "#ffffff"),
            "class_colour": colour_for_class(assigned_class_id),
            "local_field_cube": local_patch_relative_path(
                candidate.source_molecule_id, global_index
            ).as_posix(),
        }
        occurrences_by_molecule[candidate.source_molecule_id].append(occurrence)
        if class_id is not None:
            class_occurrences[class_id] += 1
            class_molecules[class_id].add(candidate.source_molecule_id)
            # Examples are navigation targets, so prefer twelve *different*
            # molecules over twelve symmetry-related occurrences in one
            # molecule.  All occurrences stay in the per-molecule scene.
            if (
                candidate.source_molecule_id not in class_example_molecules[class_id]
                and len(class_examples[class_id]) < 12
            ):
                class_example_molecules[class_id].add(candidate.source_molecule_id)
                class_examples[class_id].append({
                    "molecule_id": candidate.source_molecule_id,
                    "occurrence_id": occurrence["occurrence_id"],
                    "position": occurrence["position"],
                    "descriptor_distance_to_class_centroid": descriptor_distance,
                })

    manifest_classes: List[Dict[str, Any]] = []
    for class_id, letter_class in sorted(by_class.items()):
        record = dict(supplied.get(class_id, {}))
        record.update({
            "class_id": class_id,
            "critical_point_type": letter_class.discovery_provenance.critical_point_type,
            "colour": colour_for_class(class_id),
            "type_colour": TYPE_COLOURS.get(letter_class.discovery_provenance.critical_point_type, "#ffffff"),
            "example_realisations": class_examples[class_id],
            "n_scene_occurrences": int(class_occurrences[class_id]),
            "n_scene_molecules": len(class_molecules[class_id]),
        })
        # A representative-only visualisation must not turn a class's full
        # corpus reuse count into its smaller scene-subset count.  When the
        # caller has no report record (the legacy exporter), the scene is the
        # whole selected corpus and supplies the correct fallback.
        if record.get("n_occurrences") is None:
            record["n_occurrences"] = int(class_occurrences[class_id])
        if record.get("n_molecules") is None:
            record["n_molecules"] = len(class_molecules[class_id])
        manifest_classes.append(_json_safe(record))

    molecule_entries: List[Dict[str, Any]] = []
    all_molecule_ids = (
        sorted(scene_molecule_ids)
        if scene_molecule_ids is not None
        else sorted(set(metadata_registry) | set(occurrences_by_molecule))
    )
    for molecule_id in all_molecule_ids:
        cube_name = f"{molecule_id}_density.cube"
        cube_path = cube_root / cube_name
        header = read_cube_header(cube_path) if cube_path.exists() else None
        scene = {
            "schema_version": VISUALIZATION_SCHEMA_VERSION,
            "molecule_id": molecule_id,
            "cube_file": cube_name if cube_path.exists() else None,
            "cube_header": header,
            "metadata": _json_safe(dict(metadata_registry.get(molecule_id, {}))),
            "letters": occurrences_by_molecule.get(molecule_id, []),
            "basin_note": (
                "Nuclear-attractor markers identify density attractors. "
                "They are not exported QTAIM zero-flux basin surfaces."
            ),
            "atom_overlay_role": (
                "Atoms are a post-hoc reference layer from the cube header. They do not "
                "define candidate positions, types, descriptors, or class assignments."
            ),
            "candidate_discovery_provenance": _json_safe(provenance),
        }
        atoms = (header or {}).get("atoms", [])
        for occurrence in scene["letters"]:
            occurrence["nearest_atom_reference"] = _nearest_atom_reference(
                occurrence["position"], atoms
            )
        scene_relpath = Path("molecules") / f"{_safe_file_stem(molecule_id)}.json"
        with (out_dir / scene_relpath).open("w", encoding="utf-8") as handle:
            json.dump(scene, handle, ensure_ascii=False, indent=2)
        molecule_entries.append({
            "molecule_id": molecule_id,
            "scene_file": scene_relpath.as_posix(),
            "cube_file": cube_name if cube_path.exists() else None,
            "n_letter_realisations": len(scene["letters"]),
            "smiles": str(metadata_registry.get(molecule_id, {}).get("smiles", "") or ""),
        })

    manifest = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "coordinate_units": "cube_native_units",
        "cube_dir_hint": Path(os.path.relpath(cube_root, out_dir)).as_posix(),
        "type_colours": TYPE_COLOURS,
        "candidate_discovery_provenance": _json_safe(provenance),
        "letter_classes": manifest_classes,
        "molecules": molecule_entries,
        "scene_scope": "all_molecules" if scene_molecule_ids is None else "representative_subset",
        "notes": [
            (
                "Each marker is a field-discovered critical-point realisation assigned by the analysed alphabet."
                if field_first_confirmed else
                "Candidate-discovery provenance is legacy/unknown; this bundle must not be read as proof of a field-first extraction."
            ),
            "Atom positions are visual/reference annotations only. A nearest-atom distance is continuous post-hoc evidence, not a letter label or matching threshold.",
            "ring_critical_point and cage_critical_point markers deliberately remain visible away from nuclei; they are not discarded because they occupy an apparent void.",
            "A local cube is a field realisation around one occurrence, not an oriented average field for an invariant class.",
            "The descriptor-support shell shows the sampled local field neighbourhood, not a QTAIM basin or the physical volume of an invariant class.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + f".{os.getpid()}.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    # The viewer sees either the complete preceding snapshot or the complete
    # newly written one -- never a truncated JSON file after interruption.
    os.replace(temporary_manifest, manifest_path)
    return manifest_path
