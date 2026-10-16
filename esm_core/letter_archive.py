"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Portable, lossless-on-disk archive for ESM letter analysis.

Visual scenes are intentionally small: they make a discovered letter visible
in its molecule.  They are not an adequate research record because a later
analysis may need the complete invariant vector, the exact class centroid or
the original QTAIM critical-point data.  This module writes those data once,
without re-clustering and without keeping a second dense matrix in RAM.

The archive is deliberately ordinary JSONL + ``.npy`` files:

* ``class_centroids.npy``: float64 rows indexed by ``classes.jsonl``;
* ``descriptors/*.npy``: float64, memory-mappable occurrence-vector shards;
* ``occurrences.jsonl``: one critical-point realisation per line, including
  its class assignment and a pointer into its descriptor shard;
* ``molecules.jsonl``: source/cube references and all registered molecular
  metadata; and
* ``manifest.json``: schema, dimensions, shard boundaries and provenance.

The original Gaussian cube is retained as the authoritative complete scalar
field.  It is referenced, rather than duplicated for every critical point;
local field cubes can be extracted from it later for any occurrence,
including a ring/cage point in an apparent void.
"""

from __future__ import annotations

import json
import os
import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import numpy as np

from .corpus import LetterCandidate
from .schema import LetterClass
from .visualization import local_patch_relative_path


LETTER_ARCHIVE_SCHEMA_VERSION = "esm-letter-archive/v2"


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _descriptor_layout(candidate: LetterCandidate) -> Dict[str, Any]:
    """Persist the exact `InvariantDescriptor.to_vector()` component order."""
    descriptor = candidate.descriptor
    if hasattr(descriptor, "component_layout"):
        return {
            "l_max": int(descriptor.l_max),
            "shell_radii": [float(value) for value in descriptor.shell_radii],
            "feature_regime": getattr(descriptor, "regime", "local_field"),
            "components": descriptor.component_layout(),
        }
    power_keys = sorted(descriptor.power_spectrum)
    bispectrum_keys = sorted(descriptor.bispectrum)
    components = []
    components.extend({"kind": "power_diagonal_real", "key": list(key)} for key in power_keys if key[1] == key[2])
    components.extend({"kind": "power_cross_real", "key": list(key)} for key in power_keys if key[1] != key[2])
    components.extend({"kind": "power_cross_imag", "key": list(key)} for key in power_keys if key[1] != key[2])
    components.extend({"kind": "bispectrum_real", "key": list(key)} for key in bispectrum_keys)
    components.extend({"kind": "bispectrum_imag", "key": list(key)} for key in bispectrum_keys)
    return {
        "l_max": int(descriptor.l_max),
        "shell_radii": [float(value) for value in descriptor.shell_radii],
        "components": components,
    }


def _source_field_reference(cube_root: Path, molecule_id: str) -> Dict[str, Any]:
    cube_path = cube_root / f"{molecule_id}_density.cube"
    if not cube_path.exists():
        return {
            "cube_file": cube_path.name,
            "available": False,
            "coordinate_units": "cube_native_units",
        }
    stat = cube_path.stat()
    return {
        "cube_file": cube_path.name,
        "available": True,
        # Size + mtime are a cheap mutation sentinel.  The cube itself stays
        # the lossless source of every possible local field reconstruction.
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "coordinate_units": "cube_native_units",
    }


def write_letter_archive(
    *,
    candidates: Sequence[LetterCandidate],
    classes: Sequence[LetterClass],
    labels: Sequence[Optional[str]],
    metadata_registry: Mapping[str, Mapping[str, Any]],
    cube_dir: str | Path,
    output_dir: str | Path,
    class_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    analysis_context: Optional[Mapping[str, Any]] = None,
    descriptor_shard_rows: int = 250_000,
) -> Path:
    """Write a reusable archive without dropping any occurrence descriptor.

    ``descriptor_shard_rows`` bounds the largest dense output allocation.
    Descriptors are written through NumPy memmaps, so the writer needs only a
    single candidate vector beyond the candidate objects already owned by the
    caller.  No all-candidate descriptor matrix is constructed here.
    """
    if len(candidates) != len(labels):
        raise ValueError("candidates and labels must have identical lengths")
    if descriptor_shard_rows < 1:
        raise ValueError("descriptor_shard_rows must be positive")

    archive_dir = Path(output_dir)
    descriptor_dir = archive_dir / "descriptors"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    cube_root = Path(cube_dir)
    supplied_records = class_records or {}
    if candidates:
        descriptor_layout = _descriptor_layout(candidates[0])
        descriptor_dim = len(descriptor_layout["components"])
        first_vector = np.asarray(candidates[0].descriptor.to_vector(), dtype=np.float64)
        if first_vector.shape != (descriptor_dim,):
            raise ValueError("descriptor layout does not match vector length")
    else:
        descriptor_layout = {"l_max": None, "shell_radii": [], "components": []}
        descriptor_dim = 0

    # Class centroids are small but scientific data: preserve their float64
    # values rather than the slim-report norm/length summary.
    if classes:
        centroids = np.stack([
            np.asarray(letter_class.invariant_centroid, dtype=np.float64)
            for letter_class in classes
        ])
        if centroids.shape[1] != descriptor_dim:
            raise ValueError("class centroid dimension differs from occurrence descriptor dimension")
    else:
        centroids = np.empty((0, descriptor_dim), dtype=np.float64)
    np.save(archive_dir / "class_centroids.npy", centroids, allow_pickle=False)

    with (archive_dir / "classes.jsonl").open("w", encoding="utf-8") as handle:
        for centroid_row, letter_class in enumerate(classes):
            record = dict(supplied_records.get(letter_class.class_id, {}))
            record.update({
                "class_id": letter_class.class_id,
                "centroid_row": centroid_row,
                "critical_point_type": letter_class.discovery_provenance.critical_point_type,
                "discovery_provenance": _json_safe(letter_class.discovery_provenance),
                "cluster_stability": _json_safe(letter_class.cluster_stability),
                "semantic_status": _json_safe(letter_class.semantic_status),
            })
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    # The single occurrence stream lets another program select by type/class
    # without reading scene JSONs or loading descriptor values it does not use.
    descriptor_shards = []
    dynamic_frame_ids = set()
    n_observations_with_trajectory_context = 0
    with (archive_dir / "occurrences.jsonl").open("w", encoding="utf-8") as occurrence_handle:
        for shard_start in range(0, len(candidates), descriptor_shard_rows):
            shard_stop = min(len(candidates), shard_start + descriptor_shard_rows)
            shard_name = f"occurrence_descriptors_{shard_start:012d}_{shard_stop:012d}.npy"
            shard_relative = (Path("descriptors") / shard_name).as_posix()
            shard_path = archive_dir / shard_relative
            vectors = np.lib.format.open_memmap(
                shard_path,
                mode="w+",
                dtype=np.float64,
                shape=(shard_stop - shard_start, descriptor_dim),
            )
            for global_index in range(shard_start, shard_stop):
                candidate = candidates[global_index]
                vector = np.asarray(candidate.descriptor.to_vector(), dtype=np.float64)
                if vector.shape != (descriptor_dim,):
                    raise ValueError(
                        "variable descriptor layouts are not supported in one archive; "
                        "split the analysis by descriptor configuration"
                    )
                local_row = global_index - shard_start
                vectors[local_row] = vector
                class_id = labels[global_index]
                assigned_class_id = class_id or "unassigned_outlier"
                critical_point = candidate.critical_point
                molecule_metadata = metadata_registry.get(candidate.source_molecule_id, {})
                trajectory_id = molecule_metadata.get("trajectory_id")
                frame_index = molecule_metadata.get("frame_index")
                time_fs = molecule_metadata.get("time_fs")
                reaction_coordinate = molecule_metadata.get("reaction_coordinate")
                execution_stage = molecule_metadata.get("execution_stage")
                has_trajectory_context = trajectory_id is not None and (
                    frame_index is not None or time_fs is not None or reaction_coordinate is not None
                )
                if has_trajectory_context:
                    n_observations_with_trajectory_context += 1
                    dynamic_frame_ids.add((str(trajectory_id), str(frame_index), str(time_fs), str(reaction_coordinate)))
                occurrence = {
                    "occurrence_id": f"{candidate.source_molecule_id}:{global_index}",
                    "global_candidate_index": global_index,
                    "molecule_id": candidate.source_molecule_id,
                    "class_id": assigned_class_id,
                    "assigned_to_discovered_class": class_id is not None,
                    "critical_point_type": critical_point.field_type,
                    "environment_condition": candidate.environment_condition,
                    "dynamic_context": {
                        "trajectory_id": trajectory_id,
                        "frame_index": frame_index,
                        "time_fs": time_fs,
                        "reaction_coordinate": reaction_coordinate,
                        "execution_stage": execution_stage,
                        "has_ordered_frame_context": has_trajectory_context,
                        "cross_context_identity_status": "unresolved",
                    },
                    "critical_point": {
                        "position": np.asarray(critical_point.position, dtype=float).tolist(),
                        "field_value": float(critical_point.value),
                        "hessian_eigenvalues": np.asarray(critical_point.eigenvalues, dtype=float).tolist(),
                        "gradient_norm_at_field_root": getattr(critical_point, "gradient_norm", None),
                        "field_discovery_stage": getattr(critical_point, "discovery_stage", "unknown"),
                        "rank": int(critical_point.rank),
                        "signature": int(critical_point.signature),
                        "poincare_hopf_index": critical_point.poincare_hopf_index,
                        "basis_set_convergence": _json_safe(critical_point.basis_set_convergence),
                    },
                    "descriptor": {
                        "shard": shard_relative,
                        "row": local_row,
                        "dtype": "float64",
                        "layout": "descriptor_layout",
                        "support": {
                            "shell_radii": [float(radius) for radius in candidate.descriptor.shell_radii],
                            "outer_radius": float(max(candidate.descriptor.shell_radii)),
                            "coordinate_units": "cube_native_units",
                            "meaning": (
                                "Sampled local field support for the invariant descriptor; "
                                "not a basin boundary or inferred physical volume."
                            ),
                        },
                    },
                    "field_reference": {
                        "cube_file": f"{candidate.source_molecule_id}_density.cube",
                        "local_patch_template": local_patch_relative_path(
                            candidate.source_molecule_id, global_index
                        ).as_posix(),
                        "coordinate_units": "cube_native_units",
                    },
                }
                occurrence_handle.write(json.dumps(_json_safe(occurrence), ensure_ascii=False) + "\n")
            vectors.flush()
            del vectors
            descriptor_shards.append({
                "file": shard_relative,
                "global_row_start": shard_start,
                "global_row_stop_exclusive": shard_stop,
                "shape": [shard_stop - shard_start, descriptor_dim],
                "dtype": "float64",
            })

    with (archive_dir / "molecules.jsonl").open("w", encoding="utf-8") as handle:
        for molecule_id in sorted(metadata_registry):
            record = {
                "molecule_id": molecule_id,
                "metadata": _json_safe(dict(metadata_registry[molecule_id])),
                "field_reference": _source_field_reference(cube_root, molecule_id),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": LETTER_ARCHIVE_SCHEMA_VERSION,
        "coordinate_units": "cube_native_units",
        "cube_dir_hint": Path(os.path.relpath(cube_root, archive_dir)).as_posix(),
        "n_candidates": len(candidates),
        "n_classes": len(classes),
        "descriptor_layout": descriptor_layout,
        "descriptor_shards": descriptor_shards,
        "class_centroids": {"file": "class_centroids.npy", "dtype": "float64", "shape": list(centroids.shape)},
        "class_records": "classes.jsonl",
        "occurrences": "occurrences.jsonl",
        "molecules": "molecules.jsonl",
        "analysis_context": _json_safe(dict(analysis_context or {})),
        "dynamics_and_context": {
            "n_occurrences_with_ordered_frame_context": n_observations_with_trajectory_context,
            "n_distinct_declared_frames": len(dynamic_frame_ids),
            "transition_model_inferred": False,
            "cross_context_class_merge_performed": False,
            "identity_scope": (
                "A class ID is a descriptor equivalence under the configured observation regime. "
                "It is not proof of identity across time, environment, charge, or execution stage."
            ),
            "required_for_dynamic_identity": [
                "ordered frames with a physical time or path coordinate",
                "critical-point or basin correspondence tracking across adjacent frames",
                "a held-out transition/context test before any class merge",
            ],
        },
        "notes": [
            "Rows preserve the candidate order and labels from the completed analysis; no re-clustering occurs while writing this archive.",
            "Candidate discovery is field-first. Atomic coordinates are absent from the candidate-existence, topology, descriptor and class-assignment path; a source cube atom block is retained only as an independent reference annotation.",
            "Original density cubes are the complete field record. Per-letter local cubes are derived lazily to avoid duplicating the same molecular field thousands of times.",
            "Ring and cage critical points are retained as ordinary occurrences even when their positions lie away from atoms.",
            "Dynamic/context fields are stored when supplied by molecule metadata, but this archive never turns them into grammar or cross-context identity without a dedicated trajectory analysis.",
        ],
    }
    manifest_path = archive_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def iter_occurrences(archive_dir: str | Path) -> Iterator[Dict[str, Any]]:
    """Stream occurrence records without loading a large corpus into RAM."""
    root = Path(archive_dir)
    with (root / "occurrences.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def open_descriptor_shard(archive_dir: str | Path, shard_file: str) -> np.ndarray:
    """Memory-map one occurrence-vector shard for a downstream analyser."""
    return np.load(Path(archive_dir) / shard_file, mmap_mode="r", allow_pickle=False)
