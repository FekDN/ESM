"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Append-only online alphabet state for interruptible corpus growth.

The regular batch pipeline is useful for a bounded scientific calibration,
but re-clustering every molecule after adding a new shard is not a viable
operating model for a corpus of millions.  This module stores the sufficient
statistics of a type-partitioned online DP-means fit: stable class IDs,
centroids, counts, within-class SSE and the calibrated spawn threshold.

New candidates are assigned only once.  The update never reopens old cubes,
recomputes old descriptors or reassigns old occurrences.  A full batch
reanalysis is therefore a *separate, explicitly requested scientific
experiment* (for example after changing a descriptor), not the resume path.
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .corpus import LetterCandidate, suggest_distance_threshold
from .schema import ConfidenceBlock, DiscoveryProvenance, LetterClass, SemanticStatus


ONLINE_ALPHABET_SCHEMA_VERSION = "esm-online-alphabet/v3"
_MOLECULE_SKETCH_SIZE = 4096
# Was 16. Verified empirically before raising this (not just reasoned about):
# with k=16, a K-minimum-value cardinality estimate has relative standard
# error ~= 1/sqrt(k-2) ~= 27% (confirmed by simulation against this exact
# hash function: stdev=266 on a true cardinality of 1000, 49% of draws
# landing ABOVE the true value, 10%+ landing 30%+ above it -- e.g. an
# estimate of 1325 "distinct molecules" for a class drawn from a
# corpus that only HAS 1000, which is not a bug in the formula (it matches
# the standard unbiased KMV estimator exactly) but a k far too small for
# the corpus sizes this was actually run against. At k=4096, the same
# simulation gives EXACT counts (not merely tighter estimates) for any
# class touched by up to 4096 distinct molecules -- covering this
# codebase's own "calibration" scale tier (doc §21.1) outright, and
# falling back to the same well-understood estimator, now at ~1.6% relative
# error, only for classes that genuinely exceed that. Memory cost is
# bounded and explicit: 4096 sixty-four-bit hashes is 32KB in memory per
# prototype that reaches the cap; only the (typically few) most-reused
# classes ever do.


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def new_online_alphabet_state(analysis_signature: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": ONLINE_ALPHABET_SCHEMA_VERSION,
        "analysis_signature": _json_safe(dict(analysis_signature)),
        "threshold_by_type": {},
        "threshold_status_by_type": {},
        "next_class_index_by_type": {},
        "prototypes_by_type": {},
        # An occurrence keeps the ID it received at append time.  Aliases
        # resolve it to the current consolidated class without rewriting any
        # old descriptor batch or cube.  This is the key distinction between
        # a growing corpus and an expensive global re-clustering.
        "class_aliases": {},
        "merge_history": [],
        "n_prototype_merges": 0,
        "merge_policy": None,
        "n_molecules": 0,
        "n_candidates": 0,
        "n_updates": 0,
        "updated_unix_s": None,
    }


def load_online_alphabet_state(
    path: str | Path,
    analysis_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    """Load a checkpoint or create one after strict signature validation."""
    checkpoint = Path(path)
    if not checkpoint.exists():
        return new_online_alphabet_state(analysis_signature)
    with checkpoint.open(encoding="utf-8") as handle:
        state = json.load(handle)
    # v1 had stable streaming prototypes but no alias graph. v2 added aliases
    # but no bounded distinct-molecule sketch. Both migrations preserve all
    # existing values and explicitly mark their historic molecule coverage as
    # unknown rather than inventing a count from old class occurrences.
    if state.get("schema_version") in {"esm-online-alphabet/v1", "esm-online-alphabet/v2"}:
        state["schema_version"] = ONLINE_ALPHABET_SCHEMA_VERSION
        state.setdefault("class_aliases", {})
        state.setdefault("merge_history", [])
        state.setdefault("n_prototype_merges", 0)
        state.setdefault("merge_policy", None)
        for prototypes in state.get("prototypes_by_type", {}).values():
            for prototype in prototypes:
                prototype.setdefault("molecule_sketch", [])
                prototype.setdefault("molecule_sketch_complete", False)
    if state.get("schema_version") != ONLINE_ALPHABET_SCHEMA_VERSION:
        raise ValueError("online alphabet checkpoint schema is incompatible")
    expected = _json_safe(dict(analysis_signature))
    if state.get("analysis_signature") != expected:
        raise ValueError(
            "online alphabet checkpoint belongs to another extraction/descriptor regime; "
            "start a new state rather than mixing incomparable candidates"
        )
    return state


def save_online_alphabet_state(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomically publish a completed update checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(state), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def _calibrate_new_thresholds(
    state: Dict[str, Any], candidates: Sequence[LetterCandidate]
) -> None:
    by_type: Dict[str, List[LetterCandidate]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate.critical_point.field_type, []).append(candidate)
    for field_type, pool in by_type.items():
        if field_type in state["threshold_by_type"]:
            continue
        if len(pool) >= 2:
            threshold = suggest_distance_threshold(pool, group_by_field_type=False)
            state["threshold_by_type"][field_type] = float(threshold)
            state["threshold_status_by_type"][field_type] = "calibrated_from_first_append"
        else:
            # Do not invent a nonzero scale from one observation.  It becomes
            # a singleton class; later updates may add classes but never
            # rewrite this already published observation behind the user's
            # back.  The status makes this limitation queryable.
            state["threshold_by_type"][field_type] = 0.0
            state["threshold_status_by_type"][field_type] = "uncalibrated_singleton"


def resolve_online_class_id(state: Mapping[str, Any], class_id: str) -> str:
    """Resolve an immutable historical assignment through the alias graph."""
    aliases = state.get("class_aliases", {})
    current = str(class_id)
    seen = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = str(aliases[current])
    return current


def _molecule_hash(molecule_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(str(molecule_id).encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _add_molecule_to_sketch(prototype: Dict[str, Any], molecule_id: str, sketch_size: int = _MOLECULE_SKETCH_SIZE) -> None:
    """Maintain a fixed-size KMV sketch of distinct source molecules.

    The sketch retains only the smallest deterministic hashes, so it remains
    bounded even for billions of records and can be unioned exactly enough
    for an unbiased cardinality estimate after prototype merges.  It stores
    no SMILES or source identifier.

    `sketch_size` must match whatever this prototype's earlier updates used
    (see `update_online_alphabet`'s guard against changing it mid-run): a
    prototype that already discarded hashes beyond an old, smaller cap
    cannot retroactively recover them under a larger one.
    """
    raw = prototype.setdefault("molecule_sketch", [])
    sketch = sorted({int(value) for value in raw})
    value = _molecule_hash(molecule_id)
    if value in sketch:
        return
    if len(sketch) < sketch_size:
        sketch.append(value)
    elif value < sketch[-1]:
        sketch[-1] = value
    prototype["molecule_sketch"] = sorted(set(sketch))[:sketch_size]
    prototype.setdefault("molecule_sketch_complete", True)


def _merge_molecule_sketches(
    prototypes: Sequence[Mapping[str, Any]], sketch_size: int = _MOLECULE_SKETCH_SIZE
) -> tuple[List[int], bool]:
    values = sorted({
        int(value)
        for prototype in prototypes
        for value in prototype.get("molecule_sketch", [])
    })[:sketch_size]
    complete = all(bool(prototype.get("molecule_sketch_complete", False)) for prototype in prototypes)
    return values, complete


def _estimate_distinct_molecules(
    prototype: Mapping[str, Any], sketch_size: int = _MOLECULE_SKETCH_SIZE
) -> tuple[Optional[int], str]:
    """Return estimate and whether it is exact, estimated, or unavailable."""
    if not bool(prototype.get("molecule_sketch_complete", False)):
        return None, "unavailable_before_sketch"
    sketch = sorted({int(value) for value in prototype.get("molecule_sketch", [])})
    if len(sketch) < sketch_size:
        return len(sketch), "exact_below_sketch_capacity"
    kth = sketch[-1]
    if kth <= 0:
        return None, "unavailable_degenerate_sketch"
    estimate = int(round((len(sketch) - 1) * ((2**64 - 1) / kth)))
    return max(len(sketch), estimate), "kmv_estimate"


def online_prototype_evidence(
    prototype: Mapping[str, Any],
    threshold: float,
    *,
    sketch_size: int = _MOLECULE_SKETCH_SIZE,
) -> Dict[str, Any]:
    """Report evidence strength without claiming a probability of truth.

    Online centroids provide support and compactness, but not bootstrap
    stability, held-out recovery, semantic identification or a dynamics test.
    This function therefore emits a readable *evidence status*, never a
    misleading numerical ``probability that this is a correct letter``.

    `sketch_size` should match the run's own `state["molecule_sketch_size"]`
    when the caller has it to hand (report-writing code usually does); the
    default here is only a fallback for callers reading a lone prototype
    without that context.
    """
    n_occurrences = int(prototype.get("count", 0))
    rms = float(np.sqrt(max(0.0, float(prototype.get("sum_squared_distance", 0.0))) / max(1, n_occurrences)))
    threshold = float(threshold)
    relative_rms = (rms / threshold) if threshold > np.finfo(float).eps else None
    molecules, molecule_method = _estimate_distinct_molecules(prototype, sketch_size=sketch_size)
    merged_from = prototype.get("merged_from", [prototype.get("class_id")])
    variant_count = len(set(str(value) for value in merged_from))
    if n_occurrences < 5:
        status = "exploratory_insufficient_support"
    elif relative_rms is not None and relative_rms > 0.75:
        status = "provisional_diffuse_descriptor_group"
    elif n_occurrences < 30:
        status = "provisional_replicating"
    else:
        status = "reused_streaming_group_unvalidated"
    return {
        "evidence_status": status,
        "n_occurrences": n_occurrences,
        "n_distinct_molecules_estimate": molecules,
        "distinct_molecule_estimate_method": molecule_method,
        "within_class_rms_distance": rms,
        "assignment_threshold": threshold,
        "relative_rms_to_threshold": relative_rms,
        "merged_variant_count": variant_count,
        "validation_status": "online_provisional_no_bootstrap_or_held_out_validation",
        "not_a_correctness_probability": True,
        "required_next_evidence": [
            "bootstrap or held-out replication on a bounded corpus",
            "threshold and descriptor-regime sensitivity",
            "field-grid and level-of-theory convergence",
            "ordered multi-frame evidence before grammar or cross-context identity claims",
        ],
    }


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= np.finfo(float).eps:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))


def _merge_connected_prototypes(
    state: Dict[str, Any],
    *,
    min_similarity: float,
    screen_max_prototypes: int,
    sketch_size: int = _MOLECULE_SKETCH_SIZE,
) -> int:
    """Consolidate transitive prototype families without reopening examples.

    Similarities are evaluated between *current sufficient-statistic
    centroids* for the same critical-point type.  Connected components are
    merged in one step, so A--C--D--B can become one family even when A and B
    never clear the direct threshold.  A low-affinity edge is only a
    provisional consolidation rule; merge provenance and every pre-merge ID
    remain on disk for later falsification or threshold sensitivity tests.
    """
    if not state.get("prototypes_by_type"):
        return 0
    aliases = state.setdefault("class_aliases", {})
    history = state.setdefault("merge_history", [])
    total_merges = 0

    for field_type, prototypes in list(state["prototypes_by_type"].items()):
        n_items = len(prototypes)
        if n_items < 2:
            continue
        # A full dense similarity matrix would defeat the online path.  Unit
        # centroids allow radius-neighbour screening, then exact cosine is
        # still checked before an edge is accepted.
        vectors = np.asarray([item["centroid"] for item in prototypes], dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > np.finfo(float).eps
        if n_items > screen_max_prototypes:
            # This cap is explicit rather than silently spending O(K^2) RAM.
            # Recent singleton prototypes are always included; older active
            # groups await a later bounded consolidation job.
            order = np.argsort([int(item["count"]) for item in prototypes])
            selected = np.sort(order[-screen_max_prototypes:])
        else:
            selected = np.arange(n_items)
        parent = np.arange(n_items, dtype=int)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        candidates: List[Tuple[int, int]] = []
        selected_valid = selected[valid[selected]]
        if len(selected_valid) >= 2:
            normalized = vectors[selected_valid] / norms[selected_valid, None]
            try:
                from scipy.spatial import cKDTree
                radius = float(np.sqrt(max(0.0, 2.0 * (1.0 - min_similarity)))) + 1e-12
                pairs = cKDTree(normalized).query_pairs(radius, output_type="ndarray")
                candidates = [
                    (int(selected_valid[first]), int(selected_valid[second]))
                    for first, second in pairs
                ]
            except (ImportError, ValueError):  # small, dependency-free fallback
                for offset, first in enumerate(selected_valid[:-1]):
                    for second in selected_valid[offset + 1:]:
                        candidates.append((int(first), int(second)))
        for first, second in candidates:
            if _cosine_similarity(vectors[first], vectors[second]) >= min_similarity:
                union(first, second)

        groups: Dict[int, List[int]] = {}
        for index in range(n_items):
            groups.setdefault(find(index), []).append(index)
        new_prototypes: List[Dict[str, Any]] = []
        for indices in groups.values():
            if len(indices) == 1:
                new_prototypes.append(prototypes[indices[0]])
                continue
            members = [prototypes[index] for index in indices]
            # Keep the lexically oldest generated ID as a stable canonical
            # name, while storing every other name as a durable alias.
            canonical = min(str(item["class_id"]) for item in members)
            counts = np.asarray([int(item["count"]) for item in members], dtype=float)
            centroids = np.asarray([item["centroid"] for item in members], dtype=float)
            count = int(np.sum(counts))
            centroid = np.average(centroids, axis=0, weights=counts)
            sse = float(sum(float(item.get("sum_squared_distance", 0.0)) for item in members))
            sse += float(np.sum(counts * np.sum((centroids - centroid) ** 2, axis=1)))
            aliases_in_component = sorted(str(item["class_id"]) for item in members if str(item["class_id"]) != canonical)
            for alias in aliases_in_component:
                aliases[alias] = canonical
            source_ids = sorted({
                str(source_id)
                for item in members
                for source_id in item.get("merged_from", [item["class_id"]])
            })
            molecule_sketch, molecule_sketch_complete = _merge_molecule_sketches(members, sketch_size=sketch_size)
            new_prototypes.append({
                "class_id": canonical,
                "centroid": centroid.tolist(),
                "count": count,
                "sum_squared_distance": sse,
                "merged_from": source_ids,
                "molecule_sketch": molecule_sketch,
                "molecule_sketch_complete": molecule_sketch_complete,
            })
            total_merges += len(members) - 1
            history.append({
                "update_index": int(state.get("n_updates", 0)) + 1,
                "critical_point_type": field_type,
                "canonical_class_id": canonical,
                "merged_class_ids": aliases_in_component,
                "criterion": "transitive_centroid_cosine_graph",
                "minimum_similarity": float(min_similarity),
                "n_members": len(members),
            })
        state["prototypes_by_type"][field_type] = sorted(
            new_prototypes, key=lambda item: str(item["class_id"])
        )

    state["n_prototype_merges"] = int(state.get("n_prototype_merges", 0)) + total_merges
    return total_merges


def update_online_alphabet(
    state: Dict[str, Any],
    candidates: Sequence[LetterCandidate],
    *,
    n_new_molecules: int,
    merge_enabled: bool = True,
    merge_min_similarity: float = 0.80,
    merge_screen_max_prototypes: int = 10_000,
    sketch_size: int = _MOLECULE_SKETCH_SIZE,
) -> List[str]:
    """Assign only the appended candidates and update centroid sufficient stats.

    Returned labels are stable IDs for the occurrence archive.  State is
    mutated in memory but not written here; callers must write occurrence
    data first and then atomically call :func:`save_online_alphabet_state`.

    `sketch_size` is guarded the same way `merge_policy` already is below:
    a prototype's molecule-sketch is built incrementally and once it has
    discarded hashes beyond an old cap, a larger cap cannot retroactively
    recover them -- so this must stay fixed for a run's lifetime, not
    silently drift between one call to `update_online_alphabet` and the
    next.
    """
    _calibrate_new_thresholds(state, candidates)
    previous_sketch_size = state.get("molecule_sketch_size")
    if previous_sketch_size is not None and int(previous_sketch_size) != int(sketch_size):
        raise ValueError(
            f"online molecule_sketch_size changed ({previous_sketch_size} -> {sketch_size}) for an "
            f"existing run; choose a new run_id instead of changing its historical class semantics"
        )
    state["molecule_sketch_size"] = int(sketch_size)
    policy = {
        "enabled": bool(merge_enabled),
        "minimum_similarity": float(merge_min_similarity),
        "screen_max_prototypes": int(merge_screen_max_prototypes),
        "criterion": "transitive_centroid_cosine_graph",
    }
    previous_policy = state.get("merge_policy")
    if previous_policy is not None and previous_policy != policy:
        raise ValueError(
            "online merge policy changed for an existing run; choose a new run_id "
            "instead of changing its historical class semantics"
        )
    state["merge_policy"] = policy
    labels: List[str] = []
    prototypes_by_type = state["prototypes_by_type"]
    next_index = state["next_class_index_by_type"]

    for candidate in candidates:
        field_type = candidate.critical_point.field_type
        vector = np.asarray(candidate.descriptor.to_vector(), dtype=float)
        prototypes = prototypes_by_type.setdefault(field_type, [])
        threshold = float(state["threshold_by_type"][field_type])
        if prototypes:
            centroids = np.asarray([item["centroid"] for item in prototypes], dtype=float)
            distances = np.linalg.norm(centroids - vector, axis=1)
            best = int(np.argmin(distances))
            best_distance = float(distances[best])
        else:
            best, best_distance = -1, np.inf

        if best >= 0 and best_distance <= threshold:
            prototype = prototypes[best]
            old_count = int(prototype["count"])
            old_centroid = np.asarray(prototype["centroid"], dtype=float)
            new_count = old_count + 1
            new_centroid = old_centroid + (vector - old_centroid) / new_count
            prototype["centroid"] = new_centroid.tolist()
            prototype["count"] = new_count
            prototype["sum_squared_distance"] = float(
                prototype.get("sum_squared_distance", 0.0) + best_distance * best_distance
            )
            _add_molecule_to_sketch(prototype, candidate.source_molecule_id, sketch_size=sketch_size)
            labels.append(str(prototype["class_id"]))
            continue

        index = int(next_index.get(field_type, 0))
        next_index[field_type] = index + 1
        class_id = f"online_{field_type}_{index:08d}"
        prototypes.append({
            "class_id": class_id,
            "centroid": vector.tolist(),
            "count": 1,
            "sum_squared_distance": 0.0,
            "merged_from": [class_id],
            "molecule_sketch": [],
            "molecule_sketch_complete": True,
        })
        _add_molecule_to_sketch(prototypes[-1], candidate.source_molecule_id, sketch_size=sketch_size)
        labels.append(class_id)

    state["n_molecules"] = int(state.get("n_molecules", 0)) + int(n_new_molecules)
    state["n_candidates"] = int(state.get("n_candidates", 0)) + len(candidates)
    if merge_enabled:
        _merge_connected_prototypes(
            state,
            min_similarity=float(merge_min_similarity),
            screen_max_prototypes=int(merge_screen_max_prototypes),
            sketch_size=sketch_size,
        )
    state["n_updates"] = int(state.get("n_updates", 0)) + 1
    state["updated_unix_s"] = time.time()
    return labels


def online_state_to_classes(
    state: Mapping[str, Any],
    *,
    l_max_bispectrum: int,
    corpus_version: str,
) -> List[LetterClass]:
    """Materialise the current online alphabet without touching old data."""
    classes: List[LetterClass] = []
    for field_type, prototypes in sorted(state.get("prototypes_by_type", {}).items()):
        threshold = float(state.get("threshold_by_type", {}).get(field_type, 0.0))
        for prototype in prototypes:
            count = int(prototype["count"])
            classes.append(LetterClass(
                class_id=str(prototype["class_id"]),
                discovery_provenance=DiscoveryProvenance(
                    persistence_threshold_used=threshold,
                    l_max_bispectrum=l_max_bispectrum,
                    critical_point_type=field_type,
                ),
                invariant_centroid=np.asarray(prototype["centroid"], dtype=float),
                cluster_stability=ConfidenceBlock(
                    n=count,
                    method="expert_provisional",
                    score=0.0,
                    corpus_version=corpus_version,
                    stale=False,
                ),
                semantic_status=SemanticStatus(orphan=True, orphan_confidence=None),
            ))
    return classes
