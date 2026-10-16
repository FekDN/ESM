"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
boundary_audit.py
==================
The streaming assignment in `online_alphabet.py` compares a new occurrence
to each class's CURRENT centroid, but a class's centroid is a running mean
that keeps moving as members are added (`update_online_alphabet`'s
`new_centroid = old_centroid + (vector - old_centroid) / new_count`). An
occurrence that was correctly within its type's threshold at the moment it
was appended can therefore drift outside it later, purely because its own
class's centroid moved -- not because anything about the occurrence itself
changed. This module is the audit for that: it is not a re-run of the
original clustering rule with a stricter setting, it is a check for members
stranded by a mechanism the streaming assignment itself has.

Deliberately NOT part of this check: `environment_condition`. Classification
here is by descriptor similarity alone, exactly the criterion already in
force for the streaming assignment -- gating a reassignment on environment
would mean assuming, at the letter-cataloguing stage, that environment
governs realisation by a rule, which is a question for a later, explicit
distributional analysis over an already-built alphabet (same axis as
`positional_sensitivity.py`, with "environment" standing in for "position"),
not a filter baked into how the alphabet itself gets built.

Two passes, and the split matters for scale:
  1. `find_boundary_candidates` -- O(n_occurrences), one centroid lookup per
     occurrence: is this occurrence still within its OWN (possibly moved)
     class's threshold? Flags only the occurrences that need a second look.
  2. For each flagged occurrence, an O(k_classes) nearest-other-centroid
     scan -- paid only for the (expected to be small) flagged set, not for
     every occurrence, which is what keeps this tractable well past the
     scale where re-clustering everything is not.

This module only detects and records candidates (`find_boundary_candidates`)
and, separately, commits them to a ledger (`write_boundary_reassignments`,
`resolve_class_id_with_reassignment`). It never mutates `online_alphabet_state`
in place, for the same reason `recluster_from_archive` writes to its own
`output_state_path` rather than the live checkpoint: an audit's candidates
are a proposal, not an automatic edit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from .online_alphabet import resolve_online_class_id
from .online_archive import iter_archived_occurrences

BOUNDARY_AUDIT_SCHEMA_VERSION = "esm-boundary-audit/v1"


@dataclass
class BoundaryCandidate:
    update_index: int
    row: int
    molecule_id: str
    critical_point_type: str
    current_class_id: str
    current_distance: float
    current_threshold: float
    proposed_class_id: str
    proposed_distance: float
    proposed_threshold: float

    def occurrence_key(self) -> str:
        # (update_index, row) is unique within one online_dir: one batch
        # file per update_index, `row` the position within its .npy array
        # (see `append_to_online_archive` / `iter_archived_occurrences`).
        return f"{self.update_index}:{self.row}"

    def to_json_safe(self) -> Dict[str, Any]:
        return {
            "update_index": self.update_index,
            "row": self.row,
            "molecule_id": self.molecule_id,
            "critical_point_type": self.critical_point_type,
            "from_class": self.current_class_id,
            "from_distance": self.current_distance,
            "from_threshold": self.current_threshold,
            "to_class": self.proposed_class_id,
            "to_distance": self.proposed_distance,
            "to_threshold": self.proposed_threshold,
        }


def find_boundary_candidates(
    state: Mapping[str, Any],
    online_dir: str | Path,
    *,
    drift_margin: float = 0.0,
    min_improvement: float = 0.0,
) -> List[BoundaryCandidate]:
    """See module docstring. `drift_margin` widens Pass 1's net: an
    occurrence is flagged if its distance to its own class exceeds
    `threshold - drift_margin` (0.0 flags only occurrences that have
    actually drifted past their threshold; a small positive value also
    catches ones close enough to be worth a second look this audit, at the
    cost of more Pass-2 work). `min_improvement` is the hysteresis dead-band
    from doc-discussion §20-style damping: a proposed class must be closer
    by at least this much, not merely closer, so a point sitting almost
    exactly on a boundary does not flip on every audit.
    """
    prototypes_by_type = state.get("prototypes_by_type", {})
    centroid_index: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = {}
    for field_type, prototypes in prototypes_by_type.items():
        by_class: Dict[str, Tuple[np.ndarray, float]] = {}
        threshold = float(state.get("threshold_by_type", {}).get(field_type, 0.0))
        for prototype in prototypes:
            by_class[str(prototype["class_id"])] = (
                np.asarray(prototype["centroid"], dtype=float), threshold,
            )
        centroid_index[field_type] = by_class

    candidates: List[BoundaryCandidate] = []
    for record, vector in iter_archived_occurrences(online_dir):
        field_type = record["critical_point_type"]
        by_class = centroid_index.get(field_type)
        if not by_class:
            continue
        current_class_id = resolve_online_class_id(state, record["class_id_at_append"])
        current = by_class.get(current_class_id)
        if current is None:
            continue  # class vanished into a merge this state hasn't recorded an alias for yet
        current_centroid, current_threshold = current
        current_distance = float(np.linalg.norm(current_centroid - vector))
        if current_distance <= current_threshold - drift_margin:
            continue  # still comfortably inside its own class; nothing to audit

        best_class_id: Optional[str] = None
        best_distance = np.inf
        best_threshold = 0.0
        for other_class_id, (other_centroid, other_threshold) in by_class.items():
            if other_class_id == current_class_id:
                continue
            distance = float(np.linalg.norm(other_centroid - vector))
            if distance < best_distance:
                best_class_id, best_distance, best_threshold = other_class_id, distance, other_threshold

        if (
            best_class_id is not None
            and best_distance <= best_threshold
            and best_distance <= current_distance - min_improvement
        ):
            candidates.append(BoundaryCandidate(
                update_index=int(record["update_index"]),
                row=int(record["descriptor"]["row"]),
                molecule_id=record["molecule_id"],
                critical_point_type=field_type,
                current_class_id=current_class_id,
                current_distance=current_distance,
                current_threshold=current_threshold,
                proposed_class_id=best_class_id,
                proposed_distance=best_distance,
                proposed_threshold=best_threshold,
            ))
    return candidates


def write_boundary_reassignments(
    online_dir: str | Path,
    candidates: List[BoundaryCandidate],
    *,
    audit_label: str,
) -> Path:
    """Append accepted candidates to this archive's reassignment ledger.
    Additive and append-only, exactly like `occurrences.jsonl` itself --
    never rewrites a prior audit's entries, so which audit proposed which
    reassignment, and when, stays reconstructable (doc §21.2's discipline:
    history of assignment is kept, not overwritten)."""
    path = Path(online_dir) / "boundary_reassignments.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for candidate in candidates:
            row = candidate.to_json_safe()
            row["audit_label"] = audit_label
            row["schema_version"] = BOUNDARY_AUDIT_SCHEMA_VERSION
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def load_reassignment_index(online_dir: str | Path) -> Dict[str, str]:
    """occurrence_key -> most recently proposed `to_class` (unresolved --
    pass through `resolve_online_class_id` too, since the target class
    itself may have been merged into something else since the audit ran)."""
    path = Path(online_dir) / "boundary_reassignments.jsonl"
    index: Dict[str, str] = {}
    if not path.exists():
        return index
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[f"{row['update_index']}:{row['row']}"] = row["to_class"]
    return index


def resolve_class_id_with_reassignment(
    state: Mapping[str, Any],
    reassignment_index: Mapping[str, str],
    *,
    update_index: int,
    row: int,
    class_id_at_append: str,
) -> str:
    """The read-time counterpart: alias-resolve as `resolve_online_class_id`
    always has, then apply this occurrence's most recent audited
    reassignment if one exists, then alias-resolve THAT (its target may
    itself have since been merged elsewhere)."""
    resolved = resolve_online_class_id(state, class_id_at_append)
    override = reassignment_index.get(f"{update_index}:{row}")
    if override is None:
        return resolved
    return resolve_online_class_id(state, override)
