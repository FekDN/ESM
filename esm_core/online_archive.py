"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Append-only descriptor batches for the online alphabet checkpoint path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .corpus import LetterCandidate
from .online_alphabet import (
    new_online_alphabet_state,
    resolve_online_class_id,
    save_online_alphabet_state,
    update_online_alphabet,
)



ONLINE_ARCHIVE_SCHEMA_VERSION = "esm-online-archive/v1"


def append_online_descriptor_batch(
    output_dir: str | Path,
    *,
    update_index: int,
    candidates: Sequence[LetterCandidate],
    labels: Sequence[str],
    analysis_signature: Mapping[str, Any],
    alphabet_state: Mapping[str, Any],
) -> Path:
    """Persist new vectors before their state checkpoint is published.

    Each update has its own immutable ``.npy`` file, so an interrupted run
    cannot corrupt old descriptors.  The class state contains enough
    statistics to resume immediately; these batches retain the raw evidence
    for a later, explicitly requested retrospective validation without DFT.
    """
    if len(candidates) != len(labels):
        raise ValueError("candidates and labels must have equal length")
    root = Path(output_dir)
    batches = root / "descriptor_batches"
    batches.mkdir(parents=True, exist_ok=True)
    stem = f"update_{int(update_index):012d}"
    data_path = batches / f"{stem}.npy"
    temporary_data = data_path.with_suffix(data_path.suffix + f".{os.getpid()}.tmp")
    vectors = (
        np.stack([candidate.descriptor.to_vector() for candidate in candidates]).astype(np.float64)
        if candidates else np.empty((0, 0), dtype=np.float64)
    )
    with temporary_data.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_data, data_path)

    occurrence_path = root / "occurrences.jsonl"
    with occurrence_path.open("a", encoding="utf-8") as handle:
        for row, (candidate, label) in enumerate(zip(candidates, labels)):
            critical_point = candidate.critical_point
            current_class_id = resolve_online_class_id(alphabet_state, str(label))
            handle.write(json.dumps({
                "update_index": int(update_index),
                "molecule_id": candidate.source_molecule_id,
                "class_id": current_class_id,
                "class_id_at_append": str(label),
                "critical_point_type": critical_point.field_type,
                "critical_point": {
                    "position": np.asarray(critical_point.position, dtype=float).tolist(),
                    "field_value": float(critical_point.value),
                    "hessian_eigenvalues": np.asarray(critical_point.eigenvalues, dtype=float).tolist(),
                    "gradient_norm_at_field_root": getattr(critical_point, "gradient_norm", None),
                    "field_discovery_stage": getattr(critical_point, "discovery_stage", "unknown"),
                    "rank": int(critical_point.rank),
                    "signature": int(critical_point.signature),
                },
                "descriptor": {
                    "file": data_path.relative_to(root).as_posix(),
                    "row": row,
                    "dtype": "float64",
                    "feature_regime": getattr(candidate.descriptor, "regime", "local_field"),
                },
            }, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": ONLINE_ARCHIVE_SCHEMA_VERSION,
        "analysis_signature": dict(analysis_signature),
        "state_checkpoint": "online_alphabet_state.json",
        "occurrences": "occurrences.jsonl",
        "descriptor_batches": "descriptor_batches/",
        "notes": [
            "Append-only batches preserve every completed descriptor without repeating field extraction.",
            "class_id_at_append is immutable evidence; class_id is resolved through the current alias graph after an online transitive-prototype consolidation.",
            "Online class IDs are current streaming assignments. They are not a substitute for an explicitly requested retrospective batch-validation experiment.",
        ],
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + f".{os.getpid()}.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_manifest, manifest_path)
    return data_path

class _ArchivedCandidate:
    """Minimal stand-in for a ``LetterCandidate``, built from archived evidence.

    ``update_online_alphabet`` only reads ``.source_molecule_id``,
    ``.critical_point.field_type`` and ``.descriptor.to_vector()``; everything
    else a real ``LetterCandidate`` carries (geometry, cube path, PySCF
    handles, ...) is not needed to replay the streaming assignment and merge
    step, and is exactly what a from-cache-or-DFT rebuild would otherwise
    have to pay for.
    """

    __slots__ = ("source_molecule_id", "critical_point", "descriptor")

    def __init__(self, molecule_id: str, field_type: str, vector: np.ndarray) -> None:
        self.source_molecule_id = molecule_id
        self.critical_point = SimpleNamespace(field_type=field_type)
        self.descriptor = SimpleNamespace(to_vector=lambda vector=vector: vector)


def iter_archived_occurrences(
    online_dir: str | Path,
) -> Iterator[Tuple[Dict[str, Any], np.ndarray]]:
    """Yield ``(occurrence_record, descriptor_vector)`` for every archived
    candidate, in original append order.

    Reads only ``occurrences.jsonl`` and the ``descriptor_batches/*.npy``
    files ``append_online_descriptor_batch`` already writes on every
    online-append checkpoint. No DFT, no cube access, no CorpusLedger touch,
    and no per-molecule pickle cache reads -- at a few hundred million
    molecules, one ``.npy`` per checkpointed batch is a very different I/O
    shape than one file per molecule.

    At most one batch file is held in memory at a time: ``occurrences.jsonl``
    is written in the same append order as the batch files, so once a row
    references a new batch file the previous one is never needed again.
    """
    online_dir = Path(online_dir)
    occurrence_path = online_dir / "occurrences.jsonl"
    if not occurrence_path.exists():
        return
    batch_cache: "Dict[str, np.ndarray]" = {}
    with occurrence_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            batch_file = record["descriptor"]["file"]
            if batch_file not in batch_cache:
                batch_cache = {
                    batch_file: np.load(online_dir / batch_file, allow_pickle=False)
                }
            yield record, batch_cache[batch_file][record["descriptor"]["row"]]


def _read_manifest(online_dir: Path) -> Dict[str, Any]:
    manifest_path = online_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no descriptor archive at {online_dir}; nothing to recluster from")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _matching_extraction_signature(
    online_dirs: Sequence[Path], manifests: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Every shard combined into one alphabet must share one extraction
    signature: a descriptor is only comparable to another if both came from
    the same field/candidate-extraction settings (§15's practical
    precautions) -- merging shards extracted under different settings would
    silently pool incomparable vectors into one class. Checked up front, all
    at once, so a mismatch is reported before any replay work happens rather
    than surfacing as a confusing mid-run assignment/merge artifact.
    """
    first_dir, first_signature = online_dirs[0], manifests[0]["analysis_signature"]
    for online_dir, manifest in zip(online_dirs[1:], manifests[1:]):
        if manifest["analysis_signature"] != first_signature:
            raise ValueError(
                f"archive at {online_dir} was extracted under a different analysis_signature "
                f"than {first_dir}; combining them would silently mix incomparable descriptor "
                "vectors. Reconcile the extraction settings (or drop the mismatched shard) "
                "before reclustering them together."
            )
    return first_signature


def recluster_from_archives(
    online_dirs: str | Path | Sequence[str | Path],
    *,
    output_state_path: str | Path,
    merge_enabled: bool,
    merge_min_similarity: float,
    merge_screen_max_prototypes: int,
    molecules_per_batch: Optional[int] = None,
    checkpoint_every_batches: int = 1,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Replay one or more archives of already-extracted candidates under a
    (possibly new) shared merge policy.

    This is the cheap answer to "what would this merge/consolidation setting
    have produced": the per-candidate assignment and merge cost here is
    exactly what ``update_online_alphabet`` always paid, minus the field
    extraction (DFT, cube generation, critical-point search) that dominates a
    normal online-append run's wall-clock time and memory. Nothing here
    touches PySCF, cube files, or the ``CorpusLedger`` -- changing
    ``merge_min_similarity`` no longer needs to imply "reprocess the whole
    corpus". Given several archives -- for example one per parallel shard of
    a corpus too large for one machine to collect sequentially (see the
    control centre's sharding helper and ``iter_datasets_resumable``'s
    ``shard_start``/``shard_stop``) -- this is also the combine step that
    turns them into one alphabet, still without DFT/cube/ledger access for
    any of them. Every archive must share the same ``analysis_signature``
    (checked up front, see ``_matching_extraction_signature``); a single
    archive is trivially "shared with itself".

    Batching matters, not just for speed. ``_merge_connected_prototypes`` runs
    once per ``update_online_alphabet`` call and only ever sees the prototype
    centroids that exist *at that point*, and later assignments compare
    against whatever a previous call's merge already consolidated. Two
    replays under the identical merge policy but different batch boundaries
    can therefore legitimately produce different alphabets -- this is a
    property of the incremental algorithm, not a bug in this function. By
    default (``molecules_per_batch=None``) each archive replays using its own
    *original* ``update_index`` boundaries, which is the only choice
    guaranteed to exactly reproduce a past single-archive run under an
    unchanged policy, regardless of whether ``online_append_chunk_molecules``
    varied across the original run. Pass an explicit ``molecules_per_batch``
    only when you deliberately want to also test the effect of batch size
    itself, or to trade a small amount of path-dependence for a coarser,
    faster pass over a very large archive. With several archives, they are
    replayed in the order given, each fully separated from its neighbours by
    a hard batch boundary (a shard is never silently folded into the tail of
    the previous shard's last in-progress batch); combining the same shards
    in a different order is therefore itself a different, legitimate
    experiment, not noise -- if that sensitivity needs to be measured, rerun
    with a different order and compare, the same way a merge-similarity
    sweep already works.

    The result is written to ``output_state_path``, never to any archive's
    own live ``online_alphabet_state.json`` -- an exploratory recluster must
    never silently replace the alphabet other tools are already reading.
    Promoting a recluster to the production checkpoint is a separate,
    explicit step for the caller.

    ``on_progress(n_molecules_done)``, if given, is called after every
    checkpoint so a caller can drive a progress indicator without polling
    the output file.
    """
    dirs = [online_dirs] if isinstance(online_dirs, (str, Path)) else list(online_dirs)
    if not dirs:
        raise ValueError("online_dirs must contain at least one archive directory")
    resolved_dirs = [Path(item) for item in dirs]
    manifests = [_read_manifest(online_dir) for online_dir in resolved_dirs]
    extraction_signature = _matching_extraction_signature(resolved_dirs, manifests)

    state = new_online_alphabet_state(extraction_signature)
    batch: List[_ArchivedCandidate] = []
    batch_molecule_ids: List[str] = []
    current_molecule_id: Optional[str] = None
    current_update_index: Optional[int] = None
    current_shard_index: Optional[int] = None
    batches_since_checkpoint = 0
    n_molecules_done = 0

    def _flush() -> None:
        nonlocal batch, batch_molecule_ids, batches_since_checkpoint, n_molecules_done
        if not batch_molecule_ids:
            return
        update_online_alphabet(
            state, batch, n_new_molecules=len(batch_molecule_ids),
            merge_enabled=merge_enabled, merge_min_similarity=merge_min_similarity,
            merge_screen_max_prototypes=merge_screen_max_prototypes,
        )
        n_molecules_done += len(batch_molecule_ids)
        batch, batch_molecule_ids = [], []
        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_every_batches:
            save_online_alphabet_state(output_state_path, state)
            batches_since_checkpoint = 0
            if on_progress is not None:
                on_progress(n_molecules_done)

    for shard_index, online_dir in enumerate(resolved_dirs):
        for record, vector in iter_archived_occurrences(online_dir):
            molecule_id = record["molecule_id"]
            update_index = int(record["update_index"])
            new_shard = shard_index != current_shard_index
            starts_new_batch = new_shard or (
                molecule_id != current_molecule_id
                and (
                    (molecules_per_batch is None and update_index != current_update_index)
                    or (molecules_per_batch is not None and len(batch_molecule_ids) >= molecules_per_batch)
                )
            )
            if starts_new_batch:
                _flush()
            if new_shard or molecule_id != current_molecule_id:
                batch_molecule_ids.append(molecule_id)
                current_molecule_id = molecule_id
                current_update_index = update_index
                current_shard_index = shard_index
            batch.append(
                _ArchivedCandidate(molecule_id, record["critical_point_type"], np.asarray(vector, dtype=float))
            )

    _flush()
    save_online_alphabet_state(output_state_path, state)
    if on_progress is not None:
        on_progress(n_molecules_done)
    return state


def recluster_from_archive(
    online_dir: str | Path,
    *,
    output_state_path: str | Path,
    merge_enabled: bool,
    merge_min_similarity: float,
    merge_screen_max_prototypes: int,
    molecules_per_batch: Optional[int] = None,
    checkpoint_every_batches: int = 1,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single-archive form of :func:`recluster_from_archives`; see its
    docstring. Kept as its own name because one archive is the common,
    simplest case, and this signature is exactly what earlier callers
    already depend on -- it delegates rather than duplicating the replay
    loop, so the two can never drift apart.
    """
    return recluster_from_archives(
        [online_dir],
        output_state_path=output_state_path,
        merge_enabled=merge_enabled,
        merge_min_similarity=merge_min_similarity,
        merge_screen_max_prototypes=merge_screen_max_prototypes,
        molecules_per_batch=molecules_per_batch,
        checkpoint_every_batches=checkpoint_every_batches,
        on_progress=on_progress,
    )


def prune_recluster_outputs(
    online_dir: str | Path,
    *,
    keep: int,
    exclude: Optional[Sequence[str | Path]] = None,
) -> List[Path]:
    """Delete old exploratory ``recluster_*.json`` files under ``online_dir``.

    Matches only the ``recluster_*.json`` naming this module's own
    ``recluster_from_archive(s)`` callers write (see ``_run_recluster`` in
    ``esm_master_analytics.py``) -- never ``manifest.json``,
    ``occurrences.jsonl``, ``descriptor_batches/*.npy``, or the live
    ``online_alphabet_state.json``, none of which ever matches that pattern.
    Candidates are sorted by modification time and the ``keep`` most recent
    are retained; anything in ``exclude`` is always retained and does not
    count against ``keep``, so a caller can protect the file it just wrote
    regardless of filesystem mtime resolution. A non-existent ``online_dir``
    or a ``keep`` at or above the number of candidates is a no-op. Returns
    the deleted paths, oldest first; a file that fails to delete (e.g. a
    concurrent reader on some filesystems) is skipped rather than raising,
    since housekeeping should not fail a caller's larger operation.
    """
    online_dir = Path(online_dir)
    if not online_dir.exists():
        return []
    excluded = {Path(item).resolve() for item in (exclude or [])}
    candidates = [
        path for path in online_dir.glob("recluster_*.json")
        if path.is_file() and path.resolve() not in excluded
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime)
    keep = max(0, int(keep))
    to_delete = candidates[: max(0, len(candidates) - keep)]
    deleted: List[Path] = []
    for path in to_delete:
        try:
            path.unlink()
            deleted.append(path)
        except OSError:
            continue
    return deleted