"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Streamlit control centre for resumable ESM corpus analysis.

Run in the same activated Python environment as the main runner:

    streamlit run esm_control_center.py

The interface only exposes registered dataset loaders from ``umcs_datasets``;
it does not execute an arbitrary ``.py`` selected in a browser.  This keeps a
dataset choice explicit and reproducible: the selected loader name, its
arguments and every analytical parameter are written to the named run's
``config.json`` before the background process is launched.

Layout follows the pipeline's own dependency order (doc §14.9: "nothing
works before its turn") instead of a flat list of unrelated operations, and
makes its cost structure explicit rather than implicit, in four tiers --
each one only worth re-paying when something in an EARLIER tier changes:

    Tier 0  shared field cubes             ``shared_fields/<namespace>/``
            DFT: the most expensive step. Reusable by ANY run that shares
            this ``field_cache_namespace`` (grid resolution, level of
            theory, field kind).
    Tier 1  candidate/descriptor extraction ``runs/<id>/candidate_cache``,
            ``runs/<id>/online``
            Critical-point search + invariant descriptors. Expensive, but
            gated by ``AnalysisConfig.analysis_signature_payload`` rather
            than by run_id: unchanged, it is reusable within a run across
            many launches, and combinable across independently-collected
            shard runs (see "Recluster settings") once several runs share
            one signature.
    Tier 2  consolidation / merge policy    online_merge_* settings
            Cheap: "Recluster from archive" replays already-extracted
            candidates under a new threshold -- no DFT, cube, or ledger
            access at all. This is the fast lever for running the SAME
            collected corpus many times under different parameters in
            search of which policy is actually right (doc §16's validation
            epistemology), instead of re-collecting data per hypothesis.
    Tier 3  statistical validation budgets  n_bootstrap_*, n_comm_trials, ...
            Moderate: cache-only re-analysis of already-extracted
            candidates (doc §15 phases 3-6); still no DFT.

Collecting a corpus at the scale doc §19.6 assumes (hundreds of millions to
billions of molecules, from several datasets) needs horizontal, not merely
sequential, throughput: "Collect corpus" below can split one dataset request
into independent, disjoint shards -- see ``iter_datasets_resumable``'s
``shard_start``/``shard_stop`` -- each run under its own ``run_id`` and
later combined losslessly, still without re-touching DFT, through
``esm_core.online_archive.recluster_from_archives``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from esm_core.config import AnalysisConfig, ArtifactPaths, analysis_signature_id
from umcs_datasets import DATASET_DESCRIPTIONS, DATASET_LOCAL_INPUTS, DATASET_REGISTRY


PROJECT_DIR = Path(__file__).resolve().parent

# Display order doubles as the pipeline's dependency order: collect, THEN
# cheaply search for the right consolidation policy against what you already
# collected, THEN spend a full statistical-validation pass, with read-only
# diagnostics available at any time. The underlying strings are unchanged so
# every existing `operation == "..."` comparison keeps working; only the
# on-screen label (via `format_func`) changes.
OPERATION_STAGE_LABELS: Dict[str, str] = {
    "Online corpus append": "① Collect corpus - online append   ·  expensive: DFT + cube generation",
    "Recluster from archive (no DFT)": "② Search for truth - recluster from archive   ·  cheap: no DFT, replay only",
    "Batch validation from cache": "③ Full statistical validation - batch validation from cache   ·  moderate: cache-only",
    "Preliminary field evidence audit": "④ Diagnostics - preliminary field evidence audit   ·  cheap: read-only",
    "Threshold/dictionary diagnostic": "④ Diagnostics - threshold/dictionary diagnostic   ·  moderate: cache-only, can be slow",
    "Nuclear-attractor purity diagnostic": "④ Diagnostics - nuclear-attractor purity   ·  cheap: archive + ledger scan only",
    "Boundary reassignment audit": "④ Diagnostics - boundary reassignment audit   ·  cheap: no DFT, archive + state replay only",
    "Interpret completed batch report": "④ Diagnostics - interpret completed batch report   ·  free: read-only",
}


# ==========================================================================
# Small formatting helpers
# ==========================================================================

def _format_count(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _format_bytes(n: float) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _format_duration_hours(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


# ==========================================================================
# JSON / filesystem helpers
# ==========================================================================

def _wide_dataframe(st: Any, data: Any, **kwargs: Any) -> None:
    """``st.dataframe`` sized to the full column width, tolerant of which
    Streamlit version is installed: ``width="stretch"`` replaced
    ``use_container_width`` during 2025, with the latter now being removed.
    Mirrors this file's existing ``hasattr(st, "fragment")`` tolerance for
    the same kind of cross-version Streamlit API drift.
    """
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tail(path: Path, n_lines: int = 28) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-n_lines:])
    except OSError:
        return "No log has been written yet."


def _process_start_ticks(pid: int) -> Optional[int]:
    """Linux process start time (field 22 of ``/proc/<pid>/stat``, in clock
    ticks since boot) if readable, else ``None``. Two different processes
    can never share both a PID and this value at the same instant, so
    comparing it against a value captured at launch tells a still-running
    original apart from an unrelated process that later reused the same
    PID -- the residual failure mode once a bare PID is checked across a
    full restart of the checking process itself (see ``_is_pid_running``).
    ``None`` on any non-Linux host or unreadable ``/proc`` -- callers treat
    that as "cannot verify identity", not as "not running".
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        # comm (field 2) is parenthesised and may itself contain spaces or
        # ')', so split after the LAST ')' rather than naively on whitespace.
        fields_from_state = raw.rsplit(")", 1)[1].split()
        return int(fields_from_state[19])  # field 22; fields_from_state[0] is field 3
    except (OSError, IndexError, ValueError):
        return None


def _is_pid_running(pid: Any, expected_start_ticks: Any = None) -> bool:
    """Whether ``pid`` is still running -- correctly, not just
    ``os.kill(pid, 0)``, which looks alive for two DIFFERENT wrong reasons
    in this app's actual usage:

    1. A ``subprocess.Popen``'d child that nobody ever calls ``.wait()``/
       ``.poll()`` on becomes a zombie the instant it exits, and a zombie's
       PID keeps answering ``kill(pid, 0)`` as "exists" until reaped --
       which, since ``_launch_background`` never keeps the ``Popen`` object
       around to wait on, meant a finished run looked permanently "active"
       for as long as this Streamlit server process itself stayed up. Fixed
       here with a non-blocking ``os.waitpid(..., WNOHANG)``: it reaps the
       PID (clearing the zombie) if it is our own child and has already
       exited, confirms still-running with nothing to clean up yet if it is
       our own child and alive, and raises ``ChildProcessError`` if it was
       never our child at all (a launch recorded by an earlier server
       process, before a restart) -- handled by falling back to a plain
       existence check for that one case, since it genuinely cannot be
       reaped from here.
    2. A bare PID match -- from either path above -- only proves SOME
       process currently holds that PID, not that it is the SAME process
       that was launched: even ``waitpid`` confirming "this PID is our own
       child, running" only means some child of ours currently has it, not
       that it is the specific launch being asked about, since a PID this
       function has already reaped can be handed by the kernel to a later,
       unrelated child of the same long-lived Streamlit server before
       anything re-reads that older launch's record. On a busy host --
       especially the kind of shared, long-running WSL2 VM this project
       runs under -- that reuse can happen well within one server session,
       not just across a restart. ``expected_start_ticks`` (captured once
       at launch time, see ``_launch_background``) rules this out via
       ``_process_start_ticks`` whenever ``/proc`` makes that possible, and
       is checked after EITHER path confirms a PID match; pass ``None`` to
       skip this check where no such value was recorded.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    exists = False
    try:
        reaped_pid, _status = os.waitpid(pid_int, os.WNOHANG)
        if reaped_pid == pid_int:
            return False  # was our child; just reaped -- it had already exited
        if reaped_pid == 0:
            exists = True  # is our child, confirmed still running (by PID alone -- still subject to the identity check below)
    except ChildProcessError:
        pass  # not (or no longer) our child -- fall through to a plain existence check
    except OSError:
        return False
    if not exists:
        try:
            os.kill(pid_int, 0)
            exists = True
        except OSError:
            return False
    if exists and expected_start_ticks is not None:
        current_ticks = _process_start_ticks(pid_int)
        if current_ticks is not None and current_ticks != int(expected_start_ticks):
            return False  # PID exists (even as our own child), but it is a different process now
    return exists


def _dir_stats(path: Path, pattern: str, limit: int = 100_000) -> Dict[str, Any]:
    """Bounded file count + total size under ``path`` matching ``pattern``.

    Stops after ``limit`` files so this can never turn into an unbounded walk
    over a shared cube cache that, at the corpus sizes this project targets,
    could hold many millions of files. A hit cap is reported as a lower
    bound, not presented as a wrong exact number -- this is deliberately an
    opt-in, explicitly triggered scan (see the "Scan shared cube cache"
    button), not something computed on every page load.
    """
    if not path.exists():
        return {"count": 0, "bytes": 0, "truncated": False}
    count = 0
    total = 0
    truncated = False
    for entry in path.glob(pattern):
        if count >= limit:
            truncated = True
            break
        try:
            total += entry.stat().st_size
            count += 1
        except OSError:
            continue
    return {"count": count, "bytes": total, "truncated": truncated}


# ==========================================================================
# Ledger helpers
# ==========================================================================

def _ledger_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM analysed_molecules WHERE status = 'completed'"
            ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _ledger_stats_for_signature(path: Path, signature_id: str) -> Dict[str, Any]:
    """Completed-row count and time span for ONE analysis_signature.

    Scoped to the signature a launch would actually continue building on --
    a row completed under a different (e.g. since-changed shell_radii)
    signature would need re-extraction, so it should not count as "already
    done" for a throughput projection under the current settings.
    """
    empty = {"count": 0, "first_completed": None, "last_completed": None}
    if not path.exists():
        return empty
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*), MIN(completed_unix_s), MAX(completed_unix_s) "
                "FROM analysed_molecules WHERE analysis_signature = ? AND status = 'completed'",
                (signature_id,),
            ).fetchone()
        if not row or row[0] is None:
            return empty
        count, first, last = row
        return {
            "count": int(count or 0),
            "first_completed": float(first) if first is not None else None,
            "last_completed": float(last) if last is not None else None,
        }
    except sqlite3.Error:
        return empty


def _completed_count_for_dataset(path: Path, signature_id: str, dataset_name: str) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM analysed_molecules "
                "WHERE analysis_signature = ? AND dataset_name = ? AND status = 'completed'",
                (signature_id, dataset_name),
            ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _throughput_projection(stats: Dict[str, Any], target_total: int) -> Optional[Dict[str, float]]:
    count = stats.get("count", 0)
    first, last = stats.get("first_completed"), stats.get("last_completed")
    if count < 2 or first is None or last is None or last <= first:
        return None
    elapsed_hours = (last - first) / 3600.0
    if elapsed_hours <= 0:
        return None
    rate_per_hour = count / elapsed_hours
    if rate_per_hour <= 0:
        return None
    remaining = max(0, int(target_total) - int(count))
    return {
        "rate_per_hour": rate_per_hour,
        "remaining": remaining,
        "eta_hours": remaining / rate_per_hour,
    }


# ==========================================================================
# Path / sharding / cross-run helpers
# ==========================================================================

def _run_paths(root: str, run_id: str, field_cache_namespace: str) -> ArtifactPaths:
    return ArtifactPaths(
        root=Path(root).expanduser(), run_id=run_id,
        field_cache_namespace=field_cache_namespace,
    )


def _even_shards(total: int, n_shards: int) -> List[Tuple[int, int]]:
    """Partition ``[0, total)`` into ``n_shards`` disjoint, gap-free ranges.

    The first ``total % n_shards`` shards get one extra record so every
    record is covered exactly once regardless of divisibility.
    """
    n_shards = max(1, int(n_shards))
    total = max(0, int(total))
    base, remainder = divmod(total, n_shards)
    bounds = []
    start = 0
    for k in range(n_shards):
        size = base + (1 if k < remainder else 0)
        stop = start + size
        bounds.append((start, stop))
        start = stop
    return bounds


def _discover_runs(root: Path, limit: int = 200) -> List[Dict[str, Any]]:
    """Scan ``root/runs/*/config.json`` for a lightweight cross-run summary.

    This is the raw material for the "search for truth" comparison table:
    every run under this artifact root, its key hypothesis parameters, its
    candidate-extraction signature (so the UI can flag which runs' archives
    are directly reusable/combinable), and its current molecule/class
    counts. Best-effort and read-only throughout -- a run with a missing or
    unreadable file is skipped rather than raising, since this only ever
    informs a display panel. Bounded and sorted newest-config-first so a
    long-lived project with many experiments still renders promptly.
    """
    runs_dir = Path(root) / "runs"
    if not runs_dir.exists():
        return []
    entries = []
    for run_dir in runs_dir.iterdir():
        config_path = run_dir / "config.json"
        if run_dir.is_dir() and config_path.exists():
            try:
                entries.append((config_path.stat().st_mtime, run_dir))
            except OSError:
                continue
    entries.sort(key=lambda item: item[0], reverse=True)
    truncated = len(entries) > limit
    out: List[Dict[str, Any]] = []
    for mtime, run_dir in entries[:limit]:
        config = _read_json(run_dir / "config.json")
        if not config:
            continue
        path_info = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
        run_id = str(path_info.get("run_id", run_dir.name))
        namespace = str(path_info.get("field_cache_namespace", "?"))
        signature: Optional[str] = None
        try:
            payload = dict(config)
            payload.pop("paths", None)
            probe_paths = ArtifactPaths(root=Path(root), run_id=run_id, field_cache_namespace=namespace)
            probe_config = AnalysisConfig(paths=probe_paths, **payload)
            signature = analysis_signature_id(probe_config.analysis_signature_payload())
        except (TypeError, ValueError):
            signature = None
        run_paths = ArtifactPaths(root=Path(root), run_id=run_id, field_cache_namespace=namespace)
        state = _read_json(run_paths.online_state_path)
        online_report = _read_json(run_paths.online_report_path)
        metadata = online_report.get("metadata", {}) if isinstance(online_report.get("metadata"), dict) else {}
        n_classes = 0
        if state.get("prototypes_by_type"):
            n_classes = sum(len(v) for v in state["prototypes_by_type"].values())
        elif metadata.get("n_classes"):
            n_classes = int(metadata["n_classes"])
        out.append({
            "run_id": run_id,
            "field_cache_namespace": namespace,
            "descriptor_regime": config.get("descriptor_regime", "?"),
            "online_merge_min_similarity": config.get("online_merge_min_similarity"),
            "online_merge_enabled": config.get("online_merge_enabled"),
            "signature": signature,
            "n_molecules": int(state.get("n_molecules", metadata.get("n_molecules", 0)) or 0),
            "n_candidates": int(state.get("n_candidates", metadata.get("n_candidates", 0)) or 0),
            "n_classes": n_classes,
            "has_archive": (run_paths.online_archive_dir / "manifest.json").exists(),
            "updated_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(mtime)),
        })
    if truncated:
        out.append({"run_id": f"... and {len(entries) - limit} more (not shown)", "field_cache_namespace": "",
                    "descriptor_regime": "", "online_merge_min_similarity": None, "online_merge_enabled": None,
                    "signature": None, "n_molecules": 0, "n_candidates": 0, "n_classes": 0,
                    "has_archive": False, "updated_utc": ""})
    return out


def _validate_config(paths: ArtifactPaths, base_payload: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> AnalysisConfig:
    """Build+validate an AnalysisConfig the same way for every launch path
    (main button and the shard-launch button alike), so they can never
    silently drift into different validation behaviour."""
    payload = dict(base_payload)
    payload.pop("paths", None)
    if overrides:
        payload.update(overrides)
    return AnalysisConfig(paths=paths, **payload)


def _launch_background(
    *,
    paths: ArtifactPaths,
    command: List[str],
    env: Mapping[str, str],
    operation: str,
    datasets: List[Dict[str, Any]] | None = None,
) -> tuple[int, Path]:
    """Start a registered local ESM operation and persist its provenance."""
    paths.launch_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = paths.launch_path.parent / f"{operation}.log"
    popen_options: Dict[str, Any] = {
        "cwd": PROJECT_DIR,
        "env": dict(env),
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }
    # Streamlit can reload this UI process when the interface source changes.
    # The analytical child owns a durable ledger/checkpoint and should not be
    # terminated merely because its browser UI reloads. A full WSL shutdown
    # still stops it, but the cache/ledger resume protocol handles that case.
    if os.name != "nt":
        popen_options["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as stream:
        popen_options["stdout"] = stream
        process = subprocess.Popen(
            command,
            **popen_options,
        )
    start_ticks = _process_start_ticks(process.pid)
    _atomic_json(paths.launch_path, {
        "pid": process.pid,
        "start_ticks": start_ticks,
        "operation": operation,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(paths.config_snapshot_path),
        "datasets": datasets or [],
        "command": command,
        "log": str(log_path),
    })
    return process.pid, log_path


def _render_live_status(st: Any, paths: ArtifactPaths) -> None:
    report = _read_json(paths.online_report_path)
    batch_report = _read_json(paths.report_path)
    preliminary = _read_json(paths.preliminary_report_path)
    state = _read_json(paths.online_state_path)
    launch = _read_json(paths.launch_path)
    extraction = _read_json(paths.extraction_progress_path)
    metadata = report.get("metadata", {}) or batch_report.get("metadata", {})
    n_molecules = int(state.get("n_molecules", metadata.get("n_molecules", 0)) or 0)
    n_candidates = int(state.get("n_candidates", metadata.get("n_candidates", 0)) or 0)
    n_classes = int(metadata.get("n_classes", 0) or 0)
    columns = st.columns(4)
    columns[0].metric("Molecules completed", n_molecules)
    columns[1].metric("Candidates", n_candidates)
    columns[2].metric("Current consolidated classes", n_classes)
    columns[3].metric("Ledger rows", _ledger_count(paths.ledger_path))
    if extraction:
        status = str(extraction.get("status", "unknown"))
        if status in {"chunk_scanned", "extracting_fields", "chunk_ready_for_checkpoint"}:
            chunk_start = extraction.get("chunk_start", "?")
            chunk_stop = extraction.get("chunk_stop_exclusive", "?")
            finished = extraction.get("finished_field_extractions", 0)
            pending = extraction.get("pending_field_extractions", 0)
            st.info(
                f"Field extraction is active: {extraction.get('dataset', 'dataset')}[{chunk_start}:{chunk_stop}], "
                f"workers have finished {finished}/{pending} new fields. "
                f"Already checkpointed in this range: {extraction.get('already_checkpointed_records', 0)}; "
                f"reusable candidate caches: {extraction.get('reusable_candidate_caches', 0)}. "
                "The ledger/alphabet advance only after this bounded transaction is durable."
            )
        elif status in {"online_checkpointed", "online_append_complete", "idle_no_new_records"}:
            st.caption(
                f"Extraction heartbeat: {status}; online total "
                f"{extraction.get('online_molecules_total', n_molecules)} molecules / "
                f"{extraction.get('online_candidates_total', n_candidates)} candidates."
            )
        elif status in {"reclustering", "recluster_complete"}:
            verb = "in progress" if status == "reclustering" else "finished"
            st.info(
                f"Recluster from archive {verb}: {extraction.get('molecules_replayed', 0)} molecules replayed"
                + (f", {extraction.get('n_classes')} classes" if "n_classes" in extraction else "")
                + f". Output: `{extraction.get('output_state_path', 'unknown')}` "
                "(production checkpoint above is untouched unless this run used --promote)."
            )
        else:
            st.caption(f"Extraction heartbeat: {status} (updated {extraction.get('updated_utc', 'unknown')}).")
        st.caption(f"Extraction heartbeat updated: {extraction.get('updated_utc', 'unknown')}")
    if launch:
        running = _is_pid_running(launch.get("pid"), launch.get("start_ticks"))
        st.caption(
            f"Last launch PID: {launch.get('pid')} — {'running' if running else 'finished / unavailable'}; "
            f"started: {launch.get('started_utc', 'unknown')}"
        )
        st.caption(f"Last operation: {launch.get('operation', 'unknown')}")
    if metadata:
        st.caption(
            f"Last append: +{metadata.get('new_molecules_in_last_append', 0)} molecules, "
            f"+{metadata.get('new_candidates_in_last_append', 0)} candidates. "
            f"Prototype merges recorded: {metadata.get('n_prototype_merges', 0)}."
        )
        manifest = metadata.get("visualization_manifest")
        if manifest:
            st.code(f"streamlit run view_esm_alphabet.py -- --manifest {manifest}", language="bash")
    if preliminary:
        candidate_cache = preliminary.get("field_candidate_cache", {})
        alphabet = preliminary.get("online_alphabet", {})
        st.caption(
            "Latest cache-only audit: "
            f"{candidate_cache.get('candidate_sample_size', 0)} sampled field candidates; "
            f"{alphabet.get('n_current_classes', 0)} current online classes."
        )
    log_path = Path(str(launch.get("log"))) if launch.get("log") else paths.launch_path.parent / "esm_master.log"
    st.code(_tail(log_path), language="text")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ESM analysis control centre", layout="wide")
    st.title("ESM control centre")
    st.caption(
        "Field-first corpus analysis, laid out in cost tiers: cubes (Tier 0, shared, most expensive) → "
        "candidate/descriptor extraction (Tier 1, gated by an explicit signature) → consolidation policy "
        "(Tier 2, cheap -- this is where 'search for truth' parameter sweeps belong) → statistical "
        "validation (Tier 3, cache-only). Only a change in an earlier tier should ever force redoing a "
        "later one's work."
    )

    initial = AnalysisConfig.from_environment()
    with st.sidebar:
        st.caption("Campaign identity")
        artifact_root = st.text_input("Analysis artifact root", str(initial.paths.root))
        run_id = st.text_input("Run ID", initial.paths.run_id)
        field_cache_namespace = st.text_input(
            "Reusable field-cache namespace (Tier 0)", initial.paths.field_cache_namespace,
            help=(
                "Change this whenever field generation changes (grid resolution, level of theory, field kind). "
                "A new analysis may share cubes only inside this namespace."
            ),
        )
        st.divider()
        st.caption("Tier 1 · candidate/descriptor extraction — expensive; changing these starts a new signature")
        descriptor_regime = st.selectbox(
            "Descriptor hypothesis", ["local_field", "relational_topology", "balanced_combined"],
            index=["local_field", "relational_topology", "balanced_combined"].index(initial.descriptor_regime),
            help=(
                "Run these as separate hypotheses. None by itself establishes grammar or identity across contexts."
            ),
        )
        st.caption(
            "shell_radii, r0/sigma/l_max and the critical-point search parameters are also Tier 1 "
            "and share this same signature; edit them in Advanced configuration JSON below."
        )
        st.divider()
        st.caption("Tier 2 · consolidation policy — cheap; safe to sweep via 'Search for truth' below")
        merge_enabled = st.checkbox("Enable online transitive prototype consolidation", initial.online_merge_enabled)
        merge_similarity = st.slider(
            "Minimum centroid cosine similarity for a merge edge",
            0.0, 1.0, float(initial.online_merge_min_similarity), 0.01,
            help=(
                "A--C--D--B can consolidate through qualifying edges. Raw old assignments remain immutable aliases; "
                "this is a configurable hypothesis, not a discovered law of the language."
            ),
        )
        st.divider()
        st.caption("Compute resources — orthogonal to every tier above")
        dft_workers = st.number_input("DFT worker processes", 1, 128, initial.dft_workers)
        analysis_jobs = st.number_input("Analysis workers", 1, 128, initial.analysis_n_jobs)
        blas_threads = st.number_input(
            "BLAS threads per process", 1, 128, initial.blas_threads,
            help="Keep this at 1 when more than one DFT worker runs; raise it only for a single DFT worker.",
        )
        append_chunk_molecules = st.number_input(
            "Online append chunk (molecules)", 1, 100000, initial.online_append_chunk_molecules,
            help=(
                "Durable online transaction boundary. 32 is the RAM-safe default: new caches become ledger/checkpoint "
                "progress after a small group of fields rather than after a long 256-molecule wait."
            ),
        )

    paths = _run_paths(artifact_root, run_id, field_cache_namespace)
    root_path = Path(artifact_root).expanduser()
    discovered_runs = _discover_runs(root_path)

    # A lightweight, sidebar-only preview: everything the signature depends
    # on that ISN'T yet exposed as a sidebar widget (shell_radii, cp_*, ...)
    # falls back to environment defaults here. The authoritative signature,
    # including any Advanced JSON edits, is shown again next to "Run
    # configuration" below, right before the launch button.
    try:
        early_preview_config = dataclasses.replace(initial, paths=paths, descriptor_regime=descriptor_regime)
        early_signature_id = analysis_signature_id(early_preview_config.analysis_signature_payload())
    except (TypeError, ValueError):
        early_signature_id = None

    st.subheader("Overview: cost tiers, shared corpus & prior runs")
    overview_cols = st.columns([2, 1])
    with overview_cols[0]:
        st.caption(
            f"Tier 1 signature from current sidebar settings: `{early_signature_id[:16] + '…' if early_signature_id else 'unavailable'}`. "
            "Any other run below sharing this signature has a directly reusable/combinable candidate archive."
        )
        if discovered_runs:
            matches_current = sum(
                1 for r in discovered_runs if r.get("signature") and r["signature"] == early_signature_id
            )
            st.caption(
                f"{len(discovered_runs)} run(s) found under `{root_path}`; {matches_current} share the current Tier-1 signature."
            )
            table_rows = [
                {
                    "run_id": r["run_id"] + (" (current)" if r["run_id"] == run_id else ""),
                    "same signature?": "✅" if (r.get("signature") and r["signature"] == early_signature_id) else "—",
                    "descriptor_regime": r["descriptor_regime"],
                    "merge_similarity": r["online_merge_min_similarity"],
                    "molecules": _format_count(r["n_molecules"]),
                    "candidates": _format_count(r["n_candidates"]),
                    "classes": _format_count(r["n_classes"]),
                    "has archive": "✅" if r["has_archive"] else "—",
                    "config updated (UTC)": r["updated_utc"],
                }
                for r in discovered_runs
            ]
            _wide_dataframe(st, table_rows, hide_index=True)
        else:
            st.caption(f"No runs found yet under `{root_path}` — the one you launch below will be the first.")
    with overview_cols[1]:
        st.caption("Tier 0 · shared field cubes")
        if st.button("Scan shared cube cache now", key="scan_cubes_button", help="Bounded, opt-in filesystem scan (never automatic) so it stays safe at hundreds of millions of cubes."):
            st.session_state["cube_stats"] = _dir_stats(paths.cubes_dir, "*.cube")
            st.session_state["cube_stats_namespace"] = field_cache_namespace
        cube_stats = st.session_state.get("cube_stats")
        if cube_stats is not None and st.session_state.get("cube_stats_namespace") == field_cache_namespace:
            count_label = f"{_format_count(cube_stats['count'])}{'+' if cube_stats['truncated'] else ''}"
            st.metric("Cubes in this namespace", count_label)
            st.caption(f"~{_format_bytes(cube_stats['bytes'])} on disk" + (" (partial, scan capped)" if cube_stats["truncated"] else ""))
        else:
            st.caption(f"Not scanned yet for `{field_cache_namespace}`.")

        st.divider()
        st.caption(f"Reset run `{run_id}`")
        st.caption(
            "Clears this run's mutable state (ledger, online checkpoint, candidate "
            "cache, reports, letter archive) to start a genuinely new analysis. "
            "Never touches `shared_dir`/cubes -- disjoint by construction, checked "
            "again at reset time (`ArtifactPaths.reset_run`)."
        )
        reset_permanent = st.checkbox(
            "Delete permanently instead of archiving aside",
            key="reset_permanent",
            help="Off (default): renames the run's state aside with a timestamp, recoverable. On: shutil.rmtree, not recoverable.",
        )
        reset_confirmed = st.checkbox(f"I understand this clears `{run_id}`'s progress", key="reset_confirmed")
        if st.button("Reset this run now", key="reset_run_button", disabled=not reset_confirmed):
            destination = paths.reset_run(archive=not reset_permanent)
            if destination is not None:
                st.success(f"Archived to `{destination}`. Reload to see the cleared run.")
            elif reset_permanent:
                st.success("Deleted. Reload to see the cleared run.")
            else:
                st.info(f"Nothing to reset -- `{run_id}` had no state yet.")
            st.session_state.pop("cube_stats", None)

    operation = st.radio(
        "Operation",
        list(OPERATION_STAGE_LABELS),
        format_func=lambda key: OPERATION_STAGE_LABELS[key],
        horizontal=False,
        help=(
            "Only the online append changes the live alphabet. Recluster replays already-archived "
            "descriptors under a different merge policy with no DFT/cube/ledger access, and writes to a "
            "new file unless you explicitly ask it to promote. Batch validation and diagnostics are "
            "explicit, separate scientific checks; the preliminary audit is read-only."
        ),
    )
    needs_dataset = operation in {"Online corpus append", "Batch validation from cache"}
    if operation == "Online corpus append" and int(append_chunk_molecules) > 64:
        st.warning(
            f"The selected online transaction has {int(append_chunk_molecules)} molecules. Live totals remain unchanged "
            "until every new field in that transaction finishes. Use 32–64 for frequent crash-safe checkpoints."
        )
    st.subheader("Dataset loaders and selected corpus extent")
    if operation == "Online corpus append":
        st.caption(
            "Choose a registered loader and the number of records for this append. Previously completed records for "
            "this run are skipped by the SQLite ledger; existing field/candidate cache is reused."
        )
    elif operation == "Batch validation from cache":
        st.caption(
            "This mode is cache-only by default: a selected record without a compatible candidate cache is skipped, "
            "never turned into an implicit DFT calculation. Select a bounded already-accumulated range."
        )
    elif operation == "Recluster from archive (no DFT)":
        st.caption(
            "This operation does not select or generate new molecular fields; it reads the descriptor archive "
            "already written by past online-append checkpoints."
        )
    else:
        st.caption("This operation does not select or generate new molecular fields.")
    selected_names = st.multiselect(
        "Dataset loader", sorted(DATASET_REGISTRY), default=["qm9"], disabled=not needs_dataset,
    )
    dataset_request: List[Dict[str, Any]] = []
    for name in selected_names:
        with st.expander(f"{name}: {DATASET_DESCRIPTIONS.get(name, '')}", expanded=name == "qm9"):
            count = st.number_input(
                f"Records from {name}", min_value=1, value=600, step=100,
                key=f"count_{name}", disabled=not needs_dataset,
            )
            # Dedicated, labelled input per local-path/required parameter
            # (DATASET_LOCAL_INPUTS) instead of making the JSON box the
            # only way to set something this load-bearing -- read-only
            # into the pipeline: nothing under `reset_run`'s scope (which
            # only ever touches this run's own `run_dir`) can reach a path
            # entered here, and nothing here ever writes back to it.
            local_input_kwargs: Dict[str, Any] = {}
            local_input_error = False
            for param_name, (label, default) in DATASET_LOCAL_INPUTS.get(name, {}).items():
                value = st.text_input(label, value=default, key=f"local_{name}_{param_name}", disabled=not needs_dataset)
                if not value and default == "" and "required" in label.lower():
                    st.error(f"{label} is required for `{name}`.")
                    local_input_error = True
                if value:
                    local_input_kwargs[param_name] = value
            raw_kwargs = st.text_area(
                "Additional loader keyword arguments (JSON object)", value="{}", key=f"kwargs_{name}", disabled=not needs_dataset,
                help="For anything not already covered above -- e.g. Materials Project API options. Secrets should be supplied through environment variables, not committed here.",
            )
            try:
                kwargs = json.loads(raw_kwargs)
                if not isinstance(kwargs, dict):
                    raise ValueError("must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"Invalid loader JSON: {exc}")
                kwargs = None
            if kwargs is not None and not local_input_error:
                dataset_request.append({"name": name, "n_samples": int(count), **kwargs, **local_input_kwargs})
            loader = DATASET_REGISTRY[name]
            st.caption(f"Registered function: `{loader.__module__}.{loader.__name__}` in `{Path(loader.__code__.co_filename).name}`")
            if needs_dataset and early_signature_id:
                already_done = _completed_count_for_dataset(paths.ledger_path, early_signature_id, name)
                st.caption(f"Already completed under the current Tier-1 signature: {_format_count(already_done)} / {_format_count(int(count))} requested.")

    if needs_dataset and early_signature_id:
        throughput = _throughput_projection(
            _ledger_stats_for_signature(paths.ledger_path, early_signature_id),
            target_total=sum(int(d.get("n_samples", 0) or 0) for d in dataset_request),
        )
        if throughput is not None:
            st.caption(
                f"Observed rate under the current signature: ~{throughput['rate_per_hour']:.1f} molecules/hour. "
                f"At that rate, the remaining {_format_count(throughput['remaining'])} would take "
                f"~{_format_duration_hours(throughput['eta_hours'])} on this run alone (a straight-line "
                "extrapolation from completed-record timestamps, not a guarantee)."
            )

    shard_configs: List[Dict[str, Any]] = []
    launch_shards_locally = False
    n_shards = 1
    if operation == "Online corpus append" and dataset_request:
        with st.expander("Split this request into parallel shards (horizontal scaling across processes/machines)", expanded=False):
            st.caption(
                "Every registered loader re-reads its data from record 0 each time (most have no native "
                "offset/skip parameter), so a shard still requests the same n_samples as the unsharded run and "
                "simply keeps only its own [shard_start:shard_stop) slice -- see iter_datasets_resumable's "
                "docstring. mol_id is derived from the global position in that list, so disjoint shards -- each "
                "run under its OWN run_id, never sharing one run_id's ledger/online state concurrently -- produce "
                "cache files and archived occurrences byte-identical to an unsharded run over the same range, "
                "safe to combine afterwards in 'Search for truth' below."
            )
            n_shards = st.number_input("Number of shards", min_value=1, max_value=256, value=1, key="n_shards")
            if int(n_shards) > 1:
                shard_dft_workers = max(1, int(dft_workers) // int(n_shards))
                st.caption(
                    f"Each shard requests {shard_dft_workers} DFT worker process(es) "
                    f"({int(dft_workers)} sidebar total ÷ {int(n_shards)} shards) so launching them all locally "
                    "does not oversubscribe this machine. Give a shard the full sidebar worker count yourself "
                    "when it will run on its own separate machine instead."
                )
                shard_rows = []
                for k in range(int(n_shards)):
                    shard_run_id = f"{run_id}_shard{k}of{int(n_shards)}"
                    shard_datasets = []
                    total_this_shard = 0
                    for ds in dataset_request:
                        total = int(ds.get("n_samples", 0) or 0)
                        start, stop = _even_shards(total, int(n_shards))[k]
                        shard_ds = dict(ds)
                        shard_ds["shard_start"] = start
                        shard_ds["shard_stop"] = stop
                        shard_datasets.append(shard_ds)
                        total_this_shard += max(0, stop - start)
                    shard_rows.append({"shard": k, "suggested run_id": shard_run_id, "records this shard": total_this_shard})
                    shard_configs.append({
                        "run_id": shard_run_id, "datasets": shard_datasets, "dft_workers": shard_dft_workers,
                    })
                _wide_dataframe(st, shard_rows, hide_index=True)
                st.caption("Full per-shard dataset request + suggested run_id -- copy to another machine, or launch locally below (button lives next to Run configuration, once its settings are validated).")
                st.code(json.dumps(shard_configs, ensure_ascii=False, indent=2), language="json")
                launch_shards_locally = st.checkbox(
                    "I want to launch all shards now as local background processes", value=False, key="launch_shards_locally",
                )

    batch_cache_only = True
    if operation == "Batch validation from cache":
        batch_cache_only = st.checkbox(
            "Use only compatible candidate caches (recommended)", value=True,
            help="Leave enabled to guarantee that validation does not calculate a missing cube or DFT field.",
        )
    if operation == "Threshold/dictionary diagnostic":
        st.warning(
            "This legacy diagnostic reads every compatible candidate cache and performs several potentially expensive "
            "quadratic checks. It neither launches DFT nor changes the online checkpoint, but use it only on a small run."
        )
        diagnostic_confirmed = st.checkbox("I understand that this cache-only diagnostic can require substantial RAM/CPU.")
    else:
        diagnostic_confirmed = False

    if operation == "Nuclear-attractor purity diagnostic":
        st.caption(
            "Joins this run's archived nuclear_attractor occurrences back to each source cube's atom block "
            "(read-only reference annotation, never part of extraction/classification) and reports whether classes "
            "separate cleanly by known element -- the one critical-point type with a near ground-truth floor "
            "(integer nuclear charge). No DFT, no write to the online checkpoint."
        )

    boundary_audit_commit = False
    boundary_audit_label = ""
    boundary_drift_margin = 0.0
    boundary_min_improvement = 0.0
    if operation == "Boundary reassignment audit":
        st.subheader("Boundary audit settings")
        st.caption(
            "Finds archived occurrences whose class's running-mean centroid has drifted past their own distance to "
            "it since they were appended, and proposes a currently-closer class -- not a re-run of the streaming "
            "assignment rule, a check for members stranded by that rule's own centroid drift. Reads "
            "online_alphabet_state.json + the descriptor archive; no DFT, no cube, no CorpusLedger."
        )
        boundary_audit_label = st.text_input(
            "Audit label (recorded on every reassignment this run writes)",
            value=time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()), key="boundary_audit_label",
        )
        boundary_drift_margin = st.number_input(
            "Drift margin", min_value=0.0, value=0.0, step=0.01, key="boundary_drift_margin",
            help="0 flags only occurrences that have actually drifted past their own class's threshold; a small positive value also catches ones close enough to be worth a second look.",
        )
        boundary_min_improvement = st.number_input(
            "Minimum improvement (hysteresis)", min_value=0.0, value=0.0, step=0.01, key="boundary_min_improvement",
            help="A proposed class must be closer by at least this much, not merely closer, so a point near a boundary does not flip on every audit.",
        )
        boundary_audit_commit = st.checkbox(
            "Commit accepted reassignments to boundary_reassignments.jsonl (unchecked: dry-run report only)",
            value=False, key="boundary_audit_commit",
        )

    recluster_online_dirs: List[Path] = []
    if operation == "Recluster from archive (no DFT)":
        st.subheader("Recluster settings")
        archive_options = sorted({run_id} | {r["run_id"] for r in discovered_runs if r["has_archive"]})
        recluster_source_run_ids = st.multiselect(
            "Archive(s) to replay (select more than one to combine independently collected shards)",
            archive_options, default=[run_id], key="recluster_source_run_ids",
        )
        recluster_online_dirs = [
            ArtifactPaths(root=root_path, run_id=rid).online_archive_dir for rid in recluster_source_run_ids
        ]
        missing = [d for d in recluster_online_dirs if not (d / "manifest.json").exists()]
        if not recluster_online_dirs:
            st.warning("Select at least one run with a descriptor archive.")
        elif missing:
            st.warning(
                "No descriptor archive found at: " + ", ".join(f"`{d}`" for d in missing) +
                ". Run 'Online corpus append' at least once for that run ID first."
            )
        elif len(recluster_online_dirs) > 1:
            signatures = {rid: r.get("signature") for rid in recluster_source_run_ids for r in discovered_runs if r["run_id"] == rid}
            if len(set(signatures.values())) > 1:
                st.error(
                    "Selected archives do not all share the same Tier-1 signature -- combining them would silently "
                    "mix incomparable descriptor vectors. The launch below will refuse this; reconcile the "
                    "extraction settings, or drop the mismatched run, before combining."
                )
            else:
                st.success(f"All {len(recluster_online_dirs)} selected archives share the same Tier-1 signature -- safe to combine.")
        st.caption(
            "Independent from the sidebar's live Tier-2 settings above, so testing a threshold here never changes "
            "what a running online append actually does. Reads only occurrences.jsonl and descriptor_batches/*.npy "
            "under each selected run's online/archive/ directory -- no PySCF, no cube file, no CorpusLedger."
        )
        recluster_merge_enabled = st.checkbox(
            "Enable transitive prototype consolidation for this recluster", value=bool(merge_enabled),
            key="recluster_merge_enabled",
        )
        recluster_merge_similarity = st.slider(
            "Merge similarity threshold to test", 0.0, 1.0, float(merge_similarity), 0.01,
            key="recluster_merge_similarity",
            help="Starts from the live sidebar value above, but moving it here only affects this recluster.",
        )
        recluster_molecules_per_batch = st.number_input(
            "Molecules per replay batch (0 = use each archive's original checkpoint boundaries)",
            min_value=0, value=0, key="recluster_molecules_per_batch",
            help=(
                "Leave at 0 unless you deliberately want to test batch size itself: batching is not just about "
                "speed here, it changes which prototypes exist yet when a later candidate is assigned, so a "
                "different batch size -- or a different shard order, when combining several archives -- can "
                "legitimately produce a different alphabet even at the same threshold."
            ),
        )
        recluster_promote = st.checkbox(
            f"Promote result to `{run_id}`'s live checkpoint when finished (overwrites its online_alphabet_state.json)",
            value=False, key="recluster_promote",
        )
        if recluster_promote:
            st.warning(
                f"This will overwrite `{run_id}`'s live online_alphabet_state.json with this recluster's result "
                "when the run finishes. The previous checkpoint is not backed up automatically."
            )
        recluster_keep_outputs = st.number_input(
            "Keep last N recluster outputs (older ones are deleted automatically when this run finishes)",
            min_value=1, value=5, key="recluster_keep_outputs",
            help=(
                "Applies only to files this operation itself wrote (recluster_*.json) -- never an archive or any "
                "live online_alphabet_state.json. Give an --output-state a different name from the command line "
                "if you want a specific result exempt from this cleanup."
            ),
        )
        recluster_output_dir = paths.online_dir if len(recluster_online_dirs) > 1 else (recluster_online_dirs[0].parent if recluster_online_dirs else paths.online_dir)
        past_reclusters = sorted(recluster_output_dir.glob("recluster_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20] if recluster_output_dir.exists() else []
        if past_reclusters:
            with st.expander(f"Past recluster attempts already written to {recluster_output_dir} ({len(past_reclusters)} shown)", expanded=False):
                _wide_dataframe(
                    st,
                    [{"file": p.name, "written (UTC)": time.strftime("%Y-%m-%d %H:%M", time.gmtime(p.stat().st_mtime)), "size": _format_bytes(p.stat().st_size)} for p in past_reclusters],
                    hide_index=True,
                )
    else:
        recluster_merge_enabled = bool(merge_enabled)
        recluster_merge_similarity = float(merge_similarity)
        recluster_molecules_per_batch = 0
        recluster_promote = False
        recluster_keep_outputs = 5


    st.subheader("Run configuration")
    config = initial.to_dict()
    config.update({
        "descriptor_regime": descriptor_regime,
        "execution_mode": "batch_validation" if operation == "Batch validation from cache" else "online",
        "dft_workers": int(dft_workers),
        "analysis_n_jobs": int(analysis_jobs),
        "blas_threads": int(blas_threads),
        "online_merge_enabled": bool(merge_enabled),
        "online_merge_min_similarity": float(merge_similarity),
        "online_append_chunk_molecules": int(append_chunk_molecules),
        "batch_validation_cache_only": bool(batch_cache_only),
            "paths": {
                "root": str(paths.root),
                "run_id": paths.run_id,
                "field_cache_namespace": paths.field_cache_namespace,
        },
    })
    with st.expander("Advanced configuration JSON", expanded=False):
        advanced = st.text_area("All ESM parameters", json.dumps(config, ensure_ascii=False, indent=2), height=360)
    try:
        submitted_config = json.loads(advanced)
        if not isinstance(submitted_config, dict):
            raise ValueError("configuration must be an object")
        submitted_config["paths"] = config["paths"]
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"Advanced configuration is invalid: {exc}")
        submitted_config = None

    if submitted_config is not None:
        try:
            preview_config = _validate_config(paths, submitted_config)
            preview_payload = preview_config.analysis_signature_payload()
            preview_signature_id = analysis_signature_id(preview_payload)
            matches = [r["run_id"] for r in discovered_runs if r.get("signature") == preview_signature_id and r["run_id"] != run_id]
            with st.expander(f"This configuration's Tier-1 signature: `{preview_signature_id[:16]}…`", expanded=False):
                st.json(preview_payload)
                if matches:
                    st.caption(
                        "Matches the candidate-extraction signature of: " + ", ".join(matches) + " -- their "
                        "archives are directly reusable (e.g. combine them in Recluster settings above) without "
                        "repeating extraction."
                    )
                else:
                    st.caption("No other discovered run currently shares this exact signature.")
        except (TypeError, ValueError) as exc:
            st.caption(f"Could not preview this configuration's signature: {exc}")

    config_path = paths.config_snapshot_path
    existing_launch = _read_json(paths.launch_path)
    existing_process_running = bool(existing_launch) and _is_pid_running(existing_launch.get("pid"), existing_launch.get("start_ticks"))
    if existing_process_running:
        st.warning(
            f"Run `{paths.run_id}` already has an active `{existing_launch.get('operation', 'unknown')}` process "
            f"(PID {existing_launch.get('pid')}). Wait for it to checkpoint/finish; do not start a second process "
            "for the same run ID."
        )
    operation_disabled = submitted_config is None
    if operation in {"Online corpus append", "Batch validation from cache"}:
        operation_disabled = operation_disabled or not dataset_request
    if operation == "Threshold/dictionary diagnostic":
        operation_disabled = operation_disabled or not diagnostic_confirmed
    if operation == "Interpret completed batch report":
        operation_disabled = operation_disabled or not paths.report_path.exists()
    if operation == "Recluster from archive (no DFT)":
        operation_disabled = operation_disabled or not recluster_online_dirs or any(
            not (d / "manifest.json").exists() for d in recluster_online_dirs
        )
    # One run has a single writer for the online state and SQLite staging
    # protocol. Read-only reports are also held back while an analytical
    # process is alive, so their output cannot be mistaken for a stable view.
    operation_disabled = operation_disabled or existing_process_running
    action_label = {
        "Online corpus append": "Start / resume online append",
        "Batch validation from cache": "Run bounded batch validation",
        "Recluster from archive (no DFT)": "Recluster from archive (no DFT)",
        "Preliminary field evidence audit": "Write cache-only preliminary audit",
        "Threshold/dictionary diagnostic": "Run cache-only threshold/dictionary diagnostic",
        "Nuclear-attractor purity diagnostic": "Run nuclear-attractor purity diagnostic",
        "Boundary reassignment audit": "Run boundary reassignment audit" + (" (commit)" if boundary_audit_commit else " (dry run)"),
        "Interpret completed batch report": "Interpret completed batch report",
    }[operation]
    if st.button(action_label, type="primary", disabled=operation_disabled):
        # Validate all scalar values using the same class the runner imports;
        # do not enqueue a many-hour DFT job that will fail on a malformed
        # advanced JSON field after it has started.
        try:
            validated = _validate_config(paths, submitted_config)
        except (TypeError, ValueError) as exc:
            st.error(f"Configuration was not launched: {exc}")
            return
        recheck_launch = _read_json(paths.launch_path)
        if _is_pid_running(recheck_launch.get("pid"), recheck_launch.get("start_ticks")):
            st.error("Another process for this run became active; this launch was not started.")
            return
        _atomic_json(config_path, validated.to_dict())
        env = os.environ.copy()
        env["ESM_CONFIG_PATH"] = str(config_path.resolve())
        command: List[str]
        launch_datasets: List[Dict[str, Any]] = []
        if operation in {"Online corpus append", "Batch validation from cache"}:
            env["ESM_DATASETS_JSON"] = json.dumps(dataset_request, ensure_ascii=False)
            command = [sys.executable, "esm_master_analytics.py"]
            launch_datasets = dataset_request
            operation_key = "online_append" if operation == "Online corpus append" else "batch_validation"
        elif operation == "Preliminary field evidence audit":
            command = [sys.executable, "esm_preliminary_analysis.py"]
            operation_key = "preliminary_audit"
        elif operation == "Recluster from archive (no DFT)":
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            similarity_tag = f"{recluster_merge_similarity:.4f}".replace(".", "p")
            combining = len(recluster_online_dirs) > 1
            default_output_dir = paths.online_dir if combining else recluster_online_dirs[0].parent
            output_state_path = default_output_dir / f"recluster_sim{similarity_tag}_{stamp}.json"
            command = [
                sys.executable, "esm_master_analytics.py", "--recluster",
                "--online-dir", *[str(d) for d in recluster_online_dirs],
                "--output-state", str(output_state_path),
                "--merge-min-similarity", str(float(recluster_merge_similarity)),
                "--keep-recluster-outputs", str(int(recluster_keep_outputs)),
            ]
            if not recluster_merge_enabled:
                command.append("--no-merge")
            if int(recluster_molecules_per_batch) > 0:
                command += ["--molecules-per-batch", str(int(recluster_molecules_per_batch))]
            if recluster_promote:
                command.append("--promote")
            operation_key = "recluster"
        elif operation == "Threshold/dictionary diagnostic":
            command = [sys.executable, "diagnose_threshold_and_dictionary.py"]
            operation_key = "threshold_dictionary_diagnostic"
        elif operation == "Nuclear-attractor purity diagnostic":
            command = [
                sys.executable, "diagnose_nuclear_attractor_purity.py",
                "--online-dir", str(paths.online_archive_dir),
                "--ledger", str(paths.ledger_path),
                "--analysis-signature", preview_signature_id,
            ]
            operation_key = "nuclear_attractor_purity_diagnostic"
        elif operation == "Boundary reassignment audit":
            command = [
                sys.executable, "run_boundary_audit.py",
                "--online-dir", str(paths.online_archive_dir),
                "--state-path", str(paths.online_state_path),
                "--audit-label", boundary_audit_label,
                "--drift-margin", str(float(boundary_drift_margin)),
                "--min-improvement", str(float(boundary_min_improvement)),
            ]
            if boundary_audit_commit:
                command.append("--commit")
            operation_key = "boundary_reassignment_audit"
        else:
            digest_path = paths.report_path.parent / "alphabet_interpretation.json"
            command = [
                sys.executable, "interpret_alphabet.py", "--report", str(paths.report_path),
                "--json-out", str(digest_path),
                "--visual-manifest", str(paths.visualization_dir / "manifest.json"),
                "--letter-archive", str(paths.letter_archive_dir),
            ]
            operation_key = "interpret_batch_report"
        pid, log_path = _launch_background(
            paths=paths, command=command, env=env, operation=operation_key, datasets=launch_datasets,
        )
        st.success(f"Started PID {pid}. Log: {log_path}")

    if operation == "Online corpus append" and int(n_shards) > 1 and launch_shards_locally and shard_configs:
        if st.button(f"Launch all {int(n_shards)} shards locally now", disabled=submitted_config is None):
            launched, skipped, failed = [], [], []
            for shard in shard_configs:
                shard_paths = _run_paths(artifact_root, shard["run_id"], field_cache_namespace)
                shard_launch = _read_json(shard_paths.launch_path)
                if _is_pid_running(shard_launch.get("pid"), shard_launch.get("start_ticks")):
                    skipped.append(shard["run_id"])
                    continue
                try:
                    shard_validated = _validate_config(shard_paths, submitted_config, {"dft_workers": shard["dft_workers"]})
                except (TypeError, ValueError) as exc:
                    failed.append(f"{shard['run_id']}: {exc}")
                    continue
                _atomic_json(shard_paths.config_snapshot_path, shard_validated.to_dict())
                shard_env = os.environ.copy()
                shard_env["ESM_CONFIG_PATH"] = str(shard_paths.config_snapshot_path.resolve())
                shard_env["ESM_DATASETS_JSON"] = json.dumps(shard["datasets"], ensure_ascii=False)
                pid, _log_path = _launch_background(
                    paths=shard_paths, command=[sys.executable, "esm_master_analytics.py"],
                    env=shard_env, operation="online_append", datasets=shard["datasets"],
                )
                launched.append(f"{shard['run_id']} (PID {pid})")
            if launched:
                st.success("Launched: " + ", ".join(launched))
            if skipped:
                st.warning("Skipped (already has an active process): " + ", ".join(skipped))
            if failed:
                st.error("Failed to launch: " + "; ".join(failed))

    st.subheader("Live progress")
    st.caption("The panel refreshes every five seconds when the installed Streamlit supports fragments.")
    if hasattr(st, "fragment"):
        @st.fragment(run_every="5s")
        def live_panel() -> None:
            _render_live_status(st, paths)
        live_panel()
    else:  # pragma: no cover - compatibility with old Streamlit
        _render_live_status(st, paths)
        st.info("This Streamlit version has no auto-refresh fragment; reload the page to update the report.")

    st.subheader("Artifact layout")
    st.code(
        "\n".join([
            f"Reusable field cubes (Tier 0, retain across new analyses): {paths.shared_dir}",
            f"Mutable/resumable run data: {paths.run_dir}",
            f"  SQLite ledger: {paths.ledger_path}",
            f"  Online checkpoint + exploratory recluster outputs: {paths.online_dir}",
            f"  Raw append-only descriptor archive (Tier 1, what Recluster reads): {paths.online_archive_dir}",
            f"  Reports: {paths.online_report_path.parent}",
            f"  Current visual snapshot: {paths.visualization_dir}",
            f"All runs sharing this artifact root: {paths.root / 'runs'}  (see 'Overview' above for a comparison)",
        ]),
        language="text",
    )


if __name__ == "__main__":
    main()
