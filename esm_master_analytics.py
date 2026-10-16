"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
esm_master_analytics.py
=======================
End-to-end empirical verification pipeline for Energy Semiotics of Molecules
(theory: energy_semiotics_theory.md §14–§19).

What this runner does
---------------------
1. Multi-dataset DFT density cubes (resumable, versioned cache).
2. In parallel, field-native multi-scale region candidates (no atom/CP seeds)
   are archived; the existing named-class analysis remains a critical-point
   baseline until field-region classes are separately validated.
3. Type-partitioned alphabet clustering with bootstrap stability (§3.2, §15 step 3).
4. Intrinsic dimension (§8.2), fundamentality / MDL dictionary (§12).
5. Unsupervised scrambled-field control (§16.2).
6. Distributional Hypothesis A/B test (§3.1–§3.2).
7. Commutation test per class axis (§11).
8. Isostere 2×2 Campbell–Fiske design + Mantel (§15 step 5).
9. Per-class semantic status / orphans (§16.1 / §16.3).
10. Declension paradigms vs synonymy (§4.4–§4.6).
11. Positional sensitivity from matched build series (§17, best-effort).
12. Full §19 falsification profiles (letter / word / grammar / meaning /
    sentence / alphabet-discovery / synonym / polysemy where data allow).
13. Compact JSON report: theory verdicts first; heavy arrays optional.

Cache invalidation
------------------
Candidate caches are keyed by CACHE_VERSION. Bump CACHE_VERSION whenever
extraction hyperparameters or critical-point logic change (e.g. nuclear
seeding), so old incomplete extractions are not silently reused.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from pyscf import dft, gto
from pyscf.tools import cubegen

from umcs_datasets import load_dataset_records

from esm_core.commutation import CommutationConfig, build_axis_annotations
from esm_core.config import AnalysisConfig, analysis_signature_id as config_analysis_signature_id
from esm_core.corpus_ledger import CorpusLedger
from esm_core.corpus import (
    LetterCandidate,
    assign_candidate_labels,
    class_members,
    cluster_candidates_bootstrap,
    cluster_labels,
    dpmeans_incremental_cluster,
    dpmeans_incremental_snapshots,
    suggest_distance_threshold,
)
from esm_core.dimensionality import levina_bickel_intrinsic_dimension
from esm_core.distributional_test import (
    HypothesisTestConfig,
    build_letter_cooccurrence,
    evaluate_all_word_candidates,
    link_candidates_to_attractors,
    summarize_hypothesis_evidence,
)
from esm_core.expectation_profiles import (
    build_alphabet_discovery_profile,
    build_grammar_profile,
    build_letter_corpus_profile,
    build_polysemy_profile,
    build_sentence_meaning_profile,
    build_synonym_profile,
    build_word_corpus_profile,
    build_word_meaning_profile,
)
from esm_core.analytic_field import AnalyticDensityField
from esm_core.fields import CubeFileField
from esm_core.field_patterns import (
    extract_field_pattern_candidates,
    write_field_pattern_archive,
)
from esm_core.fundamentality import (
    FundamentalityConfig,
    compute_fundamentality,
    diagnose_dictionary_fit,
    fit_shared_dictionary,
    select_dictionary_size_by_cv,
)
from esm_core.isostere import (
    CalibrationItem,
    CalibrationPair,
    SuccessCriteria,
    build_calibration_design,
    run_calibration,
)
from esm_core.letter_archive import write_letter_archive
from esm_core.online_alphabet import (
    load_online_alphabet_state,
    online_prototype_evidence,
    online_state_to_classes,
    resolve_online_class_id,
    save_online_alphabet_state,
    update_online_alphabet,
)
from esm_core.online_archive import append_online_descriptor_batch
from esm_core.paradigm import ParadigmConfig, build_synonym_relations, detect_declension_paradigms
from esm_core.pipeline import extract_letter_candidates
from esm_core.positional_sensitivity import (
    BuildStep,
    BuildTrajectory,
    PositionalSensitivityConfig,
    positional_sensitivity_profile,
)
from esm_core.schema import CooccurrenceEntry, ConfidenceBlock, ForbiddenMapEntry, LetterClass
from esm_core.semantics import build_property_test, update_semantic_status
from esm_core.unsupervised_validation import validate_structure_against_control
from esm_core.visualization import write_visualization_bundle
from esm_core.void_topology import link_critical_points_to_voids, void_topology_profile, write_void_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("esm_master")


# ==========================================================================
# Configuration
# ==========================================================================

CONFIG = AnalysisConfig.from_environment()
# Bump ``cache_version`` only when candidate extraction changes.  A config
# snapshot is stored under the named run before work begins, so two
# descriptor regimes cannot quietly share a cache or online state.
CACHE_VERSION = CONFIG.cache_version
CANDIDATE_DISCOVERY_METHOD = CONFIG.candidate_discovery_method
DESCRIPTOR_REGIME = CONFIG.descriptor_regime
EXECUTION_MODE = CONFIG.execution_mode
ONLINE_STATE_DIR = str(CONFIG.paths.online_dir)
ONLINE_STATE_PATH = str(CONFIG.paths.online_state_path)
ONLINE_ARCHIVE_DIR = str(CONFIG.paths.online_archive_dir)
ONLINE_REPORT_PATH = str(CONFIG.paths.online_report_path)
CORPUS_LEDGER_PATH = str(CONFIG.paths.ledger_path)
EXTRACTION_PROGRESS_PATH = str(CONFIG.paths.extraction_progress_path)

# Generated data never belongs beside source code.  Cubes live in a reusable
# field namespace; all descriptors, checkpoints, SQLite and scenes live in a
# replaceable named analysis run under ``esm_analysis/`` by default.
CUBES_DIR = str(CONFIG.paths.cubes_dir)
CANDS_DIR = str(CONFIG.paths.candidates_dir)
REPORT_PATH = str(CONFIG.paths.report_path)
SUMMARY_PATH = str(CONFIG.paths.summary_path)
VISUALIZATION_DIR = str(CONFIG.paths.visualization_dir)
LETTER_ARCHIVE_DIR = str(CONFIG.paths.letter_archive_dir)
VOID_ARCHIVE_DIR = str(CONFIG.paths.void_archive_dir)
FIELD_PATTERN_PARAMETERS = {
    "method": "field_quantile_excursions_and_bounded_voids_no_cp_or_atom_seeds_v2",
    "quantiles": list(CONFIG.field_pattern_quantiles),
    "min_voxels": CONFIG.field_pattern_min_voxels,
    "max_regions_per_level": CONFIG.field_pattern_max_regions_per_level,
    "max_grid_fraction": CONFIG.field_pattern_max_grid_fraction,
    "include_boundary_components": CONFIG.field_pattern_include_boundary_components,
    "containment_min_fraction": CONFIG.field_pattern_containment_min_fraction,
    "repeated_part_distance": CONFIG.field_pattern_repeated_part_distance,
    "min_part_scale_ratio": CONFIG.field_pattern_min_part_scale_ratio,
    "descriptor_sample_voxels": CONFIG.field_pattern_descriptor_sample_voxels,
    "containment_sample_voxels": CONFIG.field_pattern_containment_sample_voxels,
}
FIELD_PATTERN_CACHE_SIGNATURE = hashlib.sha256(
    json.dumps(FIELD_PATTERN_PARAMETERS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:16]
FIELD_PATTERN_DIR = str(CONFIG.paths.field_pattern_dir / FIELD_PATTERN_CACHE_SIGNATURE)
# Existing v6 caches created before the artifact-layout migration can be
# adopted once into the new layout.  The candidate filename contains both the
# field-first extraction version and descriptor regime; nothing with a
# different signature is copied.  Set ESM_ADOPT_LEGACY_CACHE=0 to force a
# completely fresh field/candidate extraction.
ADOPT_LEGACY_CACHE = CONFIG.adopt_legacy_cache
LEGACY_CUBES_DIR = Path("cube_files_db")
LEGACY_CANDS_DIR = Path("candidates_cache")
# Scene JSON is light-weight (one marker record per extracted candidate), but
# a ten-million-molecule batch may deliberately omit it.  Local density cubes
# are never materialised here; the interactive viewer extracts only the
# selected realisation on demand.
WRITE_VISUALIZATION_BUNDLE = CONFIG.visualization_enabled
WRITE_LETTER_ARCHIVE = CONFIG.letter_archive_enabled
LETTER_ARCHIVE_SHARD_ROWS = CONFIG.letter_archive_shard_rows
# For a pilot, one scene per molecule is convenient.  At corpus scale the
# archive still retains every occurrence, while the interactive bundle is
# capped to representative molecules so it cannot silently duplicate the
# archive into millions of small JSON files.
VISUALIZATION_ALL_MOLECULE_LIMIT = CONFIG.visualization_all_molecule_limit
VISUALIZATION_REPRESENTATIVES_PER_CLASS = CONFIG.visualization_representatives_per_class
VISUALIZATION_REPRESENTATIVE_MOLECULE_LIMIT = CONFIG.visualization_representative_molecule_limit

# Extraction (Å-scale density cubes from PySCF cubegen)
RESOLUTION = CONFIG.resolution
SHELL_RADII = CONFIG.shell_radii
R0 = CONFIG.r0
SIGMA = CONFIG.sigma
L_MAX = CONFIG.l_max
COMPUTE_VOID_TOPOLOGY = CONFIG.compute_void_topology
VOID_TOPOLOGY_QUANTILE_THRESHOLDS = CONFIG.void_topology_quantile_thresholds
N_POINTS_PER_SHELL = CONFIG.n_points_per_shell
FIELD_GRID_MAXIMUM_SEED_LIMIT = CONFIG.field_grid_maximum_seed_limit
CP_KWARGS = {
    "bond_search_radius": CONFIG.cp_bond_search_radius,
    "ring_search": CONFIG.cp_ring_search,
    "n_seeds_per_axis": CONFIG.cp_n_seeds_per_axis,
    "max_step": CONFIG.cp_max_step,
    "grad_tol": CONFIG.cp_grad_tol,
    "dedupe_radius": CONFIG.cp_dedupe_radius,
    "field_grid_maximum_seed_limit": FIELD_GRID_MAXIMUM_SEED_LIMIT,
    "descriptor_regime": DESCRIPTOR_REGIME,
}

# Analysis budgets (pilot-friendly; raise for publication runs)
N_BOOTSTRAP_CLUSTER = CONFIG.n_bootstrap_cluster
N_BOOTSTRAP_DIM = CONFIG.n_bootstrap_dim
N_NULL_PERM = CONFIG.n_null_perm
N_COMM_TRIALS = CONFIG.n_comm_trials
N_CONTROL_MAX = CONFIG.n_control_max
N_CONTROL_REPS = CONFIG.n_control_reps
MAX_PAIRS_PER_QUADRANT = CONFIG.max_pairs_per_quadrant
MAX_CLASSES_FOR_COMMUTATION = CONFIG.max_classes_for_commutation
CORPUS_VERSION = CONFIG.corpus_version

# Resource controls.  They govern approximation precision / temporary
# working sets and are emitted in the report; they never change an a-priori
# significance threshold or a theory verdict rule.
ANALYSIS_N_JOBS = CONFIG.analysis_n_jobs
DICTIONARY_BATCH_SIZE = CONFIG.dictionary_batch_size
DICTIONARY_SCRATCH_MB = CONFIG.dictionary_scratch_mb
DICTIONARY_CV_FOLDS = CONFIG.dictionary_cv_folds
DICTIONARY_CV_MAX_CANDIDATES = CONFIG.dictionary_cv_max_candidates
DICTIONARY_CV_SCREEN_MAX_CANDIDATES = CONFIG.dictionary_cv_screen_max_candidates
DICTIONARY_CV_SCREEN_FOLDS = CONFIG.dictionary_cv_screen_folds
DICTIONARY_CV_FINALISTS = CONFIG.dictionary_cv_finalists
SEMANTIC_MAX_MOLECULES_PER_CLASS = CONFIG.semantic_max_molecules_per_class
PROFILE_COMPRESSION_SAMPLE = CONFIG.profile_compression_sample
USE_GPU_LABEL_ASSIGNMENT = CONFIG.use_gpu_label_assignment
DFT_BACKEND = CONFIG.dft_backend
DFT_WORKERS = CONFIG.dft_workers
BLAS_THREADS = CONFIG.blas_threads
# Similar graphs ordered by size are not physical trajectories.  Keep that
# historical §17 exploration proxy opt-in so a static corpus cannot quietly
# become a purported transition grammar.
ENABLE_STATIC_BUILD_PROXY = CONFIG.enable_static_build_proxy

# Target properties for semantics / isostere (QM9 names)
PRIMARY_PROPERTY = CONFIG.primary_property
SECONDARY_PROPERTIES = CONFIG.secondary_properties

def _datasets_config_from_environment() -> List[Dict[str, Any]]:
    """Read one append request; the SQLite ledger makes prior requests cumulative."""
    raw = os.environ.get("ESM_DATASETS_JSON")
    if not raw:
        return [{"name": "qm9", "n_samples": CONFIG.default_qm9_samples}]
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("ESM_DATASETS_JSON must be a non-empty JSON list of dataset objects")
    out = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each ESM_DATASETS_JSON entry needs a dataset name")
        normalized = dict(item)
        if normalized.get("n_samples") is not None:
            normalized["n_samples"] = max(1, int(normalized["n_samples"]))
        out.append(normalized)
    return out


DATASETS_CONFIG = _datasets_config_from_environment()


# ==========================================================================
# Utilities
# ==========================================================================

def sanitize_for_json(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return sanitize_for_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize_for_json(obj.item())
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return obj


def _cache_key(mol_id: str) -> str:
    # Keep CACHE_VERSION as the final suffix so legacy/export tools can select
    # a report's exact generation, while regime remains part of cache identity.
    return f"{mol_id}_{DESCRIPTOR_REGIME}_{CACHE_VERSION}_cands.pkl"


def _field_pattern_archive_path(mol_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in mol_id).strip("._")
    shard = hashlib.sha256(mol_id.encode("utf-8")).hexdigest()[:2]
    return Path(FIELD_PATTERN_DIR) / shard / f"{safe_id}.json.gz"


def _extract_and_archive_field_patterns(field: Any, mol_id: str, cube_path: str) -> Dict[str, Any]:
    """Run the atom/critical-point-independent proposal pass and persist it."""
    batch = extract_field_pattern_candidates(
        field,
        mol_id,
        quantiles=CONFIG.field_pattern_quantiles,
        min_voxels=CONFIG.field_pattern_min_voxels,
        max_regions_per_level=CONFIG.field_pattern_max_regions_per_level,
        max_grid_fraction=CONFIG.field_pattern_max_grid_fraction,
        include_boundary_components=CONFIG.field_pattern_include_boundary_components,
        containment_min_fraction=CONFIG.field_pattern_containment_min_fraction,
        repeated_part_distance=CONFIG.field_pattern_repeated_part_distance,
        min_part_scale_ratio=CONFIG.field_pattern_min_part_scale_ratio,
        descriptor_sample_voxels=CONFIG.field_pattern_descriptor_sample_voxels,
        containment_sample_voxels=CONFIG.field_pattern_containment_sample_voxels,
    )
    archive_path = write_field_pattern_archive(
        batch,
        FIELD_PATTERN_DIR,
        source_cube=cube_path,
        cache_signature=FIELD_PATTERN_CACHE_SIGNATURE,
    )
    summary = {
        "n_candidates": len(batch.candidates),
        "n_part_whole_relations": len(batch.part_whole_relations),
        "n_repeated_part_groups": len(batch.repeated_part_groups),
        "n_bounded_void_candidates": int(
            batch.diagnostics.get("bounded_complement_void_candidate_count", 0)
        ),
        "archive_path": str(archive_path),
        "diagnostics": batch.diagnostics,
    }
    return summary


def _field_pattern_cache_worker(mol_id: str, cube_path: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Backfill field-region archives from reusable cubes without recomputing DFT."""
    try:
        field = CubeFileField(
            path=cube_path,
            field_kind="density",
            level_of_theory=CONFIG.level_of_theory,
            environment="vacuum",
        )
        return mol_id, _extract_and_archive_field_patterns(field, mol_id, cube_path), None
    except Exception as exc:
        return mol_id, None, f"{type(exc).__name__}: {exc}"


def _update_cached_candidate_meta(cache_file: str, summary: Mapping[str, Any]) -> None:
    """Attach region-archive counts to an existing legacy candidate cache."""
    try:
        with open(cache_file, "rb") as handle:
            candidates, metadata = pickle.load(handle)
        metadata = dict(metadata)
        metadata.update({
            "field_pattern_candidate_count": int(summary["n_candidates"]),
            "field_pattern_part_whole_relation_count": int(summary["n_part_whole_relations"]),
            "field_pattern_repeated_part_group_count": int(summary["n_repeated_part_groups"]),
            "field_pattern_bounded_void_count": int(summary["n_bounded_void_candidates"]),
            "field_pattern_archive_path": str(summary["archive_path"]),
            "field_pattern_error": None,
        })
        temporary = f"{cache_file}.{os.getpid()}.field-pattern-meta.tmp"
        with open(temporary, "wb") as handle:
            pickle.dump((candidates, metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_file)
    except Exception as exc:
        log.warning("Could not refresh field-region counts in candidate cache %s: %s", cache_file, exc)


def _adopt_matching_legacy_artifacts(mol_id: str, cache_file: str, cube_path: str) -> None:
    """Copy an exact legacy v6 cache into the isolated artifact tree once.

    This is a migration convenience, not an equivalence assertion across
    arbitrary old configurations: the regime/version must match the new
    cache key byte for byte.  Copies keep the old workspace untouched and
    prevent an unnecessary second DFT run merely because outputs moved.
    """
    # Legacy workspace cubes had one known generation contract only:
    # DFT/PBE/def2-SVP density on the 0.20 grid.  Never copy them into a
    # user-selected namespace for a changed field/grid merely because the
    # filename happens to match.
    legacy_namespace = "density_pbe_def2svp_res020"
    if (
        not ADOPT_LEGACY_CACHE
        or CONFIG.paths.field_cache_namespace != legacy_namespace
        or not np.isclose(RESOLUTION, 0.20)
    ):
        return
    destination_cache, destination_cube = Path(cache_file), Path(cube_path)
    legacy_cache = LEGACY_CANDS_DIR / _cache_key(mol_id)
    legacy_cube = LEGACY_CUBES_DIR / f"{mol_id}_density.cube"
    if not destination_cube.exists() and legacy_cube.exists():
        destination_cube.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_cube, destination_cube)
        log.info("Adopted legacy field cache for %s into %s", mol_id, destination_cube)
    if not destination_cache.exists() and legacy_cache.exists():
        destination_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_cache, destination_cache)
        log.info("Adopted matching legacy candidate cache for %s into %s", mol_id, destination_cache)


def _analysis_signature_payload() -> Dict[str, Any]:
    """Everything that makes one extracted descriptor comparable to another.

    Thin wrapper: the actual payload is computed by
    ``AnalysisConfig.analysis_signature_payload`` (see its docstring for the
    full rationale), so both this script and the control centre UI compute
    it identically from a single source of truth -- the UI can preview it
    without importing this module's PySCF/RDKit-dependent import graph.
    """
    return CONFIG.analysis_signature_payload()


def _analysis_signature_id(payload: Mapping[str, Any]) -> str:
    return config_analysis_signature_id(payload)


def _publish_extraction_progress(**payload: Any) -> None:
    """Publish a compact heartbeat without touching the scientific state.

    Candidate-cache files are individually durable, while the ledger and
    online alphabet advance only after their bounded transaction is complete.
    This status document prevents the intentionally atomic interval from
    looking like a stalled calculation in the control centre.  It is
    diagnostic only: restart/resume decisions continue to rely solely on the
    cache, SQLite ledger and online checkpoint.
    """
    payload.update({
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "run_id": CONFIG.paths.run_id,
    })
    _write_json_atomically(EXTRACTION_PROGRESS_PATH, payload)


def _hist_entropy(distances: np.ndarray, n_bins: int = 8) -> float:
    d = np.asarray(distances, dtype=float)
    finite = d[np.isfinite(d)]
    if len(finite) < 2:
        return 0.0
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        return 0.0
    scaled = (finite - lo) / span
    try:
        counts, _ = np.histogram(scaled, bins=n_bins, range=(0.0, 1.0))
    except (ValueError, FloatingPointError):
        return 0.0
    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) else 0.0


def _within_class_entropy_bounded(
    candidates: Sequence[LetterCandidate],
    member_indices: Sequence[int],
    centroid: np.ndarray,
    *,
    n_bins: int = 8,
    batch_size: int = 8192,
) -> float:
    """Entropy of class radii without materialising a full large class."""
    if len(member_indices) < 2:
        return 0.0
    lo = np.inf
    hi = -np.inf
    for start in range(0, len(member_indices), batch_size):
        idxs = member_indices[start : start + batch_size]
        vectors = np.stack([candidates[i].descriptor.to_vector() for i in idxs])
        distances = np.linalg.norm(vectors - centroid, axis=1)
        finite = distances[np.isfinite(distances)]
        if len(finite):
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    span = hi - lo
    # `hi > lo` alone doesn't guarantee `np.histogram(..., range=(lo, hi))`
    # can build n_bins distinct edges: at large enough radius magnitude, a
    # nonzero but numerically tiny spread (members whose descriptors are
    # near-identical, e.g. chemically equivalent duplicate structures) makes
    # `np.linspace(lo, hi, n_bins + 1)` collapse adjacent edges to the same
    # float64 value once `span` drops below roughly `1e-13 * scale` -- numpy
    # then raises "Too many bins for data range" instead of silently merging
    # bins. Normalizing to a fixed (0, 1) range first, exactly like
    # `_hist_entropy` above does, sidesteps the magnitude entirely; the
    # try/except is a second line of defense for whatever edge case that
    # normalization itself doesn't cover. A class this numerically tight is,
    # for entropy purposes, indistinguishable from the already-handled
    # `hi <= lo` case -- treat it the same way rather than crashing the
    # whole falsification-profile step over one near-degenerate class.
    counts = np.zeros(n_bins, dtype=np.int64)
    total = 0
    for start in range(0, len(member_indices), batch_size):
        idxs = member_indices[start : start + batch_size]
        vectors = np.stack([candidates[i].descriptor.to_vector() for i in idxs])
        distances = np.linalg.norm(vectors - centroid, axis=1)
        finite = distances[np.isfinite(distances)]
        if len(finite):
            scaled = (finite - lo) / span
            try:
                partial, _ = np.histogram(scaled, bins=n_bins, range=(0.0, 1.0))
            except (ValueError, FloatingPointError):
                return 0.0
            counts += partial
            total += len(finite)
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities))) if len(probabilities) else 0.0


def _reservoir_snapshot_pools(
    candidate_batches: Sequence[Sequence[LetterCandidate]],
    checkpoint_batch_counts: Sequence[int],
    *,
    max_candidates: int,
    rng_seed: int = 0,
) -> List[List[LetterCandidate]]:
    """Uniform bounded candidate samples of growing molecule prefixes.

    §19.7's compression-gain diagnostic needs descriptors at each checkpoint,
    but retaining every prefix duplicates the full candidate corpus several
    times.  Reservoir sampling is one-pass, uniform over each prefix and
    stores at most ``max_candidates`` references per checkpoint.  The caller
    supplies the true prefix population sizes separately, so the MDL
    calculation can still amortise the dictionary over the full corpus.
    """
    checkpoints = list(checkpoint_batch_counts)
    if checkpoints != sorted(checkpoints):
        raise ValueError("checkpoint_batch_counts must be sorted")
    rng = np.random.default_rng(rng_seed)
    reservoir: List[LetterCandidate] = []
    snapshots: List[List[LetterCandidate]] = []
    seen = 0
    next_checkpoint = 0
    for batch_number, batch in enumerate(candidate_batches, start=1):
        for candidate in batch:
            seen += 1
            if len(reservoir) < max_candidates:
                reservoir.append(candidate)
            else:
                replace_at = int(rng.integers(0, seen))
                if replace_at < max_candidates:
                    reservoir[replace_at] = candidate
        while next_checkpoint < len(checkpoints) and batch_number == checkpoints[next_checkpoint]:
            snapshots.append(list(reservoir))
            next_checkpoint += 1
    return snapshots


def _verdict_tally(profile: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = Counter()
    for v in profile.values():
        if isinstance(v, dict) and "verdict" in v:
            counts[v["verdict"]] += 1
        elif hasattr(v, "verdict"):
            counts[getattr(v, "verdict")] += 1
    return dict(counts)


def _dictionary_capacity_limit(n_features: int) -> int:
    """Largest OMP dictionary whose dense per-batch scratch fits the budget.

    Sparse codes make persistent memory O(N*s), but sklearn's OMP still
    materialises a dense batch-by-atom working array.  This is a hardware
    feasibility limit, not a scientific prior; it is explicitly reported and
    an edge-selected CV optimum remains an unresolved capacity warning.
    """
    del n_features  # retained in the interface for report/provenance clarity
    scratch_bytes = DICTIONARY_SCRATCH_MB * 1024 * 1024
    # input/output/work copies: conservative factor 3, float32 units.
    return max(2, scratch_bytes // (3 * DICTIONARY_BATCH_SIZE * np.dtype(np.float32).itemsize))


# ==========================================================================
# Phase 0–1: DFT + letter extraction (resumable)
# ==========================================================================


def _gpu4pyscf_available() -> bool:
    """Whether the optional GPU DFT backend is importable in this runtime."""
    try:
        import importlib.util

        return importlib.util.find_spec("gpu4pyscf") is not None
    except Exception:
        return False


def _build_pyscf_molecule(record: Dict[str, Any]):
    xyz_text = record.get("xyz", "") or ""
    smiles = record.get("smiles", "") or ""
    if xyz_text:
        atom_lines = "\n".join(xyz_text.strip().split("\n")[2:])
        return gto.M(atom=atom_lines, basis="def2-svp", unit="angstrom")
    if not smiles:
        return None
    mol_raw = Chem.MolFromSmiles(smiles)
    if mol_raw is None:
        return None
    mol = Chem.AddHs(mol_raw)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    conformer = mol.GetConformer()
    xyz_list = [
        f"{atom.GetSymbol()} "
        f"{conformer.GetAtomPosition(atom.GetIdx()).x:.5f} "
        f"{conformer.GetAtomPosition(atom.GetIdx()).y:.5f} "
        f"{conformer.GetAtomPosition(atom.GetIdx()).z:.5f}"
        for atom in mol.GetAtoms()
    ]
    return gto.M(atom="; ".join(xyz_list), basis="def2-svp", unit="angstrom")


def _analytic_companion_paths(cube_path: str) -> Tuple[str, str]:
    base = cube_path[:-5] if cube_path.endswith(".cube") else cube_path
    return base + ".mol.json", base + ".dm.npy"


def _write_density_cube(mol_pyscf: Any, cube_path: str, backend: str) -> str:
    """Run identical PBE/def2-SVP SCF on CPU or optional GPU4PySCF."""
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("ESM_DFT_BACKEND must be one of: auto, cpu, gpu")
    selected_backend = "cpu"
    if backend in {"auto", "gpu"}:
        try:
            from gpu4pyscf import dft as gpu_dft  # type: ignore

            mf = gpu_dft.RKS(mol_pyscf)
            selected_backend = "gpu4pyscf"
        except Exception as exc:
            if backend == "gpu":
                raise RuntimeError("GPU DFT was requested but GPU4PySCF is unavailable") from exc
            mf = dft.RKS(mol_pyscf)
    else:
        mf = dft.RKS(mol_pyscf)
    mf.verbose = 0
    mf.xc = "pbe"
    mf.run()
    density_matrix = mf.make_rdm1()
    if hasattr(density_matrix, "get"):
        density_matrix = density_matrix.get()
    cubegen.density(mol_pyscf, cube_path, density_matrix, resolution=RESOLUTION)
    # Saved alongside the cube, not instead of it: the cube stays the
    # portable/legacy artefact (viewer, other tools, EDBench-style adopted
    # cubes that never have a matching mol/dm pair); these two small
    # companions are what let critical-point search use an exact analytic
    # Hessian later (`AnalyticDensityField`) instead of the cube's own
    # finite-difference-on-interpolant fallback, without redoing SCF -- on
    # a cache HIT this function is never called again, so the companions
    # persist across runs exactly as the cube itself already does.
    mol_path, dm_path = _analytic_companion_paths(cube_path)
    try:
        with open(mol_path, "w", encoding="utf-8") as handle:
            handle.write(mol_pyscf.dumps())
        np.save(dm_path, density_matrix)
    except Exception:
        # Best-effort: a failure here must not lose the cube itself, which
        # is the expensive artefact. Analytic-field callers already fall
        # back to CubeFileField's numeric derivatives when a companion is
        # missing (`load_analytic_field_for_cube`).
        pass
    return selected_backend


def load_analytic_field_for_cube(
    cube_path: str, *, level_of_theory: str, environment: str = "vacuum"
) -> Optional[AnalyticDensityField]:
    """`AnalyticDensityField` for `cube_path` if this cube has a saved
    mol/density-matrix pair (see `_write_density_cube`), else None -- the
    caller's job is to fall back to plain `CubeFileField` in that case
    (a legacy cube from before this patch, or one adopted directly from
    another source, e.g. EDBench's Psi4 cubes, which never has a PySCF
    density matrix to pair with)."""
    mol_path, dm_path = _analytic_companion_paths(cube_path)
    if not (os.path.exists(mol_path) and os.path.exists(dm_path)):
        return None
    with open(mol_path, encoding="utf-8") as handle:
        mol = gto.loads(handle.read())
    density_matrix = np.load(dm_path)
    return AnalyticDensityField(mol, density_matrix, level_of_theory=level_of_theory, environment=environment)


def _canonical_smiles(record: Mapping[str, Any]) -> Optional[str]:
    smiles = str(record.get("smiles", "") or "")
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def _extract_record_worker(
    ordinal: int,
    mol_id: str,
    record: Dict[str, Any],
    cp_kwargs: Dict[str, Any],
    dft_backend: str,
) -> Tuple[int, str, Optional[List[LetterCandidate]], Optional[Dict[str, Any]], str, Optional[str]]:
    """Process one uncached record in an isolated worker process."""
    cache_file = os.path.join(CANDS_DIR, _cache_key(mol_id))
    cube_path = os.path.join(CUBES_DIR, f"{mol_id}_density.cube")
    try:
        precomputed_path = record.get("precomputed_cube_path")
        precomputed_lot = record.get("level_of_theory")
        adopted = False
        if precomputed_path:
            # A dataset-supplied cube (e.g. EDBench's Psi4/B3LYP) is only
            # ever adopted when its level_of_theory matches THIS run's
            # configured one exactly -- silently mixing a different level
            # of theory into a cube directory/analysis_signature that
            # claims otherwise is exactly the corruption doc §15/§20.5
            # warn about, not a corner case to paper over.
            if precomputed_lot != CONFIG.level_of_theory:
                log.warning(
                    f"[{mol_id}] precomputed cube's level_of_theory ({precomputed_lot!r}) does not match "
                    f"this run's ({CONFIG.level_of_theory!r}) -- ignoring it and running ordinary DFT instead. "
                    f"To use it, configure a run with level_of_theory={precomputed_lot!r} and a "
                    f"field_cache_namespace/run_id that isn't shared with any other level of theory."
                )
            elif os.path.exists(precomputed_path):
                if not os.path.exists(cube_path):
                    shutil.copyfile(precomputed_path, cube_path)
                adopted = True
            else:
                log.warning(f"[{mol_id}] precomputed_cube_path set but not found on disk: {precomputed_path} -- falling back to DFT")

        if adopted:
            # No PySCF mol/density-matrix pair exists for an adopted cube
            # (it came from Psi4, not this pipeline's own SCF run) --
            # `load_analytic_field_for_cube` below correctly finds no
            # companion files and returns None, so this falls through to
            # CubeFileField's numeric derivatives. Expected, not a bug.
            selected_backend = "adopted_precomputed_cube"
        else:
            if os.path.exists(cube_path):
                # A shared density cube is sufficient for field-first
                # candidate extraction.  Rebuilding an RDKit conformer and
                # PySCF molecule on every candidate-cache miss wasted time
                # even though no DFT or new cube was needed.
                selected_backend = "cached_cube"
            else:
                mol_pyscf = _build_pyscf_molecule(record)
                if mol_pyscf is None:
                    return ordinal, mol_id, None, None, cube_path, "invalid or unembeddable structure"
                mol_pyscf.build()
                selected_backend = _write_density_cube(mol_pyscf, cube_path, dft_backend)
        grid_source = CubeFileField(
            path=cube_path,
            field_kind="density",
            level_of_theory=CONFIG.level_of_theory,
            environment="vacuum",
        )
        analytic = load_analytic_field_for_cube(cube_path, level_of_theory=CONFIG.level_of_theory, environment="vacuum")
        # Exact rho/gradient/Hessian from the basis set when available
        # (removes the cube's own finite-difference-on-interpolant
        # smoothing error from classification entirely); Stage 0b/1 still
        # scan `grid_source`'s sampled grid either way, since a grid is
        # what seeding needs regardless of how derivatives get evaluated.
        field = analytic.paired_with_grid(grid_source) if analytic is not None else grid_source
        field_pattern_summary = None
        field_pattern_error = None
        if CONFIG.field_patterns_enabled:
            try:
                # This proposal family reads the sampled scalar field only.
                # It is separate from, and does not seed from, the CP search.
                field_pattern_summary = _extract_and_archive_field_patterns(
                    grid_source, mol_id, cube_path
                )
                log.info(
                    "[%s] field regions: %d proposals (%d enclosed voids), %d part/whole links, %d repeated-part groups",
                    mol_id,
                    field_pattern_summary["n_candidates"],
                    field_pattern_summary["n_bounded_void_candidates"],
                    field_pattern_summary["n_part_whole_relations"],
                    field_pattern_summary["n_repeated_part_groups"],
                )
            except Exception as region_exc:
                field_pattern_error = f"{type(region_exc).__name__}: {region_exc}"
                log.exception("[%s] field-region candidate extraction failed", mol_id)
        bbox = (
            field.origin,
            field.origin + np.dot(np.array(field.dims) - 1, field.vectors),
        )
        candidates = extract_letter_candidates(
            field,
            bbox,
            mol_id,
            shell_radii=SHELL_RADII,
            r0=R0,
            sigma=SIGMA,
            l_max=L_MAX,
            n_points_per_shell=N_POINTS_PER_SHELL,
            cp_kwargs=cp_kwargs,
        )
        n_void_regions = 0
        void_snapshot_path = None
        if COMPUTE_VOID_TOPOLOGY:
            # Same field, same worker, same pass as the critical-point
            # search above -- not a separately-invoked method (doc §2.3:
            # a parallel layer, computed alongside, never instead of).
            # Its own try/except: void topology failing (e.g. GUDHI on a
            # degenerate all-uniform field) must not discard otherwise-good
            # critical-point candidates by taking down this whole worker.
            try:
                void_snapshots = void_topology_profile(
                    field, molecule_id=mol_id,
                    quantile_thresholds=VOID_TOPOLOGY_QUANTILE_THRESHOLDS,
                )
                links = link_critical_points_to_voids(
                    [c.critical_point for c in candidates],
                    [region for snap in void_snapshots for region in snap.regions],
                    boundary_distance=CONFIG.cp_dedupe_radius,
                )
                void_snapshot_path = str(write_void_snapshots(VOID_ARCHIVE_DIR, mol_id, void_snapshots, links=links))
                n_void_regions = sum(len(snap.regions) for snap in void_snapshots)
            except Exception as void_exc:
                log.warning(f"[{mol_id}] void topology failed, continuing without it: {type(void_exc).__name__}: {void_exc}")
        meta = {
            "smiles": record.get("smiles", "") or "",
            "targets": record.get("targets", {}) or {},
            "bbox": bbox,
            "n_atoms_cube": int(len(field.atom_positions)),
            "cache_version": CACHE_VERSION,
            "candidate_discovery_method": CANDIDATE_DISCOVERY_METHOD,
            "atomic_coordinates_used_to_discover_candidates": False,
            "descriptor_regime": DESCRIPTOR_REGIME,
            "dft_backend": selected_backend,
            "n_candidates": len(candidates),
            "field_pattern_candidate_count": (
                field_pattern_summary["n_candidates"] if field_pattern_summary else 0
            ),
            "field_pattern_part_whole_relation_count": (
                field_pattern_summary["n_part_whole_relations"] if field_pattern_summary else 0
            ),
            "field_pattern_repeated_part_group_count": (
                field_pattern_summary["n_repeated_part_groups"] if field_pattern_summary else 0
            ),
            "field_pattern_bounded_void_count": (
                field_pattern_summary["n_bounded_void_candidates"] if field_pattern_summary else 0
            ),
            "field_pattern_archive_path": (
                field_pattern_summary["archive_path"] if field_pattern_summary else None
            ),
            "field_pattern_error": field_pattern_error,
            "n_void_regions": n_void_regions,
            "void_snapshot_path": void_snapshot_path,
            "cp_types": dict(Counter(c.critical_point.field_type for c in candidates)),
        }
        tmp_cache = f"{cache_file}.{os.getpid()}.tmp"
        with open(tmp_cache, "wb") as f:
            pickle.dump((candidates, meta), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_cache, cache_file)
        return ordinal, mol_id, candidates, meta, cube_path, None
    except Exception as exc:
        return ordinal, mol_id, None, None, cube_path, f"{type(exc).__name__}: {exc}"


def iter_datasets_resumable(
    datasets_config: List[Dict[str, Any]],
    cp_kwargs: Dict[str, Any],
    *,
    workers: int = DFT_WORKERS,
    dft_backend: str = DFT_BACKEND,
    ledger: Optional[CorpusLedger] = None,
    include_completed_cache: bool = True,
    chunk_molecules: int = 32,
    cache_only: bool = False,
) -> Iterator[Tuple[List[LetterCandidate], Dict[str, Dict[str, Any]], List[Tuple], List[Dict[str, Any]]]]:
    """Yield bounded extraction chunks without retaining a corpus in RAM.

    In online mode a ledger row is deliberately committed *after* the
    descriptor batch and alphabet checkpoint are durable.  Therefore a crash
    can at worst leave a reusable candidate cache without a ledger row; the
    next start consumes that cache once, rather than losing or duplicating a
    molecule in the streaming alphabet.  ``chunk_molecules`` also bounds the
    list of worker futures and the one update's descriptor matrix.  A bounded
    retrospective validation can pass ``cache_only=True``: records with no
    compatible candidate cache are reported and skipped, never converted into
    an accidental new DFT calculation.

    Each entry in ``datasets_config`` may also carry optional ``shard_start``
    (default 0) and ``shard_stop`` (default: all loaded records) integer
    keys. These do not change what is *loaded* -- ``n_samples`` still governs
    that, and every loader keeps reading from record 0 as it always has,
    since most registered loaders have no native offset/skip parameter --
    they only change which of the already-loaded records this particular
    process instance actually extracts fields for, by slicing the loaded
    list to ``[shard_start:shard_stop]``. ``mol_id`` is still derived from
    the *global* position in the full ``n_samples`` list
    (``f"{ds_name}_{i:06d}"``), so several processes given disjoint
    ``[shard_start, shard_stop)`` ranges over the same ``n_samples`` request
    -- run under distinct ``run_id``s, one CorpusLedger/candidate_cache/
    online archive each -- produce cache files and archived occurrences that
    never collide and are byte-identical to what a single non-sharded run
    over the same range would have produced. This is what makes combining
    their independently-collected archives with
    ``esm_core.online_archive.recluster_from_archives`` afterwards valid.
    Re-parsing/downloading the shared ``[0, n_samples)`` prefix in every
    shard process is the deliberate, honest cost of this scheme -- it needs
    no change to any individual dataset loader -- and is normally negligible
    next to per-molecule field generation; it is not free for a loader whose
    ``n_samples`` cost is itself dominated by network/IO rather than parsing.
    """
    if chunk_molecules < 1:
        raise ValueError("chunk_molecules must be positive")
    os.makedirs(CUBES_DIR, exist_ok=True)
    os.makedirs(CANDS_DIR, exist_ok=True)

    for ds in datasets_config:
        ds_name = ds["name"]
        n_samples = ds.get("n_samples")
        shard_start = max(0, int(ds.get("shard_start", 0) or 0))
        shard_stop_request = ds.get("shard_stop")
        log.info(
            "=== Dataset %s (n_samples=%s, shard=[%s:%s]) ===",
            ds_name, n_samples, shard_start,
            shard_stop_request if shard_stop_request is not None else n_samples,
        )
        loader_kwargs = {
            key: value for key, value in ds.items()
            if key not in {"name", "n_samples", "shard_start", "shard_stop"}
        }
        records = load_dataset_records(ds_name, n_samples=n_samples, **loader_kwargs)
        shard_stop = (
            len(records) if shard_stop_request is None
            else min(len(records), int(shard_stop_request))
        )
        if shard_start >= shard_stop:
            log.warning(
                "Dataset %s: shard [%d:%d] selects nothing out of %d loaded records; skipping",
                ds_name, shard_start, shard_stop, len(records),
            )
            continue
        for chunk_start in range(shard_start, shard_stop, chunk_molecules):
            chunk_stop = min(shard_stop, chunk_start + chunk_molecules)
            record_pairs = list(enumerate(
                records[chunk_start:chunk_stop], start=chunk_start
            ))
            all_candidates: List[LetterCandidate] = []
            metadata_registry: Dict[str, Dict[str, Any]] = {}
            control_refs: List[Tuple] = []
            ledger_records: List[Dict[str, Any]] = []
            pending: List[Tuple[int, str, Dict[str, Any]]] = []
            field_pattern_backfill: List[Tuple[str, str, Optional[str]]] = []
            completed_in_ledger = 0
            reusable_cache = 0
            for i, rec in record_pairs:
                mol_id = f"{ds_name}_{i:06d}"
                cache_file = os.path.join(CANDS_DIR, _cache_key(mol_id))
                cube_path = os.path.join(CUBES_DIR, f"{mol_id}_density.cube")
                _adopt_matching_legacy_artifacts(mol_id, cache_file, cube_path)
                already_completed = ledger.is_completed(mol_id) if ledger is not None else False

                if already_completed and os.path.exists(cache_file):
                    completed_in_ledger += 1
                    if not include_completed_cache:
                        pattern_path = _field_pattern_archive_path(mol_id)
                        if (
                            CONFIG.field_patterns_enabled
                            and not pattern_path.exists()
                            and os.path.exists(cube_path)
                        ):
                            field_pattern_backfill.append((mol_id, cube_path, cache_file))
                        continue
                    try:
                        with open(cache_file, "rb") as handle:
                            cands, meta = pickle.load(handle)
                        pattern_path = _field_pattern_archive_path(mol_id)
                        if (
                            CONFIG.field_patterns_enabled
                            and not pattern_path.exists()
                            and os.path.exists(cube_path)
                        ):
                            field_pattern_backfill.append((mol_id, cube_path, cache_file))
                        all_candidates.extend(cands)
                        metadata_registry[mol_id] = meta
                        control_refs.append((mol_id, cube_path, meta["bbox"]))
                        continue
                    except Exception as exc:
                        log.warning("Completed-cache read failed %s: %s; rebuilding", cache_file, exc)

                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "rb") as handle:
                            cands, meta = pickle.load(handle)
                        pattern_path = _field_pattern_archive_path(mol_id)
                        if (
                            CONFIG.field_patterns_enabled
                            and not pattern_path.exists()
                            and os.path.exists(cube_path)
                        ):
                            field_pattern_backfill.append((mol_id, cube_path, cache_file))
                        all_candidates.extend(cands)
                        reusable_cache += 1
                        metadata_registry[mol_id] = meta
                        control_refs.append((mol_id, cube_path, meta["bbox"]))
                        ledger_records.append({
                            "molecule_id": mol_id,
                            "dataset_name": ds_name,
                            "dataset_record_index": i,
                            "record": rec,
                            "metadata": meta,
                            "cache_file": cache_file,
                            "cube_path": cube_path,
                        })
                        continue
                    except Exception as exc:
                        log.warning("Cache read failed %s: %s", cache_file, exc)
                if cache_only:
                    pattern_path = _field_pattern_archive_path(mol_id)
                    if (
                        CONFIG.field_patterns_enabled
                        and not pattern_path.exists()
                        and os.path.exists(cube_path)
                    ):
                        field_pattern_backfill.append((mol_id, cube_path, None))
                    log.warning(
                        "Validation cache-only: skipping %s; no readable candidate cache",
                        mol_id,
                    )
                    continue
                pending.append((i, mol_id, rec))

            _publish_extraction_progress(
                status="chunk_scanned",
                dataset=ds_name,
                requested_records=len(records),
                chunk_start=chunk_start,
                chunk_stop_exclusive=chunk_stop,
                records_in_chunk=len(record_pairs),
                already_checkpointed_records=completed_in_ledger,
                reusable_candidate_caches=reusable_cache,
                pending_field_extractions=len(pending),
                finished_field_extractions=0,
                failed_field_extractions=0,
                candidates_ready_for_transaction=len(all_candidates),
                ledger_records_ready=len(ledger_records),
            )

            if field_pattern_backfill:
                backfill_workers = max(1, min(workers, len(field_pattern_backfill)))
                log.info(
                    "Field-region archive backfill %s[%d:%d]: %d existing density cubes, %d worker(s); no DFT or atom/CP seeding",
                    ds_name, chunk_start, chunk_stop, len(field_pattern_backfill), backfill_workers,
                )
                futures = {}
                with ProcessPoolExecutor(max_workers=backfill_workers) as executor:
                    futures = {
                        executor.submit(_field_pattern_cache_worker, mol_id, cube_path): mol_id
                        for mol_id, cube_path, _cache_file in field_pattern_backfill
                    }
                    cache_path_by_molecule = {
                        mol_id: cache_file
                        for mol_id, _cube_path, cache_file in field_pattern_backfill
                        if cache_file is not None
                    }
                    active_futures = set(futures)
                    finished_regions = 0
                    failed_regions = 0
                    backfill_started = time.monotonic()
                    while active_futures:
                        completed, active_futures = wait(
                            active_futures, timeout=30.0, return_when=FIRST_COMPLETED,
                        )
                        if not completed:
                            elapsed = time.monotonic() - backfill_started
                            log.info(
                                "Field-region backfill still active for %s[%d:%d]: %d/%d done, %d failed; elapsed %.0fs",
                                ds_name, chunk_start, chunk_stop, finished_regions,
                                len(field_pattern_backfill), failed_regions, elapsed,
                            )
                            _publish_extraction_progress(
                                status="extracting_field_pattern_regions",
                                dataset=ds_name,
                                chunk_start=chunk_start,
                                chunk_stop_exclusive=chunk_stop,
                                field_pattern_backfill_total=len(field_pattern_backfill),
                                field_pattern_backfill_finished=finished_regions,
                                field_pattern_backfill_failed=failed_regions,
                            )
                            continue
                        for future in completed:
                            try:
                                mol_id, summary, error = future.result()
                            except Exception as exc:
                                mol_id = futures[future]
                                summary = None
                                error = f"worker failure: {type(exc).__name__}: {exc}"
                            finished_regions += 1
                            if error:
                                failed_regions += 1
                                log.error("Field-region backfill failed %s: %s", mol_id, error)
                            else:
                                if mol_id in cache_path_by_molecule:
                                    _update_cached_candidate_meta(
                                        cache_path_by_molecule[mol_id], summary
                                    )
                                if mol_id in metadata_registry:
                                    metadata_registry[mol_id].update({
                                        "field_pattern_candidate_count": int(summary["n_candidates"]),
                                        "field_pattern_part_whole_relation_count": int(summary["n_part_whole_relations"]),
                                        "field_pattern_repeated_part_group_count": int(summary["n_repeated_part_groups"]),
                                        "field_pattern_bounded_void_count": int(summary["n_bounded_void_candidates"]),
                                        "field_pattern_archive_path": str(summary["archive_path"]),
                                        "field_pattern_error": None,
                                    })
                                log.info(
                                    "[%s] field regions: %d proposals (%d enclosed voids), %d part/whole links, %d repeated-part groups; archive=%s",
                                    mol_id, summary["n_candidates"],
                                    summary["n_bounded_void_candidates"],
                                    summary["n_part_whole_relations"],
                                    summary["n_repeated_part_groups"], summary["archive_path"],
                                )
                _publish_extraction_progress(
                    status="field_pattern_backfill_complete",
                    dataset=ds_name,
                    chunk_start=chunk_start,
                    chunk_stop_exclusive=chunk_stop,
                    field_pattern_backfill_total=len(field_pattern_backfill),
                    field_pattern_backfill_finished=len(field_pattern_backfill),
                    field_pattern_backfill_failed=failed_regions,
                )

            if pending:
                effective_workers = max(1, workers)
                if dft_backend == "gpu" or (dft_backend == "auto" and _gpu4pyscf_available()):
                    # Concurrent SCF jobs on one device lead to VRAM fragmentation.
                    effective_workers = 1
                reusable_fields = 0
                for _i, _mol_id, _record in pending:
                    cube_path = os.path.join(CUBES_DIR, f"{_mol_id}_density.cube")
                    supplied_cube = _record.get("precomputed_cube_path")
                    supplied_lot = _record.get("level_of_theory")
                    supplied_cube_usable = (
                        supplied_cube
                        and supplied_lot == CONFIG.level_of_theory
                        and os.path.exists(str(supplied_cube))
                    )
                    if os.path.exists(cube_path) or supplied_cube_usable:
                        reusable_fields += 1
                log.info(
                    "Chunk %s[%d:%d]: %d checkpointed, %d reusable candidate caches, "
                    "%d candidate-cache misses (%d with a reusable density cube; %d need DFT/cube generation), "
                    "%d worker(s), backend=%s",
                    ds_name, chunk_start, chunk_stop, completed_in_ledger, reusable_cache,
                    len(pending), reusable_fields, len(pending) - reusable_fields,
                    effective_workers, dft_backend,
                )
                if effective_workers > 1:
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                        os.environ[name] = "1"
                    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                        futures = {
                            executor.submit(_extract_record_worker, i, mol_id, rec, cp_kwargs, dft_backend): mol_id
                            for i, mol_id, rec in pending
                        }
                        results = []
                        n_finished = 0
                        n_failed = 0
                        progress_stride = max(1, min(10, len(pending) // 10 or 1))
                        active_futures = set(futures)
                        chunk_wait_started = time.monotonic()
                        while active_futures:
                            completed, active_futures = wait(
                                active_futures, timeout=30.0, return_when=FIRST_COMPLETED,
                            )
                            if not completed:
                                elapsed = time.monotonic() - chunk_wait_started
                                log.info(
                                    "Extraction still active for %s[%d:%d]: %d/%d records finished, "
                                    "%d failed; waiting on %d worker task(s), elapsed %.0fs",
                                    ds_name, chunk_start, chunk_stop, n_finished, len(pending),
                                    n_failed, len(active_futures), elapsed,
                                )
                                _publish_extraction_progress(
                                    status="extracting_fields",
                                    dataset=ds_name,
                                    requested_records=len(records),
                                    chunk_start=chunk_start,
                                    chunk_stop_exclusive=chunk_stop,
                                    records_in_chunk=len(record_pairs),
                                    already_checkpointed_records=completed_in_ledger,
                                    reusable_candidate_caches=reusable_cache,
                                    pending_field_extractions=len(pending),
                                    finished_field_extractions=n_finished,
                                    failed_field_extractions=n_failed,
                                    candidates_ready_for_transaction=len(all_candidates),
                                    ledger_records_ready=len(ledger_records),
                                )
                                continue
                            for future in completed:
                                mol_id = futures[future]
                                try:
                                    result = future.result()
                                except Exception as exc:
                                    result = (-1, mol_id, None, None, "", f"worker failure: {type(exc).__name__}: {exc}")
                                results.append(result)
                                n_finished += 1
                                if result[-1] is not None:
                                    n_failed += 1
                            if n_finished % progress_stride == 0 or n_finished == len(pending):
                                _publish_extraction_progress(
                                    status="extracting_fields",
                                    dataset=ds_name,
                                    requested_records=len(records),
                                    chunk_start=chunk_start,
                                    chunk_stop_exclusive=chunk_stop,
                                    records_in_chunk=len(record_pairs),
                                    already_checkpointed_records=completed_in_ledger,
                                    reusable_candidate_caches=reusable_cache,
                                    pending_field_extractions=len(pending),
                                    finished_field_extractions=n_finished,
                                    failed_field_extractions=n_failed,
                                    candidates_ready_for_transaction=len(all_candidates),
                                    ledger_records_ready=len(ledger_records),
                                )
                else:
                    results = []
                    for n_finished, (i, mol_id, rec) in enumerate(pending, start=1):
                        results.append(_extract_record_worker(i, mol_id, rec, cp_kwargs, dft_backend))
                        _publish_extraction_progress(
                            status="extracting_fields",
                            dataset=ds_name,
                            requested_records=len(records),
                            chunk_start=chunk_start,
                            chunk_stop_exclusive=chunk_stop,
                            records_in_chunk=len(record_pairs),
                            already_checkpointed_records=completed_in_ledger,
                            reusable_candidate_caches=reusable_cache,
                            pending_field_extractions=len(pending),
                            finished_field_extractions=n_finished,
                            failed_field_extractions=sum(item[-1] is not None for item in results),
                            candidates_ready_for_transaction=len(all_candidates),
                            ledger_records_ready=len(ledger_records),
                        )
                for _ordinal, mol_id, cands, meta, cube_path, error in sorted(results):
                    if error or cands is None or meta is None:
                        log.error("Pipeline failure %s: %s", mol_id, error)
                        continue
                    all_candidates.extend(cands)
                    metadata_registry[mol_id] = meta
                    control_refs.append((mol_id, cube_path, meta["bbox"]))
                    record_index = int(_ordinal)
                    ledger_records.append({
                        "molecule_id": mol_id,
                        "dataset_name": ds_name,
                        "dataset_record_index": record_index,
                        "record": records[record_index],
                        "metadata": meta,
                        "cache_file": os.path.join(CANDS_DIR, _cache_key(mol_id)),
                        "cube_path": cube_path,
                    })
            _publish_extraction_progress(
                status="chunk_ready_for_checkpoint",
                dataset=ds_name,
                requested_records=len(records),
                chunk_start=chunk_start,
                chunk_stop_exclusive=chunk_stop,
                records_in_chunk=len(record_pairs),
                already_checkpointed_records=completed_in_ledger,
                reusable_candidate_caches=reusable_cache,
                pending_field_extractions=len(pending),
                finished_field_extractions=len(pending),
                failed_field_extractions=sum(item[-1] is not None for item in results) if pending else 0,
                candidates_ready_for_transaction=len(all_candidates),
                ledger_records_ready=len(ledger_records),
            )
            if all_candidates or ledger_records:
                yield all_candidates, metadata_registry, control_refs, ledger_records


def process_datasets_resumable(
    datasets_config: List[Dict[str, Any]],
    cp_kwargs: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[List[LetterCandidate], Dict[str, Dict[str, Any]], List[Tuple], List[Dict[str, Any]]]:
    """Compatibility collector for explicitly bounded batch validation."""
    all_candidates: List[LetterCandidate] = []
    metadata_registry: Dict[str, Dict[str, Any]] = {}
    control_refs: List[Tuple] = []
    ledger_records: List[Dict[str, Any]] = []
    for candidates, metadata, controls, records in iter_datasets_resumable(
        datasets_config, cp_kwargs, **kwargs
    ):
        all_candidates.extend(candidates)
        metadata_registry.update(metadata)
        control_refs.extend(controls)
        ledger_records.extend(records)
    return all_candidates, metadata_registry, control_refs, ledger_records


# ==========================================================================
# Helpers for later phases
# ==========================================================================

def _dominant_bond_candidates(candidates: Sequence[LetterCandidate]) -> Dict[str, LetterCandidate]:
    """Per molecule, bond CP with highest ρ (Bader bond-order proxy)."""
    dominant: Dict[str, LetterCandidate] = {}
    for cand in candidates:
        if "bond" not in cand.critical_point.field_type:
            continue
        cur = dominant.get(cand.source_molecule_id)
        if cur is None or cand.critical_point.value > cur.critical_point.value:
            dominant[cand.source_molecule_id] = cand
    return dominant


def _morgan_fps(metadata_registry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fps = {}
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    for mol_id, meta in metadata_registry.items():
        smi = meta.get("smiles") or ""
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps[mol_id] = gen.GetFingerprint(mol)
    return fps


def _property_scale(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) >= 4:
        q75, q25 = np.percentile(arr, [75, 25])
        scale = float(q75 - q25)
    else:
        scale = float(np.std(arr)) if len(arr) else 1.0
    return scale if scale > 1e-9 else 1.0


def build_isostere_design(
    candidates: Sequence[LetterCandidate],
    metadata_registry: Dict[str, Dict[str, Any]],
    property_name: str,
    distance_threshold: float,
    max_per_quadrant: int = MAX_PAIRS_PER_QUADRANT,
) -> Tuple[Optional[Any], List[CalibrationItem], Dict[str, int]]:
    """§15 step 5 Campbell–Fiske 2×2 with multiple pairs per quadrant."""
    fps = _morgan_fps(metadata_registry)
    dominant = _dominant_bond_candidates(candidates)
    mol_ids = [m for m in fps if m in dominant and property_name in (metadata_registry[m].get("targets") or {})]
    if len(mol_ids) < 4:
        return None, [], {}

    props = {m: float(metadata_registry[m]["targets"][property_name]) for m in mol_ids}
    scale = _property_scale(list(props.values()))

    buckets: Dict[str, List[Tuple[str, str]]] = {
        "similar_graph_similar_function": [],
        "similar_graph_dissimilar_function": [],
        "dissimilar_graph_similar_function": [],
        "dissimilar_graph_dissimilar_function": [],
    }

    for i in range(len(mol_ids)):
        for j in range(i + 1, len(mol_ids)):
            a, b = mol_ids[i], mol_ids[j]
            gsim = DataStructs.TanimotoSimilarity(fps[a], fps[b])
            fdiff = abs(props[a] - props[b]) / scale
            cat = None
            if gsim > 0.7 and fdiff < 0.2:
                cat = "similar_graph_similar_function"
            elif gsim > 0.7 and fdiff > 1.0:
                cat = "similar_graph_dissimilar_function"
            elif gsim < 0.3 and fdiff < 0.2:
                cat = "dissimilar_graph_similar_function"
            elif gsim < 0.3 and fdiff > 1.0:
                cat = "dissimilar_graph_dissimilar_function"
            if cat and len(buckets[cat]) < max_per_quadrant:
                buckets[cat].append((a, b))

    counts = {k: len(v) for k, v in buckets.items()}
    if any(v == 0 for v in counts.values()):
        log.warning("Isostere quadrants incomplete: %s", counts)
        # Still run if mandatory pair present
        if counts["dissimilar_graph_similar_function"] == 0 or counts["similar_graph_dissimilar_function"] == 0:
            return None, [], counts

    item_ids = set()
    pairs: List[CalibrationPair] = []
    for cat, plist in buckets.items():
        for a, b in plist:
            item_ids.add(a)
            item_ids.add(b)
            pairs.append(CalibrationPair(a, b, cat))  # type: ignore[arg-type]

    items = [
        CalibrationItem(
            item_id=mid,
            descriptor=dominant[mid].descriptor,
            functional_property=props[mid],
            property_name=property_name,
            source="qm9",
            label=metadata_registry[mid].get("smiles", ""),
        )
        for mid in item_ids
    ]

    # Success thresholds fixed a priori from clustering scale (§15 step 5).
    success = SuccessCriteria(
        max_distance_for_similar_function=float(distance_threshold),
        min_distance_for_dissimilar_function=float(max(2.0 * distance_threshold, distance_threshold + 0.5)),
        mantel_alpha=0.05,
        mantel_min_correlation=0.3,
    )
    require_full = all(counts[c] > 0 for c in counts)
    design = build_calibration_design(items, pairs, success, require_full_coverage=require_full)
    report = run_calibration(design, n_monte_carlo=2000)
    return report, items, counts


def update_class_semantics(
    classes: Sequence[LetterClass],
    candidates: Sequence[LetterCandidate],
    metadata_registry: Dict[str, Dict[str, Any]],
    property_names: Sequence[str],
    distance_threshold: float,
    corpus_version: str,
    *,
    membership_index: Optional[Mapping[str, Sequence[int]]] = None,
    max_molecules_per_class: int = 2_000,
    n_bootstrap: int = 100,
) -> Dict[str, Any]:
    """§16.1/§16.3: per-class Mantel tests against named molecular properties.

    For each class, take one exemplar descriptor per molecule that contains
    a member, paired with that molecule's measured property. Classes with
    <3 molecules get orphan_confidence score 0 (untested, not confident orphan).
    """
    if max_molecules_per_class < 4:
        raise ValueError("max_molecules_per_class must be at least 4")
    stats = {
        "n_tested": 0,
        "n_named": 0,
        "n_orphan": 0,
        "n_skipped_small": 0,
        "n_exemplars_tested": 0,
        "max_molecules_per_class": max_molecules_per_class,
    }
    for cls in classes:
        members = (
            list(membership_index.get(cls.class_id, ()))
            if membership_index is not None
            else class_members(
                candidates,
                cls,
                distance_threshold=cls.discovery_provenance.persistence_threshold_used,
            )
        )
        # One closest-to-centroid exemplar per molecule.  For a class with
        # millions of molecules, retain the deterministic bottom-k hash
        # sample of molecules rather than building an O(class size) Mantel
        # matrix.  Hash sampling is independent of descriptors/properties,
        # so it changes precision (recorded by n), not the estimand.
        selected: Dict[str, Tuple[int, float, LetterCandidate]] = {}
        # Negative priorities make this a max-heap.  Entries are never
        # duplicated except when a molecule updates its representative,
        # which does not change priority; therefore removal is O(log k).
        worst_heap: List[Tuple[int, str]] = []
        for idx in members:
            c = candidates[idx]
            molecule_id = c.source_molecule_id
            priority = int.from_bytes(
                hashlib.blake2b(
                    f"{cls.class_id}\0{molecule_id}".encode("utf-8"), digest_size=8
                ).digest(),
                "big",
            )
            distance = float(
                np.linalg.norm(c.descriptor.to_vector() - cls.invariant_centroid)
            )
            existing = selected.get(molecule_id)
            if existing is not None:
                if distance < existing[1]:
                    selected[molecule_id] = (priority, distance, c)
                continue
            if len(selected) < max_molecules_per_class:
                selected[molecule_id] = (priority, distance, c)
                heapq.heappush(worst_heap, (-priority, molecule_id))
                continue
            worst_priority = -worst_heap[0][0]
            if priority >= worst_priority:
                continue
            _neg_priority, worst_molecule = heapq.heappop(worst_heap)
            del selected[worst_molecule]
            selected[molecule_id] = (priority, distance, c)
            heapq.heappush(worst_heap, (-priority, molecule_id))

        tests = []
        for prop in property_names:
            items = []
            for mid, (_priority, _distance, cand) in selected.items():
                targets = metadata_registry.get(mid, {}).get("targets") or {}
                if prop not in targets:
                    continue
                items.append(
                    CalibrationItem(
                        item_id=mid,
                        descriptor=cand.descriptor,
                        functional_property=float(targets[prop]),
                        property_name=prop,
                    )
                )
            if len(items) < 4:
                continue
            try:
                tests.append(
                    build_property_test(
                        prop,
                        items,
                        n_boot=n_bootstrap,
                        corpus_version=corpus_version,
                    )
                )
            except Exception as e:
                log.debug("property test failed class=%s prop=%s: %s", cls.class_id, prop, e)

        if not tests:
            stats["n_skipped_small"] += 1
            # leave default orphan=True with n=0 orphan_confidence
            continue

        update_semantic_status(cls, tests, min_correlation=0.3, corpus_version=corpus_version)
        stats["n_tested"] += 1
        stats["n_exemplars_tested"] += len(selected)
        if cls.semantic_status.orphan:
            stats["n_orphan"] += 1
        else:
            stats["n_named"] += 1
    return stats


def build_close_class_pairs(
    classes: Sequence[LetterClass],
    candidates: Sequence[LetterCandidate],
    metadata_registry: Dict[str, Dict[str, Any]],
    property_name: str,
    distance_threshold: float,
) -> Tuple[List[Tuple[LetterClass, LetterClass]], Dict[Tuple[str, str], float]]:
    # The calibration item here is a bond critical point.  Comparing it to a
    # ring/cage/nuclear centroid would reintroduce precisely the type mixing
    # that §3.2 requires the discovery stage to keep separate.
    classes = [
        cls
        for cls in classes
        if cls.discovery_provenance.critical_point_type == "bond_critical_point"
    ]
    dominant = _dominant_bond_candidates(candidates)
    props = {}
    for mid, meta in metadata_registry.items():
        t = meta.get("targets") or {}
        if property_name in t and mid in dominant:
            props[mid] = float(t[property_name])
    if not classes or not props:
        return [], {}

    scale = _property_scale(list(props.values()))
    centroids = np.stack([c.invariant_centroid for c in classes])
    mol_to_class: Dict[str, str] = {}
    for mid, cand in dominant.items():
        if mid not in props:
            continue
        v = cand.descriptor.to_vector()
        d = np.linalg.norm(centroids - v, axis=1)
        mol_to_class[mid] = classes[int(np.argmin(d))].class_id
    class_by_id = {c.class_id: c for c in classes}

    close_pairs = []
    func_distances: Dict[Tuple[str, str], float] = {}
    seen = set()
    mol_ids = list(mol_to_class.keys())
    for i in range(len(mol_ids)):
        for j in range(i + 1, len(mol_ids)):
            a, b = mol_ids[i], mol_ids[j]
            ca, cb = mol_to_class[a], mol_to_class[b]
            if ca == cb:
                continue
            key = tuple(sorted((ca, cb)))
            if key in seen:
                continue
            raw = abs(props[a] - props[b])
            if raw / scale < 0.2:
                seen.add(key)
                func_distances[key] = raw
                close_pairs.append((class_by_id[ca], class_by_id[cb]))
    return close_pairs, func_distances


def build_build_trajectories(
    candidates: Sequence[LetterCandidate],
    classes: Sequence[LetterClass],
    metadata_registry: Dict[str, Dict[str, Any]],
    property_name: str,
    distance_threshold: float,
    max_trajectories: int = 40,
) -> List[BuildTrajectory]:
    """§17 best-effort: size-ordered Tanimoto series as matched build paths.

    Each trajectory is a chain of molecules with high mutual graph similarity,
    ordered by heavy-atom count; steps are nearest letter-class of the
    dominant bond CP, outcomes are the measured property.
    """
    classes = [
        cls
        for cls in classes
        if cls.discovery_provenance.critical_point_type == "bond_critical_point"
    ]
    if not classes:
        return []
    fps = _morgan_fps(metadata_registry)
    dominant = _dominant_bond_candidates(candidates)
    centroids = np.stack([c.invariant_centroid for c in classes])
    class_ids = [c.class_id for c in classes]

    usable = []
    for mid, fp in fps.items():
        if mid not in dominant:
            continue
        t = metadata_registry[mid].get("targets") or {}
        if property_name not in t:
            continue
        smi = metadata_registry[mid].get("smiles") or ""
        mol = Chem.MolFromSmiles(smi) if smi else None
        n_heavy = mol.GetNumHeavyAtoms() if mol is not None else 0
        usable.append((mid, fp, n_heavy, float(t[property_name])))
    if len(usable) < 4:
        return []

    # Greedy chains: start from small molecules, grow by high-Tanimoto larger ones.
    usable.sort(key=lambda x: x[2])
    used = set()
    trajectories: List[BuildTrajectory] = []
    for start in usable:
        if start[0] in used or len(trajectories) >= max_trajectories:
            continue
        chain = [start]
        used.add(start[0])
        cur = start
        while True:
            best = None
            best_sim = 0.55
            for cand in usable:
                if cand[0] in used:
                    continue
                if cand[2] <= cur[2]:
                    continue
                sim = DataStructs.TanimotoSimilarity(cur[1], cand[1])
                if sim > best_sim:
                    best_sim = sim
                    best = cand
            if best is None:
                break
            chain.append(best)
            used.add(best[0])
            cur = best
        if len(chain) < 3:
            for m in chain:
                used.discard(m[0])
            continue
        steps = []
        for mid, _fp, _n, prop in chain:
            v = dominant[mid].descriptor.to_vector()
            j = int(np.argmin(np.linalg.norm(centroids - v, axis=1)))
            steps.append(BuildStep(class_id_added=class_ids[j], outcome_after_step=prop))
        # series_id: this function builds independent greedy chains, each
        # from its own starting molecule -- there is no logic here pairing
        # chains as matched variants of one scaffold (that would need
        # explicit "same scaffold, different substituent" matching, not
        # greedy Tanimoto growth). Honest choice: each chain is its own
        # singleton series for now, not a shared id that would claim a
        # relatedness this function doesn't actually establish.
        tid = f"chain_{len(trajectories)}"
        trajectories.append(BuildTrajectory(trajectory_id=tid, series_id=tid, steps=steps))
    return trajectories


def build_forbidden_proxy(
    candidates: Sequence[LetterCandidate],
    labels: Sequence[Optional[str]],
    letter_class_ids: Sequence[str],
) -> List[ForbiddenMapEntry]:
    """Weak §10/§15-step-9 proxy: letter-class pairs never co-occurring in one
    molecule, while both classes appear in the corpus. Not a true physical
    forbidden map (needs failed trajectories); marked as combinatorial proxy.
    """
    present = set(letter_class_ids)
    co = set()
    by_mol: Dict[str, set] = defaultdict(set)
    for cand, lab in zip(candidates, labels):
        if lab is not None:
            by_mol[cand.source_molecule_id].add(lab)
    n_molecules = len(by_mol)
    # Same "confidence grows asymptotically with corpus size, never reaches
    # certainty" idiom already used for a heuristic, non-statistical score in
    # environment_response.py -- there's no bootstrap/permutation test behind
    # a static corpus-absence proxy, so `method="expert_provisional"` is the
    # honest label rather than implying a real hypothesis test was run.
    proxy_confidence = ConfidenceBlock(
        n=n_molecules,
        method="expert_provisional",
        score=max(0.0, 1.0 - (1.0 / n_molecules)) if n_molecules > 0 else 0.0,
        corpus_version=CORPUS_VERSION,
    )
    for labs in by_mol.values():
        labs = sorted(labs)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                co.add((labs[i], labs[j]))
    ids = sorted(present)
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (a, b) not in co and (b, a) not in co:
                out.append(
                    ForbiddenMapEntry(
                        class_id_pair_or_path=(a, b),
                        forbidden_type="combinatorial",
                        evidence_strength=proxy_confidence,
                        replicated=False,
                    )
                )
    return out


def slim_class(
    cls: LetterClass,
    candidates: Optional[Sequence[LetterCandidate]] = None,
    membership_index: Optional[Mapping[str, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Compact class record, retaining occurrence/reuse evidence.

    The invariant centroid can be large, so it stays out of the report.  The
    occurrence count, molecule count and a few concrete examples must remain:
    without them `interpret_alphabet.py` cannot distinguish a reproducible
    letter from a single-molecule observation or point a reader to a 3-D
    scene.
    """
    record = {
        "class_id": cls.class_id,
        "critical_point_type": cls.discovery_provenance.critical_point_type,
        "persistence_threshold": cls.discovery_provenance.persistence_threshold_used,
        "l_max_bispectrum": cls.discovery_provenance.l_max_bispectrum,
        "cluster_stability": sanitize_for_json(cls.cluster_stability),
        "semantic_status": sanitize_for_json(cls.semantic_status),
        "centroid_dim": int(np.asarray(cls.invariant_centroid).size),
        "centroid_norm": float(np.linalg.norm(cls.invariant_centroid)),
        "identity_scope": {
            "status": "provisional_static_equivalence",
            "claim": (
                "Class membership is descriptor equivalence within the current static "
                "field/calculation regime only; it does not establish persistence across "
                "reaction time, environment, or execution stage."
            ),
            "cross_context_identity_tested": False,
        },
    }
    if candidates is not None and membership_index is not None:
        member_indices = list(membership_index.get(cls.class_id, ()))
        molecule_ids = []
        seen_molecules = set()
        for candidate_index in member_indices:
            molecule_id = candidates[candidate_index].source_molecule_id
            if molecule_id not in seen_molecules:
                seen_molecules.add(molecule_id)
                molecule_ids.append(molecule_id)
        record.update({
            "n_occurrences": len(member_indices),
            "n_molecules": len(seen_molecules),
            "example_molecule_ids": molecule_ids[:8],
        })
    return record


def slim_fund_results(results: Sequence[Any], max_examples: int = 20) -> Dict[str, Any]:
    n_fund = sum(1 for r in results if r.is_fundamental)
    fidelities = [r.reconstruction_fidelity for r in results]
    return {
        "n_candidates": len(results),
        "n_fundamental": n_fund,
        "fraction_fundamental": n_fund / len(results) if results else 0.0,
        "fidelity_mean": float(np.mean(fidelities)) if fidelities else 0.0,
        "fidelity_median": float(np.median(fidelities)) if fidelities else 0.0,
        "examples_fundamental": [
            {
                "source_molecule_id": r.source_molecule_id,
                "reconstruction_fidelity": r.reconstruction_fidelity,
                "n_active_atoms": len(r.active_atoms),
                "reuse_breadth": sanitize_for_json(r.reuse_breadth),
            }
            for r in results
            if r.is_fundamental
        ][:max_examples],
    }


# ==========================================================================
# Main pipeline
# ==========================================================================

def _write_json_atomically(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(dict(payload)), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def _write_online_progress_report(
    state: Mapping[str, Any],
    classes: Sequence[LetterClass],
    signature: Mapping[str, Any],
    n_new_molecules: int,
    n_new_candidates: int,
    visualization_manifest: Optional[Path] = None,
    n_new_field_regions: int = 0,
    n_new_part_whole_relations: int = 0,
    n_new_repeated_part_groups: int = 0,
) -> Path:
    prototype_by_id = {
        prototype["class_id"]: prototype
        for prototypes in state.get("prototypes_by_type", {}).values()
        for prototype in prototypes
    }
    report = {
        "metadata": {
            "execution_mode": "online",
            "cache_version": CACHE_VERSION,
            "analysis_signature": signature,
            "n_molecules": int(state.get("n_molecules", 0)),
            "n_candidates": int(state.get("n_candidates", 0)),
            "n_classes": len(classes),
            "new_molecules_in_last_append": n_new_molecules,
            "new_candidates_in_last_append": n_new_candidates,
            "new_field_region_candidates_in_last_append": int(n_new_field_regions),
            "new_part_whole_relations_in_last_append": int(n_new_part_whole_relations),
            "new_repeated_part_groups_in_last_append": int(n_new_repeated_part_groups),
            "field_first_methodology": {
                "candidate_discovery_method": CANDIDATE_DISCOVERY_METHOD,
                "atomic_coordinates_used_to_discover_candidates": False,
            },
            "descriptor_regime": DESCRIPTOR_REGIME,
            "field_region_candidate_search": {
                "enabled": bool(CONFIG.field_patterns_enabled),
                "method": FIELD_PATTERN_PARAMETERS["method"],
                "archive_dir": FIELD_PATTERN_DIR,
                "cache_signature": FIELD_PATTERN_CACHE_SIGNATURE,
                "uses_atoms_or_critical_points_as_seeds": False,
                "part_whole_and_repeated_subpattern_evidence": "per-molecule provisional evidence; not a word-class verdict",
                "relationship_to_current_alphabet": "field-only proposals are archived separately; the current named-class dictionary and downstream validation still use legacy critical-point candidates",
            },
            "online_state_checkpoint": ONLINE_STATE_PATH,
            "online_descriptor_archive": ONLINE_ARCHIVE_DIR,
            "corpus_ledger": CORPUS_LEDGER_PATH,
            "artifact_root": str(CONFIG.paths.root),
            "run_id": CONFIG.paths.run_id,
            "shared_field_cache": str(CONFIG.paths.shared_dir),
            "online_merge_policy": state.get("merge_policy"),
            "n_prototype_merges": int(state.get("n_prototype_merges", 0)),
            "visualization_manifest": str(visualization_manifest) if visualization_manifest else None,
            "online_status": (
                "Append-only centroid/count update. Existing candidates were not re-extracted, "
                "re-clustered or reassigned. Bootstrap, semantic and grammar phases require an "
                "explicit bounded retrospective validation run."
            ),
        },
        "alphabet_slim": {
            "classes": [
                {
                    "class_id": item.class_id,
                    "critical_point_type": item.discovery_provenance.critical_point_type,
                    "n_occurrences": int(prototype_by_id[item.class_id]["count"]),
                    "n_molecules": online_prototype_evidence(
                        prototype_by_id[item.class_id],
                        float(state.get("threshold_by_type", {}).get(
                            item.discovery_provenance.critical_point_type, 0.0
                        )),
                    )["n_distinct_molecules_estimate"],
                    "cluster_stability": sanitize_for_json(item.cluster_stability),
                    "semantic_status": sanitize_for_json(item.semantic_status),
                    "online_sum_squared_distance": float(
                        prototype_by_id[item.class_id].get("sum_squared_distance", 0.0)
                    ),
                    "evidence": online_prototype_evidence(
                        prototype_by_id[item.class_id],
                        float(state.get("threshold_by_type", {}).get(
                            item.discovery_provenance.critical_point_type, 0.0
                        )),
                    ),
                }
                for item in classes
            ]
        },
        "honest_scope_limits": [
            "This is an append-only online alphabet checkpoint, not a full re-clustering of the accumulated corpus.",
            "Raw historical occurrence assignments are immutable. A current class may consolidate several raw IDs through a recorded, type-specific transitive centroid-similarity graph; no old cube or descriptor is recomputed.",
            "Full bootstrap stability, semantic correlations, graph controls and grammar tests are intentionally not recomputed after every append.",
        ],
    }
    return _write_json_atomically(ONLINE_REPORT_PATH, report)


def _publish_online_visualization_snapshot(
    *,
    state: Mapping[str, Any],
    candidates: Sequence[LetterCandidate],
    labels: Sequence[str],
    metadata_registry: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Atomically extend a bounded cumulative scene index for one run.

    The lossless online archive remains the authoritative corpus record.  A
    browser cannot open tens of millions of scene JSON files, so this is a
    capped cumulative *inspection index*.  It preserves earlier selected
    scenes across appends, includes new scenes while capacity remains, and
    resolves historical class IDs through the checkpoint's alias graph.
    """
    output_dir = Path(VISUALIZATION_DIR)
    manifest_path = output_dir / "manifest.json"
    previous = _load_json_if_exists(manifest_path)
    previous_molecules = {
        str(entry.get("molecule_id")): entry
        for entry in previous.get("molecules", [])
        if isinstance(entry, Mapping) and entry.get("molecule_id")
    }
    capacity = VISUALIZATION_ALL_MOLECULE_LIMIT
    available = max(0, capacity - len(previous_molecules))
    current_ids = sorted({candidate.source_molecule_id for candidate in candidates})
    include_ids = current_ids[:available]
    classes = online_state_to_classes(
        state, l_max_bispectrum=L_MAX, corpus_version=CORPUS_VERSION
    )
    prototype_by_id = {
        str(prototype["class_id"]): prototype
        for prototypes in state.get("prototypes_by_type", {}).values()
        for prototype in prototypes
    }
    class_records = {
        item.class_id: {
            "n_occurrences": int(prototype_by_id[item.class_id].get("count", 0)),
            "n_molecules": online_prototype_evidence(
                prototype_by_id[item.class_id],
                float(state.get("threshold_by_type", {}).get(
                    item.discovery_provenance.critical_point_type, 0.0
                )),
            )["n_distinct_molecules_estimate"],
            "online_sum_squared_distance": float(
                prototype_by_id[item.class_id].get("sum_squared_distance", 0.0)
            ),
            "evidence": online_prototype_evidence(
                prototype_by_id[item.class_id],
                float(state.get("threshold_by_type", {}).get(
                    item.discovery_provenance.critical_point_type, 0.0
                )),
            ),
        }
        for item in classes
    }
    write_visualization_bundle(
        candidates=candidates,
        classes=classes,
        labels=labels,
        metadata_registry=metadata_registry,
        cube_dir=CUBES_DIR,
        output_dir=output_dir,
        class_records=class_records,
        include_molecule_ids=include_ids,
        analysis_context={
            "candidate_discovery_method": CANDIDATE_DISCOVERY_METHOD,
            "atomic_coordinates_used_to_discover_candidates": False,
            "execution_mode": "online_capped_cumulative_snapshot",
            "run_id": CONFIG.paths.run_id,
            "descriptor_regime": DESCRIPTOR_REGIME,
        },
    )
    current = _load_json_if_exists(manifest_path)
    current_molecules = {
        str(entry.get("molecule_id")): entry
        for entry in current.get("molecules", [])
        if isinstance(entry, Mapping) and entry.get("molecule_id")
    }
    previous_molecules.update(current_molecules)
    current["molecules"] = [
        previous_molecules[molecule_id] for molecule_id in sorted(previous_molecules)
    ]
    current["scene_scope"] = "online_capped_cumulative_snapshot"
    current["class_aliases"] = dict(state.get("class_aliases", {}))
    current["n_scene_molecules"] = len(current["molecules"])
    current["scene_molecule_limit"] = capacity
    current.setdefault("notes", []).append(
        "The scene list is a bounded cumulative inspection index; every occurrence remains in the append-only online archive."
    )
    _write_json_atomically(manifest_path, current)
    return manifest_path


def _load_json_if_exists(path: str | Path) -> Dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload) if isinstance(payload, Mapping) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _run_online_append() -> None:
    """Resume bounded corpus chunks without re-reading old candidate data."""
    signature = _analysis_signature_payload()
    ledger = CorpusLedger(CORPUS_LEDGER_PATH, _analysis_signature_id(signature))
    try:
        state = load_online_alphabet_state(ONLINE_STATE_PATH, signature)
        staged_updates = ledger.staged_update_indices()
        for update_index in staged_updates:
            if update_index <= int(state.get("n_updates", 0)):
                # Crash after state publication but before the SQLite commit:
                # finalise without executing the append a second time.
                ledger.complete_staged_update(update_index)
        unresolved_updates = [
            update_index for update_index in ledger.staged_update_indices()
            if update_index > int(state.get("n_updates", 0))
        ]
        if unresolved_updates and unresolved_updates != [int(state.get("n_updates", 0)) + 1]:
            raise RuntimeError(
                "ledger contains more than one unfinished append; resume the oldest matching "
                "dataset request before adding another corpus shard"
            )
        pending_resume_update = unresolved_updates[0] if unresolved_updates else None
        n_new_molecules = 0
        n_new_candidates = 0
        n_new_field_regions = 0
        n_new_part_whole_relations = 0
        n_new_repeated_part_groups = 0
        existing_manifest = Path(VISUALIZATION_DIR) / "manifest.json"
        visualization_manifest: Optional[Path] = (
            existing_manifest if existing_manifest.exists() else None
        )
        appended_any = False
        for candidates, metadata_registry, _control_refs, ledger_records in iter_datasets_resumable(
            DATASETS_CONFIG,
            CP_KWARGS,
            ledger=ledger,
            include_completed_cache=False,
            chunk_molecules=CONFIG.online_append_chunk_molecules,
        ):
            if not ledger_records:
                continue
            if pending_resume_update is not None:
                staged_ids = ledger.staged_molecule_ids(pending_resume_update)
                supplied_ids = {entry["molecule_id"] for entry in ledger_records}
                if staged_ids != supplied_ids:
                    raise RuntimeError(
                        "an interrupted append must be resumed with the same dataset chunk; "
                        "its completed candidate caches are intact, but were not selected this run"
                    )
                update_index = pending_resume_update
            else:
                update_index = int(state["n_updates"]) + 1
            for entry in ledger_records:
                entry["canonical_smiles"] = _canonical_smiles(entry["record"])
            ledger.stage_update(update_index, ledger_records)
            labels = update_online_alphabet(
                state,
                candidates,
                n_new_molecules=len(ledger_records),
                merge_enabled=CONFIG.online_merge_enabled,
                merge_min_similarity=CONFIG.online_merge_min_similarity,
                merge_screen_max_prototypes=CONFIG.online_merge_screen_max_prototypes,
            )
            append_online_descriptor_batch(
                ONLINE_ARCHIVE_DIR,
                update_index=update_index,
                candidates=candidates,
                labels=labels,
                analysis_signature=signature,
                alphabet_state=state,
            )
            save_online_alphabet_state(ONLINE_STATE_PATH, state)
            # The cache, raw descriptor batch and state checkpoint are now
            # durable. Only now does this transaction suppress a future
            # resume. A crash in the gap is recognised by update_index.
            ledger.complete_staged_update(update_index)
            _publish_extraction_progress(
                status="online_checkpointed",
                dataset=ledger_records[0]["dataset_name"] if ledger_records else None,
                online_update_index=update_index,
                molecules_committed_this_update=len(ledger_records),
                candidates_committed_this_update=len(candidates),
                online_molecules_total=int(state.get("n_molecules", 0)),
                online_candidates_total=int(state.get("n_candidates", 0)),
                online_classes_total=sum(
                    len(items) for items in state.get("prototypes_by_type", {}).values()
                ),
            )
            pending_resume_update = None
            appended_any = True
            n_new_molecules += len(ledger_records)
            n_new_candidates += len(candidates)
            n_new_field_regions += sum(
                int(meta.get("field_pattern_candidate_count", 0))
                for meta in metadata_registry.values()
            )
            n_new_part_whole_relations += sum(
                int(meta.get("field_pattern_part_whole_relation_count", 0))
                for meta in metadata_registry.values()
            )
            n_new_repeated_part_groups += sum(
                int(meta.get("field_pattern_repeated_part_group_count", 0))
                for meta in metadata_registry.values()
            )
            if WRITE_VISUALIZATION_BUNDLE:
                visualization_manifest = _publish_online_visualization_snapshot(
                    state=state,
                    candidates=candidates,
                    labels=[resolve_online_class_id(state, label) for label in labels],
                    metadata_registry=metadata_registry,
                )
        if pending_resume_update is not None:
            raise RuntimeError(
                "the interrupted append was not found in this dataset request; "
                "resume with the original source and chunk size"
            )
        if not appended_any:
            log.info("No new molecules for this analysis signature; reading checkpoint only.")
        classes = online_state_to_classes(
            state, l_max_bispectrum=L_MAX, corpus_version=CORPUS_VERSION
        )
        report_path = _write_online_progress_report(
            state, classes, signature, n_new_molecules, n_new_candidates, visualization_manifest,
            n_new_field_regions, n_new_part_whole_relations, n_new_repeated_part_groups,
        )
        _publish_extraction_progress(
            status="idle_no_new_records" if not appended_any else "online_append_complete",
            online_molecules_total=int(state.get("n_molecules", 0)),
            online_candidates_total=int(state.get("n_candidates", 0)),
            online_updates_total=int(state.get("n_updates", 0)),
            online_classes_total=len(classes),
        )
        log.info(
            "Online alphabet checkpoint: +%d molecules / +%d candidates -> %d molecules / %d candidates / %d classes",
            n_new_molecules, n_new_candidates, state.get("n_molecules", 0),
            state.get("n_candidates", 0), len(classes),
        )
        log.info("Online progress report: %s", report_path)
    finally:
        ledger.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ESM analytics pipeline. With no arguments, behaviour is controlled by environment "
            "variables (ESM_EXECUTION_MODE, ESM_CONFIG_PATH, ESM_DATASETS_JSON), matching how "
            "esm_control_center.py launches this script for extraction/online-append/batch-validation. "
            "--recluster is a separate, CLI-only mode: it replays the archived, already-extracted "
            "descriptors under a (possibly new) merge/consolidation policy, with no DFT, no cube "
            "access, and no CorpusLedger touch -- see esm_core.online_archive.recluster_from_archive."
        ),
    )
    parser.add_argument(
        "--recluster", action="store_true",
        help="Replay the descriptor archive under a merge policy instead of running the normal pipeline.",
    )
    parser.add_argument(
        "--online-dir", nargs="+", type=Path, default=None,
        help=(
            "One or more archive directories to read from (default: this config's own "
            "online/ directory, CONFIG.paths.online_dir). Pass several to combine "
            "independently collected shards into one alphabet -- see "
            "esm_core.online_archive.recluster_from_archives; every archive given must "
            "share the same analysis_signature."
        ),
    )
    parser.add_argument(
        "--output-state", type=Path, default=None,
        help=(
            "Where to write the reclustered state. Defaults to a new, uniquely named file inside "
            "--online-dir. Must never be the live online_alphabet_state.json -- refused if it is."
        ),
    )
    parser.add_argument(
        "--merge-min-similarity", type=float, default=None,
        help="Overrides CONFIG.online_merge_min_similarity for this recluster only.",
    )
    parser.add_argument(
        "--merge-screen-max-prototypes", type=int, default=None,
        help="Overrides CONFIG.online_merge_screen_max_prototypes for this recluster only.",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Disable transitive consolidation entirely for this recluster (raw streaming assignment only).",
    )
    parser.add_argument(
        "--molecules-per-batch", type=int, default=None,
        help=(
            "Override the replay batching. Default: replay using the archive's own original "
            "update_index boundaries, the only choice guaranteed to exactly reproduce a past run "
            "under an unchanged policy. A batch boundary genuinely changes the merge outcome, not "
            "just its speed -- see recluster_from_archive's docstring before overriding this."
        ),
    )
    parser.add_argument(
        "--promote", action="store_true",
        help=(
            "After a successful recluster, also overwrite the live production online_alphabet_state.json "
            "with this result. Off by default: promoting a recluster is a separate, explicit decision, "
            "and the previous production checkpoint is not backed up by this script."
        ),
    )
    parser.add_argument(
        "--keep-recluster-outputs", type=int, default=5,
        help=(
            "After this recluster finishes, delete old recluster_*.json files under --online-dir, keeping "
            "this many most-recent ones (the file just written always counts as kept, separately from this "
            "number). Ignored with --no-prune. Default: 5."
        ),
    )
    parser.add_argument(
        "--no-prune", action="store_true",
        help="Skip the automatic cleanup of old recluster_*.json files after this run.",
    )
    parser.add_argument(
        "--prune-recluster-outputs", action="store_true",
        help=(
            "Standalone housekeeping: delete old recluster_*.json files under --online-dir (per "
            "--keep-recluster-outputs) and exit, without running a recluster. Useful from a scheduled "
            "job at a scale where exploratory outputs would otherwise accumulate unattended."
        ),
    )
    return parser


def _run_recluster(args: argparse.Namespace) -> None:
    """CLI entry point for ``--recluster``: see
    esm_core.online_archive.recluster_from_archives (a single ``--online-dir``
    is the common one-archive case of the same function).

    Deliberately does not call ``CONFIG.paths.ensure()``, set BLAS thread
    limits, or touch the CorpusLedger -- none of the machinery this function
    needs is DFT- or extraction-related, which is the entire point.
    """
    from esm_core.online_archive import recluster_from_archives

    online_dirs: List[Path] = (
        [Path(item) for item in args.online_dir] if args.online_dir
        else [CONFIG.paths.online_dir]
    )
    combining = len(online_dirs) > 1
    missing = [d for d in online_dirs if not (d / "manifest.json").exists()]
    if missing:
        log.error(
            "No descriptor archive at %s; nothing to recluster from. The archive is written by "
            "online append, one file per checkpoint -- run 'Online corpus append' at least once "
            "first for every shard being combined.",
            ", ".join(str(d) for d in missing),
        )
        raise SystemExit(1)

    merge_enabled = bool(CONFIG.online_merge_enabled) and not args.no_merge
    merge_min_similarity = (
        float(args.merge_min_similarity) if args.merge_min_similarity is not None
        else float(CONFIG.online_merge_min_similarity)
    )
    merge_screen_max_prototypes = (
        int(args.merge_screen_max_prototypes) if args.merge_screen_max_prototypes is not None
        else int(CONFIG.online_merge_screen_max_prototypes)
    )

    output_state_path = Path(args.output_state) if args.output_state is not None else None
    if output_state_path is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        similarity_tag = f"{merge_min_similarity:.4f}".replace(".", "p")
        # A single archive keeps its output alongside it, exactly as before.
        # Several archives have no one natural home among themselves, so the
        # combined result defaults to this config's own run instead.
        default_dir = online_dirs[0] if not combining else CONFIG.paths.online_dir
        output_state_path = default_dir / f"recluster_sim{similarity_tag}_{stamp}.json"
    if output_state_path.resolve() == Path(ONLINE_STATE_PATH).resolve():
        log.error("--output-state must not be the live online_alphabet_state.json; refusing to overwrite it implicitly.")
        raise SystemExit(1)

    log.info(
        "=== Recluster from %s archive%s: %s (merge_enabled=%s, min_similarity=%.4f, "
        "screen_max_prototypes=%d, molecules_per_batch=%s) ===",
        len(online_dirs), "s" if combining else "",
        ", ".join(str(d) for d in online_dirs), merge_enabled, merge_min_similarity,
        merge_screen_max_prototypes,
        args.molecules_per_batch if args.molecules_per_batch is not None else "original update_index boundaries",
    )

    def _on_progress(n_molecules_done: int) -> None:
        log.info("Recluster progress: %d molecules replayed", n_molecules_done)
        _publish_extraction_progress(
            status="reclustering",
            online_dir=", ".join(str(d) for d in online_dirs),
            output_state_path=str(output_state_path),
            molecules_replayed=n_molecules_done,
        )

    started = time.perf_counter()
    state = recluster_from_archives(
        online_dirs,
        output_state_path=output_state_path,
        merge_enabled=merge_enabled,
        merge_min_similarity=merge_min_similarity,
        merge_screen_max_prototypes=merge_screen_max_prototypes,
        molecules_per_batch=args.molecules_per_batch,
        on_progress=_on_progress,
    )
    elapsed = time.perf_counter() - started

    n_classes = sum(len(prototypes) for prototypes in state["prototypes_by_type"].values())
    log.info(
        "Recluster finished in %.1fs: %d molecules, %d candidates, %d classes across %d types, %d prototype merges.",
        elapsed, state.get("n_molecules", 0), state.get("n_candidates", 0),
        n_classes, len(state["prototypes_by_type"]), state.get("n_prototype_merges", 0),
    )
    log.info("Reclustered state written to %s (production checkpoint untouched).", output_state_path)
    _publish_extraction_progress(
        status="recluster_complete",
        online_dir=", ".join(str(d) for d in online_dirs),
        output_state_path=str(output_state_path),
        molecules_replayed=state.get("n_molecules", 0),
        n_classes=n_classes,
        elapsed_seconds=elapsed,
    )

    if args.promote:
        production_path = Path(ONLINE_STATE_PATH)
        log.warning(
            "--promote: overwriting the LIVE production checkpoint %s with this recluster's result. "
            "The previous checkpoint is not kept by this script; copy it first if you might want it back.",
            production_path,
        )
        save_online_alphabet_state(production_path, state)
        log.info("Promoted. %s now reflects this recluster.", production_path)

    if not args.no_prune:
        from esm_core.online_archive import prune_recluster_outputs
        # output_state_path is protected via `exclude`, independently of
        # mtime, so it does not consume any of the `keep` budget on its own;
        # reserving one slot for it here is what makes the total file count
        # after this run equal --keep-recluster-outputs, not that value plus
        # the file just written. A single source archive is pruned exactly
        # as before; several combined archives have no single natural source
        # to tidy, so the directory that actually received this run's output
        # is pruned instead.
        retain_among_others = max(0, int(args.keep_recluster_outputs) - 1)
        prune_target = online_dirs[0] if not combining else output_state_path.parent
        deleted = prune_recluster_outputs(
            prune_target, keep=retain_among_others, exclude=[output_state_path],
        )
        if deleted:
            log.info(
                "Pruned %d old recluster output(s) under %s, kept %d most recent (plus this run's own output): %s",
                len(deleted), prune_target, retain_among_others,
                ", ".join(path.name for path in deleted),
            )


def _run_prune_only(args: argparse.Namespace) -> None:
    """CLI entry point for ``--prune-recluster-outputs`` without ``--recluster``.

    Standalone housekeeping for a scheduled job: deletes old exploratory
    recluster outputs and exits. Never touches the archive
    (manifest.json/occurrences.jsonl/descriptor_batches/) or the production
    online_alphabet_state.json -- see prune_recluster_outputs's docstring for
    exactly which files can be matched.
    """
    from esm_core.online_archive import prune_recluster_outputs

    online_dirs = (
        [Path(item) for item in args.online_dir] if args.online_dir
        else [CONFIG.paths.online_dir]
    )
    for online_dir in online_dirs:
        if not online_dir.exists():
            log.info("Nothing to prune: %s does not exist.", online_dir)
            continue
        deleted = prune_recluster_outputs(online_dir, keep=int(args.keep_recluster_outputs))
        if deleted:
            log.info(
                "Pruned %d old recluster output(s) under %s, kept %d most recent: %s",
                len(deleted), online_dir, int(args.keep_recluster_outputs),
                ", ".join(path.name for path in deleted),
            )
        else:
            log.info("Nothing to prune under %s (keep=%d).", online_dir, int(args.keep_recluster_outputs))


def main() -> None:
    CONFIG.paths.ensure()
    CONFIG.write_snapshot()
    # One explicit budget prevents the common CPU/RAM oversubscription where
    # several DFT processes each start a full BLAS thread pool.  The same
    # value is recorded in config.json and can be raised deliberately.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(BLAS_THREADS)
    log.info("=== ESM verification pipeline (cache=%s) ===", CACHE_VERSION)
    log.info(
        "Run=%s mode=%s | candidate cache=%s | shared density cubes=%s | "
        "workers=%d BLAS threads=%d void topology=%s",
        CONFIG.paths.run_id, EXECUTION_MODE, CANDS_DIR, CUBES_DIR,
        DFT_WORKERS, BLAS_THREADS, COMPUTE_VOID_TOPOLOGY,
    )

    if EXECUTION_MODE == "online":
        _run_online_append()
        return

    # ------------------------------------------------------------------
    # STEP 1: Extract + cluster alphabet
    # ------------------------------------------------------------------
    log.info("--- [1] Letter extraction (resumable) ---")
    candidates, metadata_registry, control_refs, _ledger_records = process_datasets_resumable(
        DATASETS_CONFIG,
        CP_KWARGS,
        cache_only=CONFIG.batch_validation_cache_only,
    )
    if not candidates:
        log.error("No candidates extracted.")
        return

    type_counts = Counter(c.critical_point.field_type for c in candidates)
    log.info(
        "Candidates: %d from %d molecules | types: %s",
        len(candidates),
        len(metadata_registry),
        dict(type_counts),
    )
    n_nuclear = type_counts.get("nuclear_attractor", 0)
    if n_nuclear < max(10, 0.05 * len(candidates)):
        log.warning(
            "Few nuclear_attractor candidates (%d / %d). "
            "§3.1/§3.2 distributional test will be weak. "
            "The field-first search may need a denser field grid/search; "
            "atom-centred fallback points are intentionally prohibited.",
            n_nuclear,
            len(candidates),
        )

    log.info("--- [1b] Distance threshold + clustering ---")
    distance_thresholds = suggest_distance_threshold(
        candidates,
        group_by_field_type=True,
    )
    if not distance_thresholds:
        log.error("Could not estimate a type-specific distance threshold.")
        return
    # Used only by legacy scalar consumers whose comparison is restricted to
    # bond classes; all assignment/clustering paths retain the type-specific
    # values, as §3.2 requires.
    distance_threshold = float(np.median(list(distance_thresholds.values())))
    bond_distance_threshold = float(
        distance_thresholds.get("bond_critical_point", distance_threshold)
    )
    log.info("type-specific distance thresholds -> %s", distance_thresholds)

    # DP-means (O(n*k), see corpus.py module docstring) in place of the O(n^2)
    # batch method -- see the 2026-07-22 clustering-scalability discussion.
    # `cluster_candidates_bootstrap` is kept importable/callable for small
    # corpora where the exact batch result is preferred, but is no longer
    # the default path here.
    classes = dpmeans_incremental_cluster(
        candidates,
        spawn_threshold=distance_thresholds,
        l_max_bispectrum=L_MAX,
        n_bootstrap=N_BOOTSTRAP_CLUSTER,
        corpus_version=CORPUS_VERSION,
    )
    log.info("Alphabet: %d candidates -> %d classes", len(candidates), len(classes))

    # Was `cluster_labels(candidates, ...)`, a second full O(n^2) re-clustering
    # just to recover per-candidate labels. `assign_candidate_labels` does the
    # same job in O(n*k) by nearest-centroid lookup against the classes just
    # computed -- no reason to pay O(n^2) twice for the same information.
    string_labels = assign_candidate_labels(
        candidates,
        classes,
        distance_thresholds,
        use_gpu=USE_GPU_LABEL_ASSIGNMENT,
    )
    class_id_to_int = {c.class_id: i for i, c in enumerate(classes)}
    labels_arr = np.array(
        [class_id_to_int[lab] if lab is not None else -1 for lab in string_labels]
    )
    mols_per_label: Dict[int, set] = defaultdict(set)
    membership_index: Dict[str, List[int]] = defaultdict(list)
    for cand, lbl in zip(candidates, labels_arr):
        if lbl >= 0:
            mols_per_label[int(lbl)].add(cand.source_molecule_id)
    for candidate_index, class_id in enumerate(string_labels):
        if class_id is not None:
            membership_index[class_id].append(candidate_index)
    n_multi = sum(1 for s in mols_per_label.values() if len(s) > 1)
    log.info(
        "Reuse: %d/%d raw clusters span multiple molecules",
        n_multi,
        len(mols_per_label),
    )

    # ------------------------------------------------------------------
    # STEP 2: Intrinsic dimension §8.2
    # ------------------------------------------------------------------
    log.info("--- [2] Intrinsic dimension §8.2 ---")
    vectors = np.stack([c.descriptor.to_vector() for c in candidates])
    dim_result = levina_bickel_intrinsic_dimension(
        vectors, k_neighbors=10, n_bootstrap=N_BOOTSTRAP_DIM
    )
    log.info("Intrinsic dim ~ %.3f (raw dim %d)", dim_result.score, vectors.shape[1])

    # ------------------------------------------------------------------
    # STEP 3: Fundamentality §12
    # ------------------------------------------------------------------
    log.info("--- [3] Fundamentality §12 ---")
    dictionary_capacity_limit = min(
        len(candidates) - 1,
        _dictionary_capacity_limit(vectors.shape[1]),
    )
    dictionary_selection = select_dictionary_size_by_cv(
        candidates,
        n_nonzero_coefs_grid=(2, 3, 5),
        n_folds=DICTIONARY_CV_FOLDS,
        max_candidates_per_fold=DICTIONARY_CV_MAX_CANDIDATES,
        max_atoms=dictionary_capacity_limit,
        batch_size=DICTIONARY_BATCH_SIZE,
        n_jobs=ANALYSIS_N_JOBS,
        screening_max_candidates_per_fold=DICTIONARY_CV_SCREEN_MAX_CANDIDATES,
        screening_n_folds=DICTIONARY_CV_SCREEN_FOLDS,
        max_finalists=DICTIONARY_CV_FINALISTS,
    )
    n_atoms, n_nonzero_coefs = dictionary_selection["best"]
    screening_edge = dictionary_selection["screening"]["best_at_largest_tested_size"]
    if dictionary_selection["best_at_largest_tested_size"] or screening_edge:
        log.warning(
            "Dictionary CV optimum (%d atoms) is at a resource/grid edge; "
            "capacity remains unresolved, not confirmed.",
            n_atoms,
        )
    log.info(
        "Dictionary CV selected n_atoms=%d, n_nonzero_coefs=%d "
        "(capacity_limit=%d, screen=%d candidates x %d fold(s), final=%d candidates x %d fold(s))",
        n_atoms,
        n_nonzero_coefs,
        dictionary_capacity_limit,
        DICTIONARY_CV_SCREEN_MAX_CANDIDATES,
        dictionary_selection["screening"]["n_folds"],
        DICTIONARY_CV_MAX_CANDIDATES,
        dictionary_selection["n_folds"],
    )
    fund_config = FundamentalityConfig(
        n_atoms=n_atoms,
        n_nonzero_coefs=n_nonzero_coefs,
        n_null_permutations=N_NULL_PERM,
        batch_size=DICTIONARY_BATCH_SIZE,
        n_jobs=ANALYSIS_N_JOBS,
    )
    dict_fit = fit_shared_dictionary(candidates, fund_config)
    dict_diag = diagnose_dictionary_fit(candidates, dict_fit)
    log.info("Dictionary diagnostics: %s", dict_diag)
    fund_results = compute_fundamentality(candidates, dict_fit)
    n_fund = sum(1 for r in fund_results if r.is_fundamental)
    log.info("Fundamental candidates: %d / %d", n_fund, len(fund_results))

    # ------------------------------------------------------------------
    # STEP 4: Unsupervised control §16.2
    # ------------------------------------------------------------------
    log.info("--- [4] Scrambled-field control §16.2 ---")
    n_control = max(8, min(N_CONTROL_MAX, int(round(0.1 * len(control_refs)))))
    n_control = min(n_control, len(control_refs))
    control_molecules = [
        (mol_id, CubeFileField(cube_path, field_kind="density", level_of_theory=CONFIG.level_of_theory, environment="vacuum"), bbox)
        for mol_id, cube_path, bbox in control_refs[:n_control]
        if os.path.exists(cube_path)
    ]
    compression_conf = None
    if len(control_molecules) >= 2:
        compression_conf = validate_structure_against_control(
            control_molecules,
            candidates,
            dict_fit.config,
            control_mode="scrambled",
            n_controls=N_CONTROL_REPS,
            grid_resolution=8,
            shell_radii=SHELL_RADII,
            r0=R0,
            sigma=SIGMA,
            l_max=L_MAX,
            n_points_per_shell=N_POINTS_PER_SHELL,
            cp_kwargs=CP_KWARGS,
        )
        log.info(
            "Structure vs control: score=%.4f interval=%s",
            compression_conf.score,
            compression_conf.interval,
        )
    else:
        log.warning("Not enough control molecules for §16.2")

    # ------------------------------------------------------------------
    # STEP 5: Distributional A/B §3.1–3.2
    # ------------------------------------------------------------------
    log.info("--- [5] Distributional Hypothesis A/B ---")
    dist_config = HypothesisTestConfig(
        n_null_permutations=N_NULL_PERM,
        min_members_for_decision=3,
    )
    attractor_links = link_candidates_to_attractors(candidates, classes, dist_config)
    cooc_by_type: Dict[str, List[CooccurrenceEntry]] = {}
    for ft in ("bond_critical_point", "ring_critical_point"):
        try:
            cooc_by_type[ft] = build_letter_cooccurrence(
                candidates,
                classes,
                ft,
                links=attractor_links,
                config=dist_config,
                corpus_version=CORPUS_VERSION,
            )
        except Exception as e:
            log.debug("cooccurrence %s failed: %s", ft, e)
            cooc_by_type[ft] = []
    hypothesis_results = evaluate_all_word_candidates(
        classes,
        candidates,
        config=dist_config,
        corpus_version=CORPUS_VERSION,
        links=attractor_links,
        cooccurrence_by_type=cooc_by_type,
        membership_index=membership_index,
    )
    hyp_summary = summarize_hypothesis_evidence(hypothesis_results)
    log.info("Hypothesis tally by type: %s", hyp_summary)

    # ------------------------------------------------------------------
    # STEP 6: Commutation §11
    # ------------------------------------------------------------------
    log.info("--- [6] Commutation test §11 ---")
    comm_config = CommutationConfig(
        n_continuous_trials=N_COMM_TRIALS,
        n_categorical_trials=N_COMM_TRIALS,
        n_null_bootstrap=N_NULL_PERM,
    )
    # Prefer stable multi-molecule classes; cap cost
    ranked = sorted(
        classes,
        key=lambda c: c.cluster_stability.score,
        reverse=True,
    )[:MAX_CLASSES_FOR_COMMUTATION]
    axis_annotations = {}
    comm_classes_by_type: Dict[str, List[LetterClass]] = defaultdict(list)
    comm_candidates_by_type: Dict[str, List[LetterCandidate]] = defaultdict(list)
    for cls in classes:
        comm_classes_by_type[cls.discovery_provenance.critical_point_type].append(cls)
    for candidate in candidates:
        comm_candidates_by_type[candidate.critical_point.field_type].append(candidate)
    comm_vectors_by_type: Dict[str, np.ndarray] = {}
    continuous_cache_by_type: Dict[str, Dict[int, ConfidenceBlock]] = defaultdict(dict)
    for cls in ranked:
        try:
            field_type = cls.discovery_provenance.critical_point_type
            if field_type not in comm_vectors_by_type:
                comm_vectors_by_type[field_type] = np.stack(
                    [
                        candidate.descriptor.to_vector()
                        for candidate in comm_candidates_by_type[field_type]
                    ]
                )
            axis_annotations[cls.class_id] = build_axis_annotations(
                cls,
                comm_classes_by_type[field_type],
                config=comm_config,
                vectors=comm_vectors_by_type[field_type],
                continuous_cache=continuous_cache_by_type[field_type],
            )
        except Exception as e:
            log.debug("commutation failed %s: %s", cls.class_id, e)

    # ------------------------------------------------------------------
    # STEP 7: Isostere §15 step 5
    # ------------------------------------------------------------------
    log.info("--- [7] Isostere Campbell–Fiske §15 step 5 ---")
    calib_report, calib_items, quadrant_counts = build_isostere_design(
        candidates,
        metadata_registry,
        PRIMARY_PROPERTY,
        bond_distance_threshold,
        max_per_quadrant=MAX_PAIRS_PER_QUADRANT,
    )
    if calib_report is not None:
        log.info(
            "Isostere done; quadrants=%s overall_pass=%s",
            quadrant_counts,
            getattr(calib_report, "overall_pass", None),
        )
    else:
        log.warning("Isostere incomplete; quadrants=%s", quadrant_counts)

    # ------------------------------------------------------------------
    # STEP 8: Semantics / orphans §16.1–16.3
    # ------------------------------------------------------------------
    log.info("--- [8] Per-class semantics §16.1/16.3 ---")
    prop_names = (PRIMARY_PROPERTY,) + tuple(
        p for p in SECONDARY_PROPERTIES if p != PRIMARY_PROPERTY
    )
    semantic_stats = update_class_semantics(
        classes,
        candidates,
        metadata_registry,
        prop_names,
        distance_threshold,
        CORPUS_VERSION,
        membership_index=membership_index,
        max_molecules_per_class=SEMANTIC_MAX_MOLECULES_PER_CLASS,
    )
    log.info("Semantics: %s", semantic_stats)

    # ------------------------------------------------------------------
    # STEP 9: Paradigms / synonyms §4.4–4.6
    # ------------------------------------------------------------------
    log.info("--- [9] Paradigms vs synonymy ---")
    close_pairs, func_distances = build_close_class_pairs(
        classes, candidates, metadata_registry, PRIMARY_PROPERTY, distance_threshold
    )
    para_config = ParadigmConfig(min_group_size=2, n_bootstrap=N_NULL_PERM)
    paradigms = detect_declension_paradigms(
        classes, candidate_pairs=close_pairs or None, config=para_config
    )
    synonyms = build_synonym_relations(
        close_pairs, paradigms, para_config, function_distances=func_distances
    )
    log.info("Paradigms=%d synonym_relations=%d close_pairs=%d", len(paradigms), len(synonyms), len(close_pairs))

    # ------------------------------------------------------------------
    # STEP 10: Positional sensitivity §17
    # ------------------------------------------------------------------
    log.info("--- [10] Positional sensitivity §17 ---")
    trajectories: List[BuildTrajectory] = []
    if ENABLE_STATIC_BUILD_PROXY:
        # Useful only for exploratory software diagnostics.  These are
        # graph-similarity chains, not ordered field frames of an executed
        # reaction, so their output cannot establish a grammar or merge
        # context-dependent letter realisations.
        log.warning(
            "ESM_ENABLE_STATIC_BUILD_PROXY=1: using non-dynamic graph-similarity "
            "chains as a clearly labelled §17 proxy"
        )
        trajectories = build_build_trajectories(
            candidates, classes, metadata_registry, PRIMARY_PROPERTY, distance_threshold
        )
    else:
        log.info(
            "Skipped static build proxy: §8–10/§17 require physically tracked, "
            "ordered field frames; static QM9 snapshots cannot establish grammar"
        )
    pos_profile = None
    if trajectories:
        try:
            pos_profile = positional_sensitivity_profile(
                trajectories, config=PositionalSensitivityConfig(n_null_permutations=N_NULL_PERM)
            )
            log.info("Positional profile on %d trajectories", len(trajectories))
        except Exception as e:
            log.warning("Positional sensitivity failed: %s", e)
    else:
        log.info("No physical trajectories supplied; positional/dynamic conclusion remains inconclusive")

    # ------------------------------------------------------------------
    # STEP 11: Alphabet growth snapshots + §19 profiles
    # ------------------------------------------------------------------
    log.info("--- [11] §19 falsification profiles ---")
    # Occurrence / entropy for letter profile
    class_occurrence_counts = [
        len(membership_index.get(cls.class_id, ()))
        for cls in classes
    ]
    within_class_entropies = []
    for cls in classes:
        within_class_entropies.append(
            _within_class_entropy_bounded(
                candidates,
                membership_index.get(cls.class_id, ()),
                cls.invariant_centroid,
            )
        )
    total_occ = sum(class_occurrence_counts) or 1
    class_probs = np.array(class_occurrence_counts, dtype=float) / total_occ
    class_probs = class_probs[class_probs > 0]
    between_class_entropy = (
        float(-np.sum(class_probs * np.log2(class_probs))) if len(class_probs) else 0.0
    )

    # Growth snapshots must respect an independent held-out set (§19.7): a
    # prefix at 100% that already contains the alleged held-out molecules
    # makes rediscovery tautological.  Use a seeded random *molecule* order,
    # then reserve its final 10% before any snapshot is built.
    all_mol_ids = sorted(metadata_registry.keys())
    profile_rng = np.random.default_rng(0)
    mol_order = [
        all_mol_ids[i] for i in profile_rng.permutation(len(all_mol_ids))
    ]
    if len(mol_order) >= 20:
        n_held_out = max(1, int(round(0.1 * len(mol_order))))
        training_mol_order = mol_order[:-n_held_out]
        held_out_mols = set(mol_order[-n_held_out:])
    else:
        training_mol_order = mol_order
        held_out_mols = set()

    cand_by_mol: Dict[str, List[LetterCandidate]] = defaultdict(list)
    for candidate in candidates:
        cand_by_mol[candidate.source_molecule_id].append(candidate)
    training_batches = [cand_by_mol[mol_id] for mol_id in training_mol_order]
    fractions = [0.25, 0.5, 0.75, 1.0]
    checkpoints = sorted({
        max(1, min(len(training_batches), int(round(fraction * len(training_batches)))))
        for fraction in fractions
    }) if training_batches else []

    snapshot_classes: List[List[LetterClass]] = []
    snapshot_sizes: List[int] = []
    snapshot_pools: List[List[LetterCandidate]] = []
    corpus_size_snapshots: List[int] = []
    n_classes_snapshots: List[int] = []
    if checkpoints:
        # One DP-means stream, rather than re-clustering every growing prefix.
        snapshot_classes = dpmeans_incremental_snapshots(
            training_batches,
            checkpoints,
            spawn_threshold=distance_thresholds,
            l_max_bispectrum=L_MAX,
            corpus_version=CORPUS_VERSION,
            class_id_prefix="growth",
        )
        snapshot_pools = _reservoir_snapshot_pools(
            training_batches,
            checkpoints,
            max_candidates=PROFILE_COMPRESSION_SAMPLE,
        )
        prefix_candidate_count = 0
        checkpoint_iter = iter(checkpoints)
        next_checkpoint = next(checkpoint_iter, None)
        for molecule_count, batch in enumerate(training_batches, start=1):
            prefix_candidate_count += len(batch)
            if molecule_count == next_checkpoint:
                snapshot_sizes.append(molecule_count)
                corpus_size_snapshots.append(prefix_candidate_count)
                next_checkpoint = next(checkpoint_iter, None)
        n_classes_snapshots = [len(snapshot) for snapshot in snapshot_classes]

    held_out_cands = [
        candidate for candidate in candidates
        if candidate.source_molecule_id in held_out_mols
    ]

    letter_prof = build_letter_corpus_profile(
        # §19.2's alphabet curve is against growing *molecule* corpus size;
        # raw CP counts belong separately to §19.7's candidate-count curve.
        corpus_size_snapshots=snapshot_sizes,
        n_classes_snapshots=n_classes_snapshots,
        letter_classes=classes,
        class_occurrence_counts=class_occurrence_counts,
        intrinsic_dimension=dim_result,
        raw_dimension=vectors.shape[1],
        within_class_entropies=within_class_entropies,
        between_class_entropy=between_class_entropy,
        corpus_version=CORPUS_VERSION,
    )

    alphabet_disc = build_alphabet_discovery_profile(
        snapshot_classes=snapshot_classes,
        snapshot_sizes=snapshot_sizes,
        distance_threshold=distance_thresholds,
        snapshot_candidate_counts=corpus_size_snapshots,
        held_out_candidates=held_out_cands or None,
        fundamentality_config=fund_config,
        snapshot_candidate_pools=snapshot_pools or None,
        corpus_version=CORPUS_VERSION,
    )

    # Word profile from distributional links (proxy for word_class layer)
    word_lengths = []
    occupancies = []
    for r in hypothesis_results:
        if r.dominant_attractor_letters is not None:
            word_lengths.append(len(r.dominant_attractor_letters))
        if r.tuple_concentration is not None:
            occupancies.append(r.tuple_concentration.score)
    word_prof = build_word_corpus_profile(
        word_lengths=word_lengths,
        occupancy_frequencies=occupancies,
        topology_variances=[],
        invariant_distances_for_topology=[],
        corpus_version=CORPUS_VERSION,
    )

    # Grammar: co-occurrence + forbidden proxy
    all_cooc = [e for lst in cooc_by_type.values() for e in lst]
    cand_labels = assign_candidate_labels(
        candidates,
        classes,
        distance_thresholds,
        use_gpu=USE_GPU_LABEL_ASSIGNMENT,
    )
    letter_ids = [c.class_id for c in classes if c.discovery_provenance.critical_point_type == "nuclear_attractor"]
    if not letter_ids:
        letter_ids = [c.class_id for c in classes]
    forbidden = build_forbidden_proxy(candidates, cand_labels, letter_ids)
    n_combo = max(1, len(letter_ids) * (len(letter_ids) - 1) // 2)
    grammar_prof = build_grammar_profile(
        cooccurrence_entries=all_cooc,
        forbidden_entries=forbidden,
        n_combinatorially_valid_pairs=n_combo,
        transition_model=None,  # needs MSM / trajectories steps 7–9
        corpus_version=CORPUS_VERSION,
    )

    meaning_prof = build_word_meaning_profile(
        word_classes_semantic_status=[c.semantic_status for c in classes],
        descriptor_matrix=None,  # multi-axis functional descriptors not assembled
        paradigms=paradigms,
        paradigm_member_descriptor_shifts=None,
        corpus_version=CORPUS_VERSION,
    )

    sentence_prof = build_sentence_meaning_profile(
        additive_prediction_errors=[],
        full_sentence_prediction_errors=[],
        success_by_grammaticality_status={},
        corpus_version=CORPUS_VERSION,
    )

    synonym_prof = build_synonym_profile(
        synonym_relations=synonyms,
        realization_counts_per_member=None,
        typical_within_class_realization_count=None,
        corpus_version=CORPUS_VERSION,
    )

    # Polysemy: classes with high within-class entropy vs multi-property conflict
    # Use environment_response only if multi-env; here all vacuum -> inconclusive
    polysemy_prof = build_polysemy_profile(
        environment_response_profiles=[],
        corpus_version=CORPUS_VERSION,
    )

    falsification = {
        "ExpectedAlphabetDiscoveryProfile_S19_7": alphabet_disc,
        "ExpectedLetterCorpusProfile_S19_2": letter_prof,
        "ExpectedWordCorpusProfile_S19_3": word_prof,
        "ExpectedGrammarProfile_S19_4": grammar_prof,
        "ExpectedWordMeaningProfile_S19_5": meaning_prof,
        "ExpectedSentenceMeaningProfile_S19_6": sentence_prof,
        "ExpectedSynonymProfile_S19_8": synonym_prof,
        "ExpectedPolysemyProfile_S19_9": polysemy_prof,
    }

    # ------------------------------------------------------------------
    # STEP 12: Assemble compact report
    # ------------------------------------------------------------------
    log.info("--- [12] Write reports ---")

    # These records are shared by the report and visualisation manifest, so
    # the visible class ID and semantic/reuse evidence are exactly those from
    # this analysis run -- never a re-clustering done only for display.
    class_records = {
        cls.class_id: slim_class(cls, candidates, membership_index)
        for cls in classes
    }
    letter_archive_manifest = None
    if WRITE_LETTER_ARCHIVE:
        try:
            letter_archive_manifest = write_letter_archive(
                candidates=candidates,
                classes=classes,
                labels=string_labels,
                metadata_registry=metadata_registry,
                cube_dir=CUBES_DIR,
                output_dir=LETTER_ARCHIVE_DIR,
                class_records=class_records,
                analysis_context={
                    "cache_version": CACHE_VERSION,
                    "corpus_version": CORPUS_VERSION,
                    "candidate_discovery_method": CANDIDATE_DISCOVERY_METHOD,
                    "atomic_coordinates_used_to_discover_candidates": False,
                    "distance_threshold_by_critical_point_type": distance_thresholds,
                    "level_of_theory": "DFT/pbe/def2-svp",
                    "environment": "vacuum",
                },
                descriptor_shard_rows=LETTER_ARCHIVE_SHARD_ROWS,
            )
            log.info("Letter analysis archive: %s", letter_archive_manifest)
        except Exception as exc:
            # Persisting an extra artefact must not turn a completed theory
            # run into a false computational failure, but the report records
            # whether the reusable archive was actually produced.
            log.exception("Could not write letter analysis archive: %s", exc)
    visualization_manifest = None
    if WRITE_VISUALIZATION_BUNDLE:
        try:
            scene_molecule_ids = None
            if len(metadata_registry) > VISUALIZATION_ALL_MOLECULE_LIMIT:
                selected = set()
                for record in class_records.values():
                    for molecule_id in record.get("example_molecule_ids", [])[:VISUALIZATION_REPRESENTATIVES_PER_CLASS]:
                        selected.add(molecule_id)
                        if len(selected) >= VISUALIZATION_REPRESENTATIVE_MOLECULE_LIMIT:
                            break
                    if len(selected) >= VISUALIZATION_REPRESENTATIVE_MOLECULE_LIMIT:
                        break
                scene_molecule_ids = sorted(selected)
                log.info(
                    "3-D scenes capped at %d representative molecules; full occurrence data remain in %s",
                    len(scene_molecule_ids),
                    letter_archive_manifest,
                )
            visualization_manifest = write_visualization_bundle(
                candidates=candidates,
                classes=classes,
                labels=string_labels,
                metadata_registry=metadata_registry,
                cube_dir=CUBES_DIR,
                output_dir=VISUALIZATION_DIR,
                class_records=class_records,
                include_molecule_ids=scene_molecule_ids,
                analysis_context={
                    "cache_version": CACHE_VERSION,
                    "candidate_discovery_method": CANDIDATE_DISCOVERY_METHOD,
                    "atomic_coordinates_used_to_discover_candidates": False,
                },
            )
            log.info(
                "3-D letter scenes: %s (local fields are extracted on demand)",
                visualization_manifest,
            )
        except Exception as exc:
            # A missing cube from a partial/resumed run must not invalidate
            # the scientific analysis.  The report preserves the failure so
            # it cannot be mistaken for an exported visualisation.
            log.exception("Could not write 3-D letter scenes: %s", exc)

    structure_beats_control = None
    if compression_conf is not None and compression_conf.interval is not None:
        structure_beats_control = compression_conf.score > compression_conf.interval[1]

    summary = {
        "metadata": {
            "corpus_version": CORPUS_VERSION,
            "cache_version": CACHE_VERSION,
            "n_molecules": len(metadata_registry),
            "n_candidates": len(candidates),
            "n_field_pattern_candidates": sum(
                int(meta.get("field_pattern_candidate_count", 0))
                for meta in metadata_registry.values()
            ),
            "n_field_pattern_part_whole_relations": sum(
                int(meta.get("field_pattern_part_whole_relation_count", 0))
                for meta in metadata_registry.values()
            ),
            "n_field_pattern_repeated_part_groups": sum(
                int(meta.get("field_pattern_repeated_part_group_count", 0))
                for meta in metadata_registry.values()
            ),
            "n_field_pattern_bounded_void_candidates": sum(
                int(meta.get("field_pattern_bounded_void_count", 0))
                for meta in metadata_registry.values()
            ),
            "n_classes": len(classes),
            "distance_threshold_by_critical_point_type": distance_thresholds,
            "legacy_median_distance_threshold": distance_threshold,
            "candidate_type_counts": dict(type_counts),
            "field_region_candidate_search": {
                "enabled": bool(CONFIG.field_patterns_enabled),
                "method": FIELD_PATTERN_PARAMETERS["method"],
                "archive_dir": FIELD_PATTERN_DIR,
                "cache_signature": FIELD_PATTERN_CACHE_SIGNATURE,
                "uses_atoms_or_critical_points_as_seeds": False,
                "part_whole_and_repeated_subpattern_evidence": "per-molecule provisional evidence; not a word-class verdict",
                "relationship_to_current_alphabet": "field-only proposals are archived separately; the current named-class dictionary and downstream validation still use legacy critical-point candidates",
            },
            "field_first_methodology": {
                "candidate_discovery_method": CANDIDATE_DISCOVERY_METHOD,
                "atomic_coordinates_used_to_discover_candidates": False,
                "atomic_coordinates_do_not_determine": [
                    "candidate existence",
                    "critical-point type",
                    "local invariant descriptor",
                    "descriptor clustering / class ID",
                ],
                "atomic_information_roles": [
                    "nuclear geometry used upstream to calculate the molecular field",
                    "independent overlay/reference for post-hoc spatial comparison",
                    "explicit graph baseline only in the declared graph-vs-field isostere control",
                ],
                "comparison_rule": (
                    "Atom coincidence is reported as a continuous nearest-reference distance; "
                    "it is not used as a label or acceptance criterion."
                ),
            },
            "datasets": DATASETS_CONFIG,
            "level_of_theory": "DFT/pbe/def2-svp",
            "environment": "vacuum",
            "resource_controls": {
                "analysis_n_jobs": ANALYSIS_N_JOBS,
                "dictionary_batch_size": DICTIONARY_BATCH_SIZE,
                "dictionary_scratch_mb": DICTIONARY_SCRATCH_MB,
                "dictionary_capacity_limit": dictionary_capacity_limit,
                "dictionary_cv_max_candidates_per_fold": DICTIONARY_CV_MAX_CANDIDATES,
                "dictionary_cv_screen_max_candidates_per_fold": DICTIONARY_CV_SCREEN_MAX_CANDIDATES,
                "dictionary_cv_screen_folds": DICTIONARY_CV_SCREEN_FOLDS,
                "dictionary_cv_finalists": DICTIONARY_CV_FINALISTS,
                "semantic_max_molecules_per_class": SEMANTIC_MAX_MOLECULES_PER_CLASS,
                "gpu_label_assignment_requested": USE_GPU_LABEL_ASSIGNMENT,
                "dft_backend_requested": DFT_BACKEND,
                "dft_workers_requested": DFT_WORKERS,
                "field_grid_maximum_seed_limit": FIELD_GRID_MAXIMUM_SEED_LIMIT,
                "static_build_proxy_enabled": ENABLE_STATIC_BUILD_PROXY,
                "visualization_bundle_requested": WRITE_VISUALIZATION_BUNDLE,
                "letter_archive_requested": WRITE_LETTER_ARCHIVE,
                "letter_archive_shard_rows": LETTER_ARCHIVE_SHARD_ROWS,
                "visualization_all_molecule_limit": VISUALIZATION_ALL_MOLECULE_LIMIT,
                "visualization_representatives_per_class": VISUALIZATION_REPRESENTATIVES_PER_CLASS,
                "visualization_representative_molecule_limit": VISUALIZATION_REPRESENTATIVE_MOLECULE_LIMIT,
            },
            "letter_archive_manifest": str(letter_archive_manifest) if letter_archive_manifest else None,
            "visualization_manifest": str(visualization_manifest) if visualization_manifest else None,
            "dynamics_and_context_status": {
                "observation_regime": "static_equilibrium_density_snapshots",
                "transition_model_inferred": False,
                "grammar_inferred": False,
                "cross_context_identity_inferred": False,
                "identity_rule": (
                    "Different static classes may be contextual or execution-stage "
                    "realisations of one unit; they are not merged without tracked "
                    "multi-frame and multi-context evidence."
                ),
                "required_evidence": [
                    "ordered reaction or relaxation frames with physical time/path coordinate",
                    "tracked correspondence of critical-point/basin realisations between frames",
                    "controlled environment, charge, or external-field perturbations",
                    "held-out transition and response statistics",
                ],
            },
        },
        "theory_gate_checks": {
            "S3_2_type_partition_enforced": True,
            "S3_1_hypothesis_tally": hyp_summary,
            "S8_2_intrinsic_dimension": sanitize_for_json(dim_result),
            "S12_fundamentality": slim_fund_results(fund_results),
            "S12_dictionary_selection_cv": dictionary_selection,
            "S12_dictionary_diagnostics": dict_diag,
            "S15_2_structure_beats_scrambled_control": structure_beats_control,
            "S15_2_control_block": sanitize_for_json(compression_conf),
            "S14_4_isostere_quadrants": quadrant_counts,
            "S14_4_isostere_overall_pass": (
                getattr(calib_report, "overall_pass", None) if calib_report is not None else None
            ),
            "S15_3_semantics": semantic_stats,
            "S4_4_n_paradigms": len(paradigms),
            "S4_4_n_synonym_relations": len(synonyms),
            "S17_n_build_trajectories": len(trajectories),
            "multi_molecule_cluster_fraction": n_multi / max(1, len(mols_per_label)),
        },
        "falsification_profile_tallies_S19": {
            name: _verdict_tally(sanitize_for_json(prof) if not isinstance(prof, dict) else sanitize_for_json(prof))
            for name, prof in falsification.items()
            if isinstance(prof, dict) or dataclasses.is_dataclass(prof)
        },
        "honest_scope_limits": [
            "This runner still materialises LetterCandidate objects for the complete corpus. Its bounded clustering/CV/semantic paths remove several avoidable O(N^2) and duplicate-prefix costs, but a ten-million-molecule production run additionally requires a sharded on-disk candidate store and streaming reductions.",
            "Exact PBE/def2-SVP cube generation with PySCF is not a credible ten-million-molecule field source. Per the theory's computational-barrier premise, that scale requires a separately validated ML field/density provider, benchmarked against the DFT reference subset before its results can enter the same falsification report.",
            "Phases 6–8 (MSM trajectories, true forbidden map from failed reactions) not fully available on static QM9 snapshots; grammar.transition_model and sentence profile remain inconclusive by design.",
            "Environment response / true polysemy (§4.3, §19.9) requires multi-solvent or pocket recalculations; all fields are vacuum.",
            "Isostere graph similarity uses Morgan fingerprints (atom-graph), not field topology — consistent with §15 step 5's graph vs function design, not a field-only test.",
            "A static descriptor class is not an ontological identity across execution stages. Contextual declension, polysemy and state transitions require physically tracked multi-frame fields; the runner deliberately does not merge static classes on that hypothesis alone.",
            "Candidate discovery is blind to atom/nucleus coordinates. A sparse grid can miss sharp field features; coverage must be improved with field-resolution/search convergence studies, never by inserting atom-labelled fallback candidates.",
        ],
    }

    full_report = {
        **summary,
        "alphabet_slim": {
            "n_candidates": len(candidates),
            "n_classes": len(classes),
            "classes": list(class_records.values()),
        },
        "phase_analyses": {
            "dimensionality_S8_2": sanitize_for_json(dim_result),
            "fundamentality_S12_summary": slim_fund_results(fund_results),
            "dictionary_selection_cv_S12": dictionary_selection,
            "dictionary_diagnostics_S12": dict_diag,
            "unsupervised_validation_S15_2": sanitize_for_json(compression_conf),
            "distributional_test_S3_1_summary": {
                "tally": hyp_summary,
                "n_word_classes_tested": len(hypothesis_results),
                "n_determinate": sum(
                    1
                    for r in hypothesis_results
                    if r.hypothesis_status.decision != "indeterminate"
                ),
                # keep compact per-class decisions only
                "per_class_decisions": [
                    {
                        "class_id": r.class_id,
                        "type": r.critical_point_type,
                        "decision": r.hypothesis_status.decision,
                        "n_linked": r.n_members_linked,
                        "n_total": r.n_members_total,
                    }
                    for r in hypothesis_results
                ],
            },
            "commutation_S11": {
                "n_classes_tested": len(axis_annotations),
                "annotations": sanitize_for_json(axis_annotations),
            },
            "isostere_S14_4": sanitize_for_json(calib_report),
            "paradigms_S4_4": {
                "paradigms": sanitize_for_json(paradigms),
                "synonyms": sanitize_for_json(synonyms),
                "n_close_pairs": len(close_pairs),
            },
            "positional_sensitivity_S17": sanitize_for_json(pos_profile),
        },
        "falsification_profiles_S19": sanitize_for_json(falsification),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(full_report), f, indent=2, ensure_ascii=False)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(REPORT_PATH) / 1e6
    log.info("Done. Report: %s (%.1f MB)  Summary: %s", REPORT_PATH, size_mb, SUMMARY_PATH)

    # Console theory snapshot
    log.info("=== THEORY SNAPSHOT ===")
    log.info("Hypothesis A/B: %s", hyp_summary)
    log.info("Fundamental fraction: %.3f", n_fund / max(1, len(fund_results)))
    log.info("Structure > scrambled: %s", structure_beats_control)
    log.info("Semantics: %s", semantic_stats)
    for name, tally in summary["falsification_profile_tallies_S19"].items():
        log.info("  %s: %s", name, tally)


if __name__ == "__main__":
    _cli_args = _build_arg_parser().parse_args()
    if _cli_args.recluster:
        _run_recluster(_cli_args)
    elif _cli_args.prune_recluster_outputs:
        _run_prune_only(_cli_args)
    else:
        main()
