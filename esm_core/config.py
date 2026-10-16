"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
Single, serialisable configuration for an ESM analysis run.

The source tree is deliberately not an output directory.  ``AnalysisConfig``
maps every generated artifact either to an immutable/shared field cache or to
one named analysis run.  A run can be resumed safely as long as its signature
is unchanged; a scientifically new analysis gets a new ``run_id`` rather
than silently mixing checkpoints and visual scenes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "auto"}


def _integer(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


@dataclasses.dataclass(frozen=True)
class ArtifactPaths:
    """Layout of generated data, rooted outside the code modules."""

    root: Path = Path("esm_analysis")
    run_id: str = "default"
    field_cache_namespace: str = "density_pbe_def2svp_res020"

    @property
    def shared_dir(self) -> Path:
        """Reusable field inputs.  Do not delete merely to restart analysis."""
        return self.root / "shared_fields" / self.field_cache_namespace

    @property
    def run_dir(self) -> Path:
        """Replaceable/checkpointed result namespace for one analysis."""
        return self.root / "runs" / self.run_id

    @property
    def cubes_dir(self) -> Path:
        return self.shared_dir / "cubes"

    @property
    def candidates_dir(self) -> Path:
        return self.run_dir / "candidate_cache"

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / "ledger" / "corpus.sqlite3"

    @property
    def online_dir(self) -> Path:
        return self.run_dir / "online"

    @property
    def online_archive_dir(self) -> Path:
        """Raw append-only descriptor archive (occurrences.jsonl,
        descriptor_batches/, manifest.json) -- what `recluster_from_archive`
        reads, kept apart from `online_state_path`'s checkpoint so a
        `boundary_audit`/recluster pass never confuses the two. Must stay
        equal to `esm_master_analytics.py`'s own `ONLINE_ARCHIVE_DIR`
        constant (that module derives it the same way, from `online_dir`,
        rather than importing this property -- the two having drifted
        apart is exactly the shape of bug that produced the AttributeError
        this property fixes: `esm_control_center.py` expected this
        attribute to exist here, and it did not)."""
        return self.online_dir / "archive"

    @property
    def online_state_path(self) -> Path:
        return self.online_dir / "online_alphabet_state.json"

    @property
    def online_report_path(self) -> Path:
        return self.run_dir / "reports" / "online_alphabet_report.json"

    @property
    def visualization_dir(self) -> Path:
        return self.run_dir / "visualization"

    @property
    def letter_archive_dir(self) -> Path:
        return self.run_dir / "letter_archive"

    @property
    def void_archive_dir(self) -> Path:
        """Per-molecule void-topology snapshots (`void_topology.py`),
        written from the same extraction worker that finds critical
        points, on the same field, in the same pass -- not a separate,
        optionally-run method. Its own directory rather than a subfolder
        of `letter_archive_dir`: a void region is not a `CriticalPoint`
        (doc §2.3/§20.2 -- no position, no Hessian), so giving it a
        distinct store keeps that distinction real on disk, not just in
        the type system. Nested under `run_dir` like every other mutable
        artefact, so `reset_run()` already covers it without change."""
        return self.run_dir / "void_archive"

    @property
    def field_pattern_dir(self) -> Path:
        """Appendable, field-native region proposals and their part/whole links.

        This archive is independent of the legacy critical-point candidate
        cache: changing the region-proposal parameters rebuilds only these
        compact descriptors from the already-preserved density cubes.
        """
        return self.run_dir / "field_pattern_candidates"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "reports" / "esm_analytics_report.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "reports" / "esm_analytics_summary.json"

    @property
    def preliminary_report_path(self) -> Path:
        """Read-only audit of the existing caches/checkpoint for this run."""
        return self.run_dir / "reports" / "preliminary_field_audit.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def launch_path(self) -> Path:
        return self.run_dir / "runtime" / "launch.json"

    @property
    def extraction_progress_path(self) -> Path:
        """Small, atomically updated heartbeat for a live field-extraction run."""
        return self.run_dir / "runtime" / "extraction_progress.json"

    def ensure(self) -> None:
        """Create directories; no source-directory outputs are created."""
        for path in (
            self.cubes_dir,
            self.candidates_dir,
            self.ledger_path.parent,
            self.online_dir,
            self.online_archive_dir,
            self.online_report_path.parent,
            self.visualization_dir,
            self.letter_archive_dir,
            self.void_archive_dir,
            self.field_pattern_dir,
            self.launch_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def reset_run(self, *, archive: bool = True) -> Optional[Path]:
        """Clear this run's mutable state (ledger, online alphabet state,
        candidate cache, reports, visualization, letter archive) to start a
        genuinely new analysis, WITHOUT touching `shared_dir`/`cubes_dir` --
        those hold the expensive, reusable field inputs `shared_dir`'s own
        docstring already says not to delete just to restart.

        This is safe by construction, not by convention: every mutable
        artefact this class defines (`candidates_dir`, `ledger_path`,
        `online_dir`, `online_archive_dir`, `visualization_dir`,
        `letter_archive_dir`, `void_archive_dir`, `field_pattern_dir`,
        `report_path`, `summary_path`,
        `preliminary_report_path`, `config_snapshot_path`, `launch_path`,
        `extraction_progress_path`)
        is nested under `run_dir`, and `run_dir` is nested under
        `root / "runs"`, entirely disjoint from `shared_dir`
        (`root / "shared_fields" / ...`) -- so deleting `run_dir` wholesale
        can never reach a cube, regardless of what a caller has done to
        `run_id`. The two assertions below check that disjointness directly
        rather than trusting it silently, since this method deletes.

        With `archive=True` (the default), `run_dir` is renamed aside with
        a timestamp instead of removed, so an accidental reset is still
        recoverable; pass `archive=False` for a permanent delete. Returns
        the archived path, or None if there was nothing to reset or the
        delete was permanent.
        """
        run_dir = self.run_dir.resolve()
        shared_dir = self.shared_dir.resolve()
        root = self.root.resolve()
        disjoint = (
            run_dir != shared_dir
            and shared_dir not in (run_dir, *run_dir.parents)
            and run_dir not in (shared_dir, *shared_dir.parents)
        )
        if not disjoint:
            raise RuntimeError(
                f"refusing to reset: run_dir ({run_dir}) and shared_dir ({shared_dir}) "
                f"are not disjoint -- check `root`/`run_id`/`field_cache_namespace`"
            )
        if root not in run_dir.parents:
            raise RuntimeError(
                f"refusing to reset: run_dir ({run_dir}) is not nested under root ({root})"
            )
        if not run_dir.exists():
            return None
        if archive:
            destination = run_dir.parent / f"{run_dir.name}.reset_{int(time.time())}"
            run_dir.rename(destination)
            return destination
        shutil.rmtree(run_dir)
        return None


def analysis_signature_id(payload: Mapping[str, Any]) -> str:
    """Stable content hash of an :meth:`AnalysisConfig.analysis_signature_payload`."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """All runtime-tunable ESM parameters used by the runner and UI.

    The descriptor ``regime`` is an explicitly testable representation choice
    -- it is not a discovered grammar.  Use distinct run IDs when comparing
    regimes so their caches and online states cannot be accidentally mixed.
    """

    paths: ArtifactPaths = dataclasses.field(default_factory=ArtifactPaths)
    cache_version: str = "v6_field_first_multiview_l2"
    candidate_discovery_method: str = "blind_scalar_field_grid_extremum_newton_search"
    level_of_theory: str = "DFT/pbe/def2-svp"
    descriptor_regime: str = "balanced_combined"
    execution_mode: str = "online"
    resolution: float = 0.20
    shell_radii: Tuple[float, ...] = (1.0, 1.8, 2.6)
    r0: float = 2.0
    sigma: float = 0.4
    l_max: int = 2
    n_points_per_shell: int = 60
    field_grid_maximum_seed_limit: int = 4096
    cp_bond_search_radius: float = 3.0
    cp_ring_search: bool = True
    cp_n_seeds_per_axis: int = 8
    cp_max_step: float = 0.35
    cp_grad_tol: float = 1e-4
    cp_dedupe_radius: float = 0.08
    # Run in the SAME extraction worker as critical-point search, on the
    # same field -- not a separately-invoked script (see
    # `_extract_record_worker`). A toggle rather than a deletion because it
    # is still a real per-molecule cost (a GUDHI cubical complex per
    # threshold): off is for a corpus-scale run that has already decided
    # it does not need this layer, not the default path to skipping it.
    compute_void_topology: bool = True
    void_topology_quantile_thresholds: Tuple[float, ...] = (0.02, 0.05, 0.10)
    # A separate field-region proposal family. These parameters never use
    # atoms or critical points and do not alter the legacy descriptor cache.
    field_patterns_enabled: bool = True
    field_pattern_quantiles: Tuple[float, ...] = (
        0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.995,
    )
    field_pattern_min_voxels: int = 8
    field_pattern_max_regions_per_level: int = 512
    field_pattern_max_grid_fraction: float = 0.98
    field_pattern_include_boundary_components: bool = False
    field_pattern_containment_min_fraction: float = 0.80
    field_pattern_repeated_part_distance: float = 0.25
    field_pattern_min_part_scale_ratio: float = 1.5
    field_pattern_descriptor_sample_voxels: int = 4096
    field_pattern_containment_sample_voxels: int = 8192
    n_bootstrap_cluster: int = 40
    n_bootstrap_dim: int = 40
    n_null_perm: int = 40
    n_comm_trials: int = 20
    n_control_max: int = 40
    n_control_reps: int = 3
    max_pairs_per_quadrant: int = 25
    max_classes_for_commutation: int = 80
    corpus_version: str = "v_multi_1.0"
    primary_property: str = "electronic_gap"
    secondary_properties: Tuple[str, ...] = (
        "homo_energy", "lumo_energy", "dipole_moment", "polarizability"
    )
    enable_static_build_proxy: bool = False
    analysis_n_jobs: int = 1
    dft_workers: int = 1
    blas_threads: int = 1
    dft_backend: str = "auto"
    use_gpu_label_assignment: bool = True
    dictionary_batch_size: int = 1024
    dictionary_scratch_mb: int = 512
    dictionary_cv_folds: int = 3
    dictionary_cv_max_candidates: int = 5000
    dictionary_cv_screen_max_candidates: int = 2000
    dictionary_cv_screen_folds: int = 1
    dictionary_cv_finalists: int = 3
    semantic_max_molecules_per_class: int = 2000
    profile_compression_sample: int = 5000
    visualization_enabled: bool = True
    letter_archive_enabled: bool = True
    letter_archive_shard_rows: int = 250000
    visualization_all_molecule_limit: int = 10000
    visualization_representatives_per_class: int = 2
    visualization_representative_molecule_limit: int = 20000
    online_merge_min_similarity: float = 0.80
    online_merge_enabled: bool = True
    online_merge_screen_max_prototypes: int = 10000
    # This is a durable online transaction boundary, not merely a worker
    # batch.  Keeping it modest makes newly finished fields visible in the
    # ledger/checkpoint soon after a restart instead of waiting for hundreds
    # of unrelated SCF calculations to finish.
    online_append_chunk_molecules: int = 32
    # Batch validation deliberately defaults to existing candidate caches.  It
    # therefore cannot silently start a new DFT campaign merely because a
    # selected dataset range extends beyond the accumulated corpus.
    batch_validation_cache_only: bool = True
    preliminary_max_cache_files: int = 1000
    preliminary_max_candidates: int = 100000
    default_qm9_samples: int = 600
    adopt_legacy_cache: bool = True

    def __post_init__(self) -> None:
        if self.descriptor_regime not in {
            "local_field", "relational_topology", "balanced_combined"
        }:
            raise ValueError("descriptor_regime is not supported")
        if self.execution_mode not in {"online", "batch_validation"}:
            raise ValueError("execution_mode must be online or batch_validation")
        if len(self.shell_radii) == 0 or any(radius <= 0 for radius in self.shell_radii):
            raise ValueError("shell_radii must be non-empty and positive")
        if not 0.0 <= self.online_merge_min_similarity <= 1.0:
            raise ValueError("online_merge_min_similarity must be in [0, 1]")
        if not self.field_pattern_quantiles or any(
            not 0.0 < q < 1.0 for q in self.field_pattern_quantiles
        ):
            raise ValueError("field_pattern_quantiles must lie strictly inside (0,1)")
        if any(value < 1 for value in (
            self.field_pattern_min_voxels,
            self.field_pattern_max_regions_per_level,
            self.field_pattern_descriptor_sample_voxels,
            self.field_pattern_containment_sample_voxels,
        )):
            raise ValueError("field-pattern voxel and component limits must be positive")
        if not 0.0 < self.field_pattern_max_grid_fraction <= 1.0:
            raise ValueError("field_pattern_max_grid_fraction must be in (0,1]")
        if not 0.0 <= self.field_pattern_containment_min_fraction <= 1.0:
            raise ValueError("field_pattern_containment_min_fraction must be in [0,1]")
        if self.field_pattern_repeated_part_distance < 0.0:
            raise ValueError("field_pattern_repeated_part_distance must be non-negative")
        if self.field_pattern_min_part_scale_ratio < 1.0:
            raise ValueError("field_pattern_min_part_scale_ratio must be at least 1")

    @classmethod
    def from_environment(cls) -> "AnalysisConfig":
        """Load optional JSON config, then use environment values as overrides."""
        raw: Dict[str, Any] = {}
        config_path = os.environ.get("ESM_CONFIG_PATH")
        if config_path:
            with Path(config_path).expanduser().open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, Mapping):
                raise ValueError("ESM_CONFIG_PATH must contain a JSON object")
            raw = dict(payload)

        path_payload = dict(raw.pop("paths", {}))
        root = os.environ.get("ESM_ANALYSIS_ROOT", path_payload.get("root", "esm_analysis"))
        run_id = os.environ.get("ESM_RUN_ID", path_payload.get("run_id", "default"))
        namespace_override = os.environ.get("ESM_FIELD_CACHE_NAMESPACE")
        if namespace_override is None:
            namespace_override = path_payload.get("field_cache_namespace")
        preliminary_paths = ArtifactPaths(Path(root), str(run_id))

        cpu_default = max(1, os.cpu_count() or 1)
        defaults = cls(paths=preliminary_paths, analysis_n_jobs=cpu_default, dft_workers=max(1, cpu_default // 2))
        values: Dict[str, Any] = {
            field.name: raw.get(field.name, getattr(defaults, field.name))
            for field in dataclasses.fields(cls)
            if field.name != "paths"
        }
        env_map = {
            "cache_version": "ESM_CACHE_VERSION",
            "descriptor_regime": "ESM_DESCRIPTOR_REGIME",
            "execution_mode": "ESM_EXECUTION_MODE",
            "resolution": "ESM_RESOLUTION",
            "field_grid_maximum_seed_limit": "ESM_FIELD_GRID_MAXIMUM_SEED_LIMIT",
            "analysis_n_jobs": "ESM_ANALYSIS_N_JOBS",
            "dft_workers": "ESM_DFT_WORKERS",
            "blas_threads": "ESM_BLAS_THREADS",
            "dft_backend": "ESM_DFT_BACKEND",
            "use_gpu_label_assignment": "ESM_GPU_LABEL_ASSIGNMENT",
            "online_merge_min_similarity": "ESM_ONLINE_MERGE_MIN_SIMILARITY",
            "online_merge_enabled": "ESM_ONLINE_MERGE_ENABLED",
            "online_merge_screen_max_prototypes": "ESM_ONLINE_MERGE_SCREEN_MAX_PROTOTYPES",
            "online_append_chunk_molecules": "ESM_ONLINE_APPEND_CHUNK_MOLECULES",
            "batch_validation_cache_only": "ESM_BATCH_VALIDATION_CACHE_ONLY",
            "preliminary_max_cache_files": "ESM_PRELIMINARY_MAX_CACHE_FILES",
            "preliminary_max_candidates": "ESM_PRELIMINARY_MAX_CANDIDATES",
            "default_qm9_samples": "ESM_QM9_SAMPLES",
            "adopt_legacy_cache": "ESM_ADOPT_LEGACY_CACHE",
            "enable_static_build_proxy": "ESM_ENABLE_STATIC_BUILD_PROXY",
            "field_patterns_enabled": "ESM_FIELD_PATTERNS_ENABLED",
            "field_pattern_min_voxels": "ESM_FIELD_PATTERN_MIN_VOXELS",
            "field_pattern_max_regions_per_level": "ESM_FIELD_PATTERN_MAX_REGIONS_PER_LEVEL",
            "field_pattern_max_grid_fraction": "ESM_FIELD_PATTERN_MAX_GRID_FRACTION",
            "field_pattern_include_boundary_components": "ESM_FIELD_PATTERN_INCLUDE_BOUNDARIES",
            "field_pattern_containment_min_fraction": "ESM_FIELD_PATTERN_CONTAINMENT_MIN_FRACTION",
            "field_pattern_repeated_part_distance": "ESM_FIELD_PATTERN_REPEAT_DISTANCE",
            "field_pattern_min_part_scale_ratio": "ESM_FIELD_PATTERN_MIN_PART_SCALE_RATIO",
            "field_pattern_descriptor_sample_voxels": "ESM_FIELD_PATTERN_DESCRIPTOR_SAMPLE_VOXELS",
            "field_pattern_containment_sample_voxels": "ESM_FIELD_PATTERN_CONTAINMENT_SAMPLE_VOXELS",
        }
        for name, env_name in env_map.items():
            if env_name in os.environ:
                values[name] = os.environ[env_name]

        values["descriptor_regime"] = str(values["descriptor_regime"]).lower()
        values["execution_mode"] = str(values["execution_mode"]).lower()
        values["dft_backend"] = str(values["dft_backend"]).lower()
        values["shell_radii"] = tuple(float(x) for x in values["shell_radii"])
        for name, minimum in (
            ("field_grid_maximum_seed_limit", 1), ("analysis_n_jobs", 1),
            ("field_pattern_min_voxels", 1), ("field_pattern_max_regions_per_level", 1),
            ("field_pattern_descriptor_sample_voxels", 1),
            ("field_pattern_containment_sample_voxels", 1),
            ("dft_workers", 1), ("blas_threads", 1), ("dictionary_batch_size", 64),
            ("dictionary_scratch_mb", 64), ("dictionary_cv_folds", 1),
            ("dictionary_cv_max_candidates", 1000),
            ("dictionary_cv_screen_max_candidates", 500),
            ("dictionary_cv_screen_folds", 1), ("dictionary_cv_finalists", 1),
            ("semantic_max_molecules_per_class", 4),
            ("profile_compression_sample", 500), ("letter_archive_shard_rows", 1),
            ("visualization_all_molecule_limit", 1),
            ("visualization_representatives_per_class", 1),
            ("visualization_representative_molecule_limit", 1),
            ("online_merge_screen_max_prototypes", 1),
            ("online_append_chunk_molecules", 1),
            ("preliminary_max_cache_files", 1),
            ("preliminary_max_candidates", 1),
            ("default_qm9_samples", 1),
            ("n_bootstrap_cluster", 0), ("n_bootstrap_dim", 0),
            ("n_null_perm", 0), ("n_comm_trials", 0), ("n_control_max", 0),
            ("n_control_reps", 0), ("max_pairs_per_quadrant", 1),
            ("max_classes_for_commutation", 1),
        ):
            values[name] = _integer(values[name], int(getattr(defaults, name)), minimum)
        for name in (
            "cp_ring_search", "use_gpu_label_assignment", "visualization_enabled",
            "letter_archive_enabled", "online_merge_enabled",
            "enable_static_build_proxy", "adopt_legacy_cache",
            "batch_validation_cache_only",
            "field_patterns_enabled", "field_pattern_include_boundary_components",
        ):
            values[name] = _truth(values[name])
        for name in (
            "resolution", "r0", "sigma", "cp_bond_search_radius", "cp_max_step",
            "cp_grad_tol", "cp_dedupe_radius", "online_merge_min_similarity",
            "field_pattern_max_grid_fraction", "field_pattern_containment_min_fraction",
            "field_pattern_repeated_part_distance", "field_pattern_min_part_scale_ratio",
        ):
            values[name] = float(values[name])
        for name in ("l_max", "n_points_per_shell", "cp_n_seeds_per_axis"):
            values[name] = _integer(values[name], int(getattr(defaults, name)), 1)
        values["secondary_properties"] = tuple(str(x) for x in values["secondary_properties"])
        values["field_pattern_quantiles"] = tuple(float(x) for x in values["field_pattern_quantiles"])
        # A cube is reusable only if the actual field-generation settings
        # agree.  When no explicit namespace is supplied, make grid
        # resolution part of the default rather than reusing a 0.20 cube for
        # a 0.15 request by accident.
        namespace = namespace_override or f"density_pbe_def2svp_res{int(round(values['resolution'] * 100)):03d}"
        paths = ArtifactPaths(Path(root), str(run_id), str(namespace))
        return cls(paths=paths, **values)

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["paths"]["root"] = str(self.paths.root)
        data["shell_radii"] = list(self.shell_radii)
        return data

    def analysis_signature_payload(self) -> Dict[str, Any]:
        """Everything that makes one extracted descriptor comparable to another.

        Deliberately excludes online-consolidation settings (merge enabled,
        similarity threshold, screen size): those change how already-extracted
        candidates get grouped into named classes, not what a candidate's
        descriptor vector is. Coupling them into this signature used to mean
        that changing only a merge threshold invalidated the CorpusLedger's
        "already completed" status for every molecule ever processed under the
        old signature -- forcing a full re-walk of the corpus (re-staging every
        molecule, rebuilding the online alphabet state from empty) even though
        no field, cube, or candidate needed to be recomputed. At a few hundred
        million molecules that made ordinary hyperparameter tuning prohibitively
        expensive. Use ``recluster_from_archive(s)`` in ``esm_core.online_archive``
        to explore a different merge policy instead: it replays the archived,
        already-extracted candidates and pays only the (much cheaper) streaming
        assignment and merge cost, with no DFT, cube, or ledger access at all.

        Lives here rather than in ``esm_master_analytics.py`` (the only other
        place it was ever computed) specifically so a light caller -- the
        control centre UI, in particular -- can preview whether a parameter
        change would reuse an existing candidate archive or trigger fresh
        extraction, without importing anything that pulls in PySCF/RDKit.
        ``field_kind``/``environment`` are not yet exposed as
        ``AnalysisConfig`` fields (this codebase only ever generates a
        vacuum density), so they are fixed here rather than configurable;
        if that ever changes, add the corresponding fields above and read
        them here instead, the same way ``level_of_theory`` now is (added
        specifically so a run can be pointed at a dataset supplying its own
        precomputed cubes at a different level of theory, e.g. EDBench's
        Psi4/B3LYP -- see `_extract_record_worker`'s precomputed-cube
        adoption path -- without silently mislabelling its signature).
        """
        return {
            "cache_version": self.cache_version,
            "candidate_discovery_method": self.candidate_discovery_method,
            "descriptor_regime": self.descriptor_regime,
            "shell_radii": list(self.shell_radii),
            "r0": self.r0,
            "sigma": self.sigma,
            "l_max": self.l_max,
            "n_points_per_shell": self.n_points_per_shell,
            "critical_point_search": {
                "bond_search_radius": self.cp_bond_search_radius,
                "ring_search": self.cp_ring_search,
                "n_seeds_per_axis": self.cp_n_seeds_per_axis,
                "max_step": self.cp_max_step,
                "grad_tol": self.cp_grad_tol,
                "dedupe_radius": self.cp_dedupe_radius,
                "field_grid_maximum_seed_limit": self.field_grid_maximum_seed_limit,
            },
            "field_kind": "density",
            "level_of_theory": self.level_of_theory,
            "environment": "vacuum",
        }

    def write_snapshot(self) -> Path:
        self.paths.ensure()
        destination = self.paths.config_snapshot_path
        temporary = destination.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
