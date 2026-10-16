"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
fundamentality.py
==================
Doc §12 ("Критерий «фундаментальное vs производное»"), operationalized at
§15 step 10 ("Разреженное разложение на фундаментальный базис: применение
критерия из §12 к полному набору кандидатов-дескрипторов для отделения
фундаментальных единиц от производных комбинаций").

The doc is explicit that raw signal magnitude is the wrong proxy for
"fundamental" (§12: "искать фундаментальные «буквы» нужно по критерию
сжимаемости/переиспользуемости, а не по величине вклада в сигнал") -- a
parameter can dominate a descriptor's norm and still be derived noise, and
a barely-there parameter can be the fundamental unit. The right proxy is
reuse ACROSS THE CORPUS: does this motif get reconstructed, at a fixed
sparsity, from dictionary atoms that also recur in *other, unrelated*
molecules -- or does reconstructing it always pull in atoms that are
private to its own narrow neighborhood?

Pipeline (mirrors the doc's own wording step by step):
    1. `fit_shared_dictionary` -- one dictionary-learning fit
       (`sklearn.decomposition.DictionaryLearning`, OMP sparse coding)
       across the pooled descriptor vectors of the WHOLE corpus (every
       `corpus.LetterCandidate.descriptor.to_vector()`, from every
       molecule) at a fixed sparsity -- the corpus-wide "reusable
       vocabulary" the doc's dictionary-learning principle asks for.
    2. `active_atoms` -- for each candidate, which atoms its code actually
       uses.
    3. `reuse_breadth` -- of the *other* molecules in the corpus (not the
       candidate's own `source_molecule_id`), what fraction have at least
       one candidate that also uses one of the same atoms? High breadth =
       these atoms are a shared, corpus-wide vocabulary element, not
       private to one molecule's context.
    4. `reuse_breadth_confidence` -- is this breadth higher than chance?
       Permute which candidate belongs to which molecule (holding the
       learned codes -- and hence each candidate's active atoms -- fixed)
       and recompute breadth under the permuted labeling, many times. This
       is the same "compare to a matched null, not an arbitrary constant"
       idiom used throughout this codebase (`commutation.py`'s
       random-direction null, `corpus.py`'s bootstrap cohesion), applied
       here to ask "is this atom's spread across molecules more than
       coincidence", not to an arbitrary reuse-fraction cutoff.
    5. `compute_fundamentality` -- combines reconstruction fidelity (is
       this candidate well described at all by the shared dictionary,
       i.e. is it compressible) with reuse breadth clearing its own null
       band (is what it compresses INTO actually shared, not private). A
       candidate is only called fundamental on BOTH counts: being
       compressible into an idiosyncratic, unshared combination is
       exactly the doc's "derived" case, and being widely "reused" by an
       atom set that doesn't even describe this candidate well would be a
       coincidence, not evidence this candidate itself is a letter.

What this module deliberately does NOT do (a distinct, complementary check,
built in `unsupervised_validation.py` instead, §16.2): compare corpus
compressibility against a scrambled/randomized control FIELD of the same
scale. The doc keeps this apart from §12's own criterion (§16.2: "Помимо
проверки корреляции инварианта с именованными свойствами, проверяется...
даёт ли пространство инварианта лучшую сжимаемость (§12) по сравнению с
контрольной моделью, построенной на скремблированных... патчах" -- notice
this is phrased as comparing §12's own measure against a scrambled control,
i.e. §16.2 consumes this module's `fit_shared_dictionary` /
`reconstruction_fidelity` machinery, it does not replace the null used
inside this module). The null used HERE is a permutation of molecule labels
over the *real* learned codes (step 4 below); `unsupervised_validation.py`'s
null is a scrambled/random reconstruction of the *field itself*, run
through the full extraction pipeline -- a different question (does the
corpus compress at all, vs. is a candidate's reuse more than relabeling
chance) answered with different, non-overlapping machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import warnings

import numpy as np

from .corpus import LetterCandidate
from .schema import ConfidenceBlock

# Set at IMPORT time, not inside a per-call `warnings.catch_warnings()`
# scope (see the narrower one still inside `_omp_codes` below). This matters
# specifically because `_omp_codes` is called with `n_jobs > 1` at real
# corpus scale: `sklearn.decomposition.sparse_encode(..., n_jobs=n_jobs)`
# then dispatches OMP row-fits to joblib worker processes, and a
# `catch_warnings()` block entered in the parent's current call stack has NO
# effect on those workers -- each worker's own warnings-filter state was
# fixed when the worker started (fork) or gets re-established by its own
# fresh top-level import of this module (spawn); a context manager entered
# later in the parent process reaches neither. A module-level call is either
# copied into the worker's memory at fork time or re-executed verbatim on
# spawn's re-import, so it is the one form of suppression that reliably
# reaches worker processes regardless of joblib's backend or start method.
# `_omp_codes`'s docstring explains why this specific, narrow message is
# safe to suppress globally: it documents a handled case (OMP returning a
# valid solution with fewer than the requested nonzero coefficients because
# the dictionary contains a locally dependent atom subset), not a silent
# numerical failure -- the actual active-atom count survives in the fit and
# is exposed by `diagnose_dictionary_fit`.
warnings.filterwarnings(
    "ignore",
    message="Orthogonal matching pursuit ended prematurely due to linear dependence.*",
    category=RuntimeWarning,
    module=r"sklearn\..*",
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class FundamentalityConfig:
    """Dictionary-learning hyperparameters. Both `n_atoms` and
    `n_nonzero_coefs` are corpus-scale choices, not universal constants.
    Select them on molecule-held-out data with
    `select_dictionary_size_by_cv`; the old `suggest_n_atoms` helper remains
    only for backwards-compatible exploratory use.

    n_atoms             -- dictionary size (number of candidate "basis
                           letters" the shared vocabulary is allowed).
    n_nonzero_coefs     -- sparsity: maximum atoms used to reconstruct any
                           candidate's descriptor vector (OMP sparsity --
                           the doc's "разреженное... кодирование").
    n_null_permutations -- molecule-label permutations for the
                           reuse-breadth null (module docstring, step 4).
    active_atom_tol     -- coefficient magnitude below which an atom counts
                           as inactive in a candidate's code (numerical
                           zero, not a meaningful but weak contribution --
                           OMP codes are exactly sparse up to floating
                           point, so this only needs to clear solver noise).
    rng_seed
    """

    n_atoms: int
    n_nonzero_coefs: int
    n_null_permutations: int = 500
    active_atom_tol: float = 1e-8
    rng_seed: int = 0
    # Computational controls, not evidentiary thresholds.  A bounded batch
    # and compiled OMP parallelism are required at corpus scale.
    batch_size: int = 4096
    n_jobs: int = 1
    kmeans_n_init: int = 3

    def __post_init__(self) -> None:
        if self.n_nonzero_coefs > self.n_atoms:
            raise ValueError("n_nonzero_coefs cannot exceed n_atoms")
        if self.n_atoms < 1 or self.n_nonzero_coefs < 1:
            raise ValueError("n_atoms and n_nonzero_coefs must be positive")
        if self.n_null_permutations < 1:
            raise ValueError("n_null_permutations must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        if self.kmeans_n_init < 1:
            raise ValueError("kmeans_n_init must be positive")


def suggest_n_atoms(candidates: Sequence[LetterCandidate], atoms_per_candidate: float = 0.5) -> int:
    """Data-driven starting point for `FundamentalityConfig.n_atoms`,
    analogous to `corpus.suggest_distance_threshold`: a dictionary needs
    enough atoms to represent genuine corpus-wide diversity, but a
    dictionary with as many atoms as candidates lets every candidate claim
    its own private atom, which trivially maximizes reconstruction
    fidelity for everyone while saying nothing about reuse -- defeating
    the entire point of this module. `atoms_per_candidate < 1` keeps the
    dictionary smaller than the candidate pool by construction, forcing
    atoms to be shared across multiple candidates. Still a heuristic
    starting point, not a substitute for checking results are stable
    across a range of `n_atoms` (same caveat `suggest_distance_threshold`
    carries).
    """
    n = len(candidates)
    if n < 2:
        raise ValueError("need at least 2 candidates to suggest a dictionary size")
    return max(2, int(round(n * atoms_per_candidate)))


# --------------------------------------------------------------------------
# step 1: shared dictionary fit
# --------------------------------------------------------------------------


@dataclass
class DictionaryFit:
    dictionary: np.ndarray  # (n_atoms, n_features)
    codes: object  # (n_candidates, n_atoms), normally a scipy CSR matrix
    config: FundamentalityConfig
    # The dictionary lives in this transformed coordinate system.  Keeping
    # its parameters prevents validation data from being re-standardised
    # independently (which would leak held-out information).
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def _column_standardize(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean / unit-std per feature. Near-constant columns (std ~ 0)
    get std=1 so they stay near zero after centering and never blow up
    OMP. Returns (X_std, mean, std)."""
    mu = vectors.mean(axis=0)
    sigma = vectors.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (vectors - mu) / sigma, mu, sigma


def _standardize_with_fit(
    vectors: np.ndarray, feature_mean: np.ndarray, feature_scale: np.ndarray
) -> np.ndarray:
    """Apply the training split transform to validation/new vectors."""
    safe = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    return (safe - feature_mean) / feature_scale


def _omp_codes(
    X: np.ndarray,
    dictionary: np.ndarray,
    n_nonzero: int,
    *,
    n_jobs: int = 1,
    batch_size: int = 4096,
):
    """Sparse-code rows of X in bounded batches.

    The result is CSR rather than an N-by-K dense array.  At fixed OMP
    sparsity it stores O(N*s) coefficients instead of O(N*K), which is the
    difference between a usable and impossible RAM footprint when K grows.
    """
    from scipy import sparse

    n, dim = X.shape
    n_atoms = dictionary.shape[0]
    k = min(n_nonzero, n_atoms, dim)
    if n == 0:
        return sparse.csr_matrix((0, n_atoms), dtype=X.dtype)

    try:
        from sklearn.decomposition import sparse_encode

        batches = []
        for start in range(0, n, batch_size):
            # A redundant overcomplete dictionary can contain a locally
            # dependent atom set.  OMP then returns a valid solution with
            # fewer than k atoms (k is a maximum), but sklearn emits one
            # RuntimeWarning per row.  Suppress only that documented case;
            # the actual active-atom count remains in the fit and is exposed
            # by diagnose_dictionary_fit, so no numerical failure is hidden.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Orthogonal matching pursuit ended prematurely due to linear dependence.*",
                    category=RuntimeWarning,
                    module=r"sklearn\..*",
                )
                encoded = sparse_encode(
                    X[start : start + batch_size],
                    dictionary,
                    algorithm="omp",
                    n_nonzero_coefs=k,
                    n_jobs=n_jobs,
                )
            batches.append(sparse.csr_matrix(np.asarray(encoded, dtype=X.dtype)))
        return sparse.vstack(batches, format="csr")
    except Exception:
        from sklearn.linear_model import OrthogonalMatchingPursuit

        dictionary_t = dictionary.T
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
        row_indices: List[int] = []
        column_indices: List[int] = []
        values: List[float] = []
        for i, x in enumerate(X):
            if not np.any(np.abs(x) > 1e-15):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Orthogonal matching pursuit ended prematurely due to linear dependence.*",
                        category=RuntimeWarning,
                        module=r"sklearn\..*",
                    )
                    omp.fit(dictionary_t, x)
                active = np.flatnonzero(np.abs(omp.coef_) > 0.0)
                row_indices.extend([i] * len(active))
                column_indices.extend(active.tolist())
                values.extend(omp.coef_[active].astype(float).tolist())
            except Exception:
                dots = dictionary @ x
                j = int(np.argmax(np.abs(dots)))
                denom = float(dictionary[j] @ dictionary[j]) + 1e-15
                row_indices.append(i)
                column_indices.append(j)
                values.append(float(dots[j] / denom))
        return sparse.csr_matrix(
            (np.asarray(values, dtype=X.dtype), (row_indices, column_indices)),
            shape=(n, n_atoms),
            dtype=X.dtype,
        )


def fit_shared_dictionary(
    candidates: Sequence[LetterCandidate], config: FundamentalityConfig
) -> DictionaryFit:
    """Module docstring step 1: one dictionary, learned jointly across
    every candidate's descriptor vector in the corpus, at fixed sparsity
    `config.n_nonzero_coefs`.

    Implementation (rewritten after real-corpus failure):
        sklearn `DictionaryLearning` alternating minimization was observed
        to collapse to empty codes for ~70%+ of real bispectrum vectors
        (wild per-candidate norms + near-constant feature columns). The
        reliable construction used here is:

          1. Column-standardize the descriptor matrix (constant features
             neutralized; scale differences no longer dominate).
          2. Build dictionary atoms as MiniBatchKMeans centers in that
             standardized space (or a random data subsample if k-means
             cannot run) -- atoms are real corpus modes, not an unstable
             alternating fixed point.
          3. Sparse-code every candidate with row-wise OMP at fixed
             `n_nonzero_coefs`.

    Reconstruction fidelity (§12) is evaluated in the SAME standardized
    space the dictionary was fit in, so a fidelity number is comparable
    across candidates regardless of raw bispectrum magnitude. Codes and
    dictionary are returned already in that space; callers should not
    mix them with raw `descriptor.to_vector()` without re-standardizing.
    """
    if len(candidates) < 2:
        raise ValueError("need at least 2 candidates to learn a shared dictionary")

    # float32 halves the working set and is sufficient after standardisation.
    vectors = np.stack([c.descriptor.to_vector() for c in candidates]).astype(np.float32)
    # Drop non-finite rows from dictionary construction (keep zeros for them
    # in the final codes so index alignment with `candidates` is preserved).
    finite_mask = np.all(np.isfinite(vectors), axis=1)
    if not np.any(finite_mask):
        raise ValueError("all candidate descriptor vectors are non-finite")

    X_all, feature_mean, feature_scale = _column_standardize(
        np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    )
    X_fit = X_all[finite_mask]
    n_atoms = min(config.n_atoms, max(1, X_fit.shape[0]))

    rng = np.random.default_rng(config.rng_seed)
    dictionary: np.ndarray
    if X_fit.shape[0] >= n_atoms and X_fit.shape[1] >= 1:
        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=n_atoms,
                random_state=config.rng_seed,
                n_init=config.kmeans_n_init,
                batch_size=min(config.batch_size, X_fit.shape[0]),
                compute_labels=False,
            )
            dictionary = km.fit(X_fit).cluster_centers_
        except Exception:
            idx = rng.choice(X_fit.shape[0], size=n_atoms, replace=False)
            dictionary = X_fit[idx].copy()
    else:
        idx = rng.choice(X_fit.shape[0], size=n_atoms, replace=True)
        dictionary = X_fit[idx].copy()

    # Guard against zero-norm atoms (degenerate k-means centers).
    atom_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    dictionary = dictionary / np.maximum(atom_norms, 1e-12)

    codes = _omp_codes(
        X_all, dictionary, config.n_nonzero_coefs,
        n_jobs=config.n_jobs, batch_size=config.batch_size
    )
    # Re-pack config if n_atoms was clamped.
    cfg = config
    if n_atoms != config.n_atoms:
        cfg = FundamentalityConfig(
            n_atoms=n_atoms,
            n_nonzero_coefs=min(config.n_nonzero_coefs, n_atoms),
            n_null_permutations=config.n_null_permutations,
            active_atom_tol=config.active_atom_tol,
            rng_seed=config.rng_seed,
            batch_size=config.batch_size,
            n_jobs=config.n_jobs,
            kmeans_n_init=config.kmeans_n_init,
        )
    return DictionaryFit(
        dictionary=dictionary,
        codes=codes,
        config=cfg,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def diagnose_dictionary_fit(
    candidates: Sequence[LetterCandidate], fit: DictionaryFit, norm_floor: float = 1e-6
) -> Dict[str, object]:
    """Distinguishes a genuine null result ("this candidate's descriptor is
    real but shares nothing with the rest of the corpus") from a degenerate
    fit ("the dictionary-learning procedure itself failed to find ANY
    usable code for a candidate whose descriptor is not even close to the
    zero vector") -- these currently look IDENTICAL in `FundamentalityResult`
    (empty `active_atoms`, `reconstruction_fidelity=0.0`), but have very
    different implications: the first is data the doc's §12 criterion wants
    to see (it is what "not fundamental" is supposed to mean); the second
    is a numerics problem that should not be read as evidence about the
    theory at all. Call this after `fit_shared_dictionary` and before
    trusting a corpus-wide "N/M candidates are not fundamental" summary.

    Returns a plain dict (not part of the doc's schema -- this is pipeline
    self-diagnostics, not a theory-relevant measurement) with:
        n_empty_code                 -- candidates with zero active atoms
        n_empty_despite_nonzero_norm -- of those, how many have a descriptor
                                         norm above `norm_floor` -- THESE are
                                         the suspicious ones; a large count
                                         here means treat is_fundamental=False
                                         results from this fit as unreliable
                                         until re-run/investigated further,
                                         not as a real §12 result.
        vector_norm_range            -- (min, max) descriptor norm in the
                                         corpus, to sanity-check for the
                                         wildly-varying-scale pattern that
                                         reproduced this failure mode in
                                         testing.
    """
    vectors = np.stack([c.descriptor.to_vector() for c in candidates])
    norms = np.linalg.norm(vectors, axis=1)
    tol = fit.config.active_atom_tol
    active_counts = np.fromiter(
        (len(active_atoms(fit.codes[i], tol)) for i in range(len(candidates))),
        dtype=int,
        count=len(candidates),
    )
    empty_mask = active_counts == 0
    suspicious = empty_mask & (norms > norm_floor)
    return {
        "n_candidates": len(candidates),
        "n_empty_code": int(empty_mask.sum()),
        "n_empty_despite_nonzero_norm": int(suspicious.sum()),
        "vector_norm_range": (float(norms.min()), float(norms.max())),
        "n_dictionary_atoms": int(fit.dictionary.shape[0]),
        "active_atom_count": {
            "min": int(active_counts.min()) if len(active_counts) else 0,
            "median": float(np.median(active_counts)) if len(active_counts) else 0.0,
            "max": int(active_counts.max()) if len(active_counts) else 0,
            "n_below_requested_max": int(
                np.sum(active_counts < fit.config.n_nonzero_coefs)
            ),
        },
    }


# --------------------------------------------------------------------------
# steps 2-3: active atoms, reconstruction fidelity, reuse breadth
# --------------------------------------------------------------------------


def _dense_code(code: np.ndarray) -> np.ndarray:
    """Convert one dense or sparse code row to a 1-D dense vector."""
    if hasattr(code, "toarray"):
        return np.asarray(code.toarray()).ravel()
    return np.asarray(code).ravel()


def active_atoms(code: np.ndarray, tol: float) -> Set[int]:
    """Indices whose sparse-code coefficient is numerically active."""
    if hasattr(code, "indices") and hasattr(code, "data"):
        indices = np.asarray(code.indices)
        data = np.asarray(code.data)
        return set(indices[np.abs(data) > tol].tolist())
    return set(np.flatnonzero(np.abs(_dense_code(code)) > tol).tolist())


def reconstruction_fidelity(vector: np.ndarray, code: np.ndarray, dictionary: np.ndarray) -> float:
    """Normalised reconstruction fidelity in dictionary-fit coordinates."""
    dense_code = _dense_code(code)
    residual = vector - dense_code @ dictionary
    vec_norm = np.linalg.norm(vector)
    if vec_norm < 1e-12:
        return 1.0 if np.linalg.norm(residual) < 1e-12 else 0.0
    return float(max(0.0, 1.0 - np.linalg.norm(residual) / vec_norm))


def _molecule_atom_sets(
    molecule_ids: Sequence[str], codes: np.ndarray, tol: float
) -> Dict[str, Set[int]]:
    """Union of active atoms across every candidate belonging to each
    molecule -- the per-molecule "vocabulary" `reuse_breadth` checks a
    candidate's own atoms against."""
    out: Dict[str, Set[int]] = {}
    for j, mol in enumerate(molecule_ids):
        out.setdefault(mol, set()).update(active_atoms(codes[j], tol))
    return out


def reuse_breadth(atoms: Set[int], own_molecule: str, molecule_atom_sets: Dict[str, Set[int]]) -> float:
    """Module docstring step 3, literally: fraction of *other* molecules
    (excluding `own_molecule`) whose candidates collectively use at least
    one atom in `atoms`. 0.0 if `atoms` is empty (a candidate reconstructed
    from zero atoms trivially reuses nothing) or if there are no other
    molecules in the corpus to test reuse against."""
    if not atoms:
        return 0.0
    others = {m: a for m, a in molecule_atom_sets.items() if m != own_molecule}
    if not others:
        return 0.0
    sharing = sum(1 for a in others.values() if a & atoms)
    return sharing / len(others)


# --------------------------------------------------------------------------
# step 4: null for reuse breadth
# --------------------------------------------------------------------------


def _all_reuse_breadths(
    molecule_ids: Sequence[str], codes: object, tol: float
) -> np.ndarray:
    """`reuse_breadth` for every candidate at once, against the molecule
    labeling given (real or permuted) -- computed together so the
    per-molecule atom sets are only built once per labeling, not once per
    candidate (this is what keeps the null in `reuse_breadth_confidences`
    at O(n_permutations * n_candidates) instead of
    O(n_permutations * n_candidates^2))."""
    molecule_atom_sets = _molecule_atom_sets(molecule_ids, codes, tol)
    n_molecules = len(molecule_atom_sets)
    if n_molecules < 2:
        return np.zeros(len(molecule_ids), dtype=np.float32)

    # Invert the molecule vocabulary once.  The former implementation called
    # `reuse_breadth` per candidate; that rebuilt and scanned the entire
    # molecule dictionary for every candidate, i.e. O(N*M) per permutation
    # despite claiming O(N).  Sparse codes contain at most s active atoms,
    # so unioning their precomputed molecule sets is the intended cost.
    atom_to_molecules: Dict[int, Set[str]] = {}
    for molecule_id, atoms in molecule_atom_sets.items():
        for atom in atoms:
            atom_to_molecules.setdefault(atom, set()).add(molecule_id)

    breadths = np.empty(len(molecule_ids), dtype=np.float32)
    denominator = n_molecules - 1
    for candidate_index, molecule_id in enumerate(molecule_ids):
        shared_molecules: Set[str] = set()
        for atom in active_atoms(codes[candidate_index], tol):
            shared_molecules.update(atom_to_molecules.get(atom, ()))
        shared_molecules.discard(molecule_id)
        breadths[candidate_index] = len(shared_molecules) / denominator
    return breadths


def reuse_breadth_confidences(
    candidates: Sequence[LetterCandidate], fit: DictionaryFit
) -> List[ConfidenceBlock]:
    """Module docstring step 4, for every candidate at once (see
    `_all_reuse_breadths` for why this is batched rather than per-candidate):
    `score` is the observed reuse breadth; `interval` is the 2.5/97.5
    percentile band of the null breadth obtained by permuting which
    candidate has which `source_molecule_id` (holding each candidate's
    learned code -- and hence its active atoms -- fixed),
    `config.n_null_permutations` times. A candidate's atoms count as
    genuinely *reused*, not just incidentally shared, only when `score`
    clears `interval[1]` -- the same "beat a matched null, not an
    arbitrary constant" idiom as `commutation.py`.

    The 2.5/97.5% interval is exact for the realised permutations, but the
    implementation keeps only the necessary lower/upper order statistics
    per candidate rather than a ``n_permutations × n_candidates`` matrix.
    At the pipeline's 40 permutations this is four float32 values per
    candidate instead of forty float64 values; crucial for a large corpus.
    """
    config = fit.config
    molecule_ids = [c.source_molecule_id for c in candidates]
    codes = fit.codes
    tol = config.active_atom_tol

    observed = _all_reuse_breadths(molecule_ids, codes, tol)

    rng = np.random.default_rng(config.rng_seed)
    n = len(candidates)
    molecule_ids_arr = np.array(molecule_ids)
    n_null = config.n_null_permutations

    # NumPy's linear percentile interpolation at q uses the two order
    # statistics around h=(B-1)q. Retain just those tails, not every null
    # sample. `low_tail`/`high_tail` are kept ascending by `np.partition`.
    lower_h = (n_null - 1) * 0.025
    upper_h = (n_null - 1) * 0.975
    n_low_keep = int(np.ceil(lower_h)) + 1
    n_high_keep = n_null - int(np.floor(upper_h))
    low_tail = np.full((n_low_keep, n), np.inf, dtype=np.float32)
    high_tail = np.full((n_high_keep, n), -np.inf, dtype=np.float32)
    for b in range(config.n_null_permutations):
        permuted = rng.permutation(molecule_ids_arr)
        null_values = _all_reuse_breadths(list(permuted), codes, tol)
        low_tail = np.sort(np.partition(
            np.concatenate((low_tail, null_values[None, :]), axis=0),
            kth=n_low_keep - 1,
            axis=0,
        )[:n_low_keep], axis=0)
        high_tail = np.sort(np.partition(
            np.concatenate((high_tail, null_values[None, :]), axis=0),
            kth=1,
            axis=0,
        )[-n_high_keep:], axis=0)

    lower_index = int(np.floor(lower_h))
    lower_fraction = lower_h - lower_index
    null_lower = (
        low_tail[lower_index]
        if lower_fraction == 0.0 else
        low_tail[lower_index] * (1.0 - lower_fraction) + low_tail[lower_index + 1] * lower_fraction
    )
    high_base_index = int(np.floor(upper_h)) - (n_null - n_high_keep)
    upper_fraction = upper_h - np.floor(upper_h)
    null_upper = (
        high_tail[high_base_index]
        if upper_fraction == 0.0 else
        high_tail[high_base_index] * (1.0 - upper_fraction) + high_tail[high_base_index + 1] * upper_fraction
    )

    blocks: List[ConfidenceBlock] = []
    for i in range(n):
        blocks.append(
            ConfidenceBlock(
                n=config.n_null_permutations,
                method="permutation",
                score=float(observed[i]),
                interval=(float(null_lower[i]), float(null_upper[i])),
                corpus_version="n/a",
                stale=False,
            )
        )
    return blocks


# --------------------------------------------------------------------------
# step 5: combined verdict
# --------------------------------------------------------------------------


@dataclass
class FundamentalityResult:
    """Per-candidate output of the §12 criterion (§15 step 10). Not part of
    the doc's fixed schema block (`schema.py`) -- §12/step 10 is a
    corpus-wide diagnostic that feeds into deciding which discovered
    `LetterClass`es deserve continued treatment as fundamental "letters",
    not a field the doc's `letter_class` schema itself reserves a slot
    for."""

    source_molecule_id: str
    active_atoms: Set[int]
    reconstruction_fidelity: float
    reuse_breadth: ConfidenceBlock
    is_fundamental: bool
    # True iff BOTH: reconstruction_fidelity is at least the corpus median
    # (this candidate is at least as compressible as a typical corpus
    # member -- a data-relative floor, not an arbitrary constant, mirroring
    # corpus.suggest_distance_threshold's use of a corpus quantile rather
    # than a fixed number) AND reuse_breadth.score clears
    # reuse_breadth.interval[1] (its atoms are reused across other
    # molecules significantly more than the permutation null).


def compute_fundamentality(
    candidates: Sequence[LetterCandidate], fit: DictionaryFit
) -> List[FundamentalityResult]:
    """Module docstring step 5: combine reconstruction fidelity and
    reuse-breadth significance into one verdict per candidate.

    The reconstruction-fidelity floor is the corpus's own median fidelity
    under this dictionary, not a fixed constant -- consistent with the
    doc's warning (§12) against judging "fundamental" by an arbitrary
    magnitude cutoff. A candidate whose fidelity is below the corpus
    median is, by construction, less compressible than a typical corpus
    member under the *shared* vocabulary -- whatever is special about it
    is not (yet) well captured by atoms this dictionary learned as broadly
    useful.
    """
    if fit.codes.shape[0] != len(candidates):
        raise ValueError("fit.codes does not match the given candidates -- was it fit on this pool?")

    # Same column-standardization as fit_shared_dictionary, so fidelity is
    # measured in the space the dictionary was actually learned in.
    raw = np.stack([c.descriptor.to_vector() for c in candidates]).astype(np.float32)
    vectors = _standardize_with_fit(raw, fit.feature_mean, fit.feature_scale)
    fidelities = [
        reconstruction_fidelity(vectors[i], fit.codes[i], fit.dictionary)
        for i in range(len(candidates))
    ]
    fidelity_floor = float(np.median(fidelities))

    confidences = reuse_breadth_confidences(candidates, fit)
    tol = fit.config.active_atom_tol

    results = []
    for i, c in enumerate(candidates):
        atoms = active_atoms(fit.codes[i], tol)
        conf = confidences[i]
        reused_beyond_chance = conf.interval is not None and conf.score > conf.interval[1]
        is_fundamental = (fidelities[i] >= fidelity_floor) and reused_beyond_chance
        results.append(
            FundamentalityResult(
                source_molecule_id=c.source_molecule_id,
                active_atoms=atoms,
                reconstruction_fidelity=fidelities[i],
                reuse_breadth=conf,
                is_fundamental=is_fundamental,
            )
        )
    return results


def select_dictionary_size_by_cv(
    candidates: Sequence[LetterCandidate],
    n_atoms_grid: Sequence[int] | None = None,
    n_nonzero_coefs_grid: Sequence[int] = (2, 3, 5),
    holdout_fraction: float = 0.3,
    n_folds: int = 3,
    max_candidates_per_fold: Optional[int] = 100_000,
    max_atoms: Optional[int] = None,
    batch_size: int = 4096,
    n_jobs: int = 1,
    screening_max_candidates_per_fold: Optional[int] = 2_000,
    screening_n_folds: int = 1,
    max_finalists: int = 3,
    rng_seed: int = 0,
) -> dict:
    """Select dictionary capacity without imposing a fixed size ratio.

    The selected score is an approximate *held-out two-part MDL code length*
    (bits per validation candidate), not reconstruction fidelity alone.
    Fidelity almost always rewards a larger dictionary, so maximising it
    would turn CV into a disguised "choose the largest allowed dictionary"
    rule.  The MDL score combines a Gaussian code for the held-out residual
    with the BIC cost of dictionary parameters and the support/coefficient
    cost of held-out sparse codes.  A capacity is therefore selected only
    when its reconstruction gain pays for its shared vocabulary.

    Splitting by source_molecule_id is essential: a candidate-wise split
    leaks multiple critical points from the same density field into train and
    validation, inflating apparent generalisation.

    With a bounded representative sample per fold, this remains a model
    selection step rather than a memory-sized version of the complete
    corpus.  The final dictionary must then be fitted with the selected
    capacity on the full corpus (or its streaming equivalent).  The return
    value includes an upper-edge flag; a best value at that edge means the
    grid must be extended before treating the selection as settled.
    """
    n = len(candidates)
    if n < 10:
        raise ValueError("need at least 10 candidates for cross-validation")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if screening_n_folds < 1:
        raise ValueError("screening_n_folds must be >= 1")
    if max_finalists < 1:
        raise ValueError("max_finalists must be >= 1")

    by_molecule: Dict[str, List[int]] = {}
    for i, candidate in enumerate(candidates):
        by_molecule.setdefault(candidate.source_molecule_id, []).append(i)
    if len(by_molecule) < 2:
        raise ValueError("need candidates from at least two molecules for cross-validation")

    rng = np.random.default_rng(rng_seed)
    molecule_ids = np.asarray(sorted(by_molecule), dtype=object)
    molecule_ids = molecule_ids[rng.permutation(len(molecule_ids))]
    if n_folds == 1:
        n_validation_molecules = max(1, int(round(len(molecule_ids) * holdout_fraction)))
        validation_folds = [molecule_ids[:n_validation_molecules]]
    else:
        validation_folds = [
            fold
            for fold in np.array_split(molecule_ids, min(n_folds, len(molecule_ids)))
            if len(fold)
        ]

    def bounded(indices: List[int], seed: int) -> List[int]:
        if max_candidates_per_fold is None or len(indices) <= max_candidates_per_fold:
            return indices
        return np.random.default_rng(seed).choice(
            indices, size=max_candidates_per_fold, replace=False
        ).tolist()

    all_molecules = molecule_ids.tolist()
    folds: List[Tuple[List[int], List[int]]] = []
    for fold_no, validation_molecules in enumerate(validation_folds):
        validation_set = set(validation_molecules.tolist())
        # Preserve the seeded molecule order.  Iterating a Python set here
        # would make MiniBatchKMeans input order hash-random despite the
        # fixed numerical RNG seed.
        train_indices = [
            i for molecule_id in all_molecules if molecule_id not in validation_set
            for i in by_molecule[molecule_id]
        ]
        validation_indices = [
            i for molecule_id in validation_molecules.tolist()
            for i in by_molecule[molecule_id]
        ]
        train_indices = bounded(train_indices, rng_seed + fold_no * 2)
        validation_indices = bounded(validation_indices, rng_seed + fold_no * 2 + 1)
        if len(train_indices) >= 2 and validation_indices:
            folds.append((train_indices, validation_indices))
    if not folds:
        raise ValueError("cross-validation split left no usable folds")

    # First screen the broad capacity grid on a bounded representative sample
    # after (never before) the molecule split.  Only finalists are later
    # evaluated over every requested fold.  This is an explicit
    # successive-halving resource control, not a hidden scientific cap.
    screening_folds: List[Tuple[List[int], List[int]]] = []
    for fold_no, (train_indices, validation_indices) in enumerate(folds[:screening_n_folds]):
        if screening_max_candidates_per_fold is None:
            screen_train, screen_validation = train_indices, validation_indices
        else:
            screen_train = (
                train_indices if len(train_indices) <= screening_max_candidates_per_fold
                else np.random.default_rng(rng_seed + 10_000 + fold_no * 2).choice(
                    train_indices, size=screening_max_candidates_per_fold, replace=False
                ).tolist()
            )
            screen_validation = (
                validation_indices if len(validation_indices) <= screening_max_candidates_per_fold
                else np.random.default_rng(rng_seed + 10_000 + fold_no * 2 + 1).choice(
                    validation_indices, size=screening_max_candidates_per_fold, replace=False
                ).tolist()
            )
        if len(screen_train) >= 2 and screen_validation:
            screening_folds.append((screen_train, screen_validation))
    if not screening_folds:
        raise ValueError("dictionary-CV screen left no usable folds")

    smallest_train = min(len(train_indices) for train_indices, _ in screening_folds)
    if max_atoms is not None and max_atoms < 2:
        raise ValueError("max_atoms must be at least 2 when provided")
    if n_atoms_grid is None:
        upper = min(smallest_train - 1, max_atoms or smallest_train - 1)
        # This is a wide geometric capacity grid, deliberately not a fraction
        # cap such as the previous 25% heuristic.  A rising edge is reported.
        n_atoms_grid = sorted({
            atom_count
            for fraction in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9)
            if (atom_count := max(2, int(round(smallest_train * fraction)))) <= upper
        })

    settings = [
        (n_atoms, n_nonzero)
        for n_atoms in sorted(set(int(x) for x in n_atoms_grid))
        if n_atoms >= 2 and (max_atoms is None or n_atoms <= max_atoms)
        for n_nonzero in sorted(set(int(x) for x in n_nonzero_coefs_grid))
        if 1 <= n_nonzero <= n_atoms
    ]

    def score_settings(
        settings_to_score: Sequence[Tuple[int, int]],
        evaluation_folds: Sequence[Tuple[List[int], List[int]]],
        seed_offset: int,
    ) -> Tuple[List[Tuple[int, int, float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """Return MDL sweep plus fidelity diagnostics for fixed settings.

        Dictionary atoms depend on capacity but not OMP sparsity.  Fit each
        (capacity, fold) dictionary once and reuse it for all requested
        sparsities; the previous nested loop trained the same MiniBatchKMeans
        model three times and was the principal Step-3 CPU waste.
        """
        local_scores: Dict[str, List[float]] = {
            f"{n_atoms}:{n_nonzero}": [] for n_atoms, n_nonzero in settings_to_score
        }
        local_fidelities: Dict[str, List[float]] = {
            f"{n_atoms}:{n_nonzero}": [] for n_atoms, n_nonzero in settings_to_score
        }
        sparsities_by_capacity: Dict[int, List[int]] = {}
        for n_atoms, n_nonzero in settings_to_score:
            sparsities_by_capacity.setdefault(n_atoms, []).append(n_nonzero)

        for n_atoms, sparsities in sorted(sparsities_by_capacity.items()):
            for fold_no, (train_indices, validation_indices) in enumerate(evaluation_folds):
                train = [candidates[i] for i in train_indices]
                validation_raw = np.stack([
                    candidates[i].descriptor.to_vector() for i in validation_indices
                ]).astype(np.float32)
                # The dictionary itself is independent of n_nonzero_coefs.
                # Build its unavoidable training codes at sparsity 1, then
                # code only held-out vectors at every candidate sparsity.
                base_config = FundamentalityConfig(
                    n_atoms=n_atoms,
                    n_nonzero_coefs=1,
                    rng_seed=rng_seed + seed_offset + fold_no,
                    batch_size=batch_size,
                    n_jobs=n_jobs,
                    kmeans_n_init=1,
                )
                try:
                    fit = fit_shared_dictionary(train, base_config)
                    validation_vectors = _standardize_with_fit(
                        validation_raw, fit.feature_mean, fit.feature_scale
                    )
                except Exception:
                    continue

                for n_nonzero in sorted(set(sparsities)):
                    key = f"{n_atoms}:{n_nonzero}"
                    try:
                        validation_codes = _omp_codes(
                            validation_vectors,
                            fit.dictionary,
                            n_nonzero,
                            n_jobs=n_jobs,
                            batch_size=batch_size,
                        )
                    except Exception:
                        continue

                    # Evaluate residuals in bounded batches. Codes remain
                    # CSR, so this never expands an N-by-K coefficient matrix.
                    residual_sum_squares = 0.0
                    active_total = 0
                    fidelity_sum = 0.0
                    n_validation = len(validation_vectors)
                    for start in range(0, n_validation, batch_size):
                        stop = min(n_validation, start + batch_size)
                        code_batch = validation_codes[start:stop]
                        reconstruction = np.asarray(code_batch @ fit.dictionary)
                        residual = validation_vectors[start:stop] - reconstruction
                        residual_norms = np.linalg.norm(residual, axis=1)
                        vector_norms = np.linalg.norm(validation_vectors[start:stop], axis=1)
                        residual_sum_squares += float(np.sum(residual * residual))
                        ratio = np.zeros_like(residual_norms)
                        np.divide(residual_norms, vector_norms, out=ratio, where=vector_norms >= 1e-12)
                        fidelities_here = np.where(
                            vector_norms < 1e-12,
                            (residual_norms < 1e-12).astype(float),
                            np.maximum(0.0, 1.0 - ratio),
                        )
                        fidelity_sum += float(np.sum(fidelities_here))
                        active_total += int(np.sum(np.abs(code_batch.data) > base_config.active_atom_tol))

                    n_observations = max(1, validation_vectors.size)
                    residual_variance = max(residual_sum_squares / n_observations, 1e-12)
                    residual_bits = 0.5 * n_observations * (
                        np.log2(2.0 * np.pi * residual_variance) + 1.0
                    )
                    # BIC/MDL model cost: K*d shared centroids, active
                    # coefficient values, and sparse support indices.
                    n_train = max(2, len(train_indices))
                    dictionary_bits = 0.5 * n_atoms * fit.dictionary.shape[1] * np.log2(n_train)
                    coefficient_bits = 0.5 * active_total * np.log2(n_train)
                    support_bits = active_total * np.log2(max(2, n_atoms))
                    local_scores[key].append(float(
                        (residual_bits + dictionary_bits + coefficient_bits + support_bits)
                        / max(1, n_validation)
                    ))
                    local_fidelities[key].append(float(fidelity_sum / max(1, n_validation)))

        local_sweep = [
            (n_atoms, n_nonzero, float(np.mean(local_scores[f"{n_atoms}:{n_nonzero}"])))
            for n_atoms, n_nonzero in settings_to_score
            if local_scores[f"{n_atoms}:{n_nonzero}"]
        ]
        local_scores = {key: values for key, values in local_scores.items() if values}
        local_fidelities = {key: values for key, values in local_fidelities.items() if values}
        return local_sweep, local_scores, local_fidelities

    # Stage 1: broad, bounded screen. Entries are
    # (n_atoms, n_nonzero_coefs, mean held-out MDL bits/candidate), where
    # lower is better. Fidelity is a diagnostic, never the selection target.
    screening_sweep, screening_scores, screening_fidelities = score_settings(
        settings, screening_folds, seed_offset=0
    )
    if not screening_sweep:
        raise RuntimeError("no dictionary setting produced a usable screening fit")
    finalist_settings = [
        (n_atoms, n_nonzero)
        for n_atoms, n_nonzero, _ in sorted(screening_sweep, key=lambda row: row[2])[:max_finalists]
    ]

    # Stage 2: complete molecule-held-out CV only for the preselected
    # finalists. This preserves the test's independence while turning 72
    # expensive fits into a bounded screen plus at most max_finalists*folds.
    sweep, fold_scores_by_setting, fold_fidelities_by_setting = score_settings(
        finalist_settings, folds, seed_offset=1_000
    )

    if not sweep:
        raise RuntimeError(
            "no (n_atoms, n_nonzero_coefs) combination in the grid produced a usable fit"
        )
    best = min(sweep, key=lambda row: row[2])
    tested_sizes = sorted({row[0] for row in sweep})
    return {
        "best": (best[0], best[1]),
        "sweep": sweep,
        "fold_scores": fold_scores_by_setting,
        "fold_validation_fidelities": fold_fidelities_by_setting,
        "selection_metric": "held_out_two_part_MDL_bits_per_candidate (lower is better)",
        "score_components": (
            "Gaussian residual code + BIC dictionary/coefficient cost + sparse-support cost"
        ),
        "n_folds": len(folds),
        "screening": {
            "n_folds": len(screening_folds),
            "max_candidates_per_fold": screening_max_candidates_per_fold,
            "sweep": screening_sweep,
            "fold_scores": screening_scores,
            "fold_validation_fidelities": screening_fidelities,
            "finalists": finalist_settings,
            "best_at_largest_tested_size": (
                min(screening_sweep, key=lambda row: row[2])[0]
                == max(row[0] for row in screening_sweep)
            ),
        },
        "split_unit": "source_molecule_id",
        "max_candidates_per_fold": max_candidates_per_fold,
        "best_at_largest_tested_size": best[0] == tested_sizes[-1],
    }

