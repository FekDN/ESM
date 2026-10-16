"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
commutation.py
===============
Doc §11, §15 step 4 ("Commutation test на кандидатах-параметрах"): once a
LetterClass exists (corpus.cluster_candidates_bootstrap), decide which axes
of `InvariantDescriptor.to_vector()` are signifying (§11.2, Saussure's
*differential* sign -- a value counts only through contrast with the
alternatives that could have occupied the same slot) versus silent
(allophonic, §11.3) *within* that class. This is one level below corpus.py's
job: corpus.py decides which units (classes) exist at all; this module
decides, given a unit already found, which of its internal coordinates
actually carry functional load (§11.6 -- the two are complementary, not
duplicate, criteria).

Operational stand-in for "meaning" (§11.1 / §16.1 caveat):
    At step 4 there is no external, named property to test commutation
    against yet -- that only enters at step 5 (isostere/Mantel calibration)
    and §16 (named correlates). This is not a shortcut invented for
    convenience: it is the same level the original phonological commutation
    test operates at. Trubetzkoy's test asks whether swapping a feature
    changes *which phoneme/word* results, not what the word refers to.
    Here, analogously, "does the outcome change" == "does the nearest-class
    assignment change" against the fixed set of already-discovered class
    centroids (corpus.py's cluster output). Everything downstream of class
    identity (§16 named correlates, orphan status) is untouched by this
    module.

Two independent axes of sensitivity, deliberately not collapsed into one
score (§11.7):
    continuous_sensitivity  -- §11.3: averaged effect of perturbing axis i
                               alone, over random points drawn from across
                               the whole corpus. Answers "does this axis
                               usually matter".
    categorical_sensitivity -- §11.7: effect of perturbing axis i alone, at
                               points deliberately placed on the boundary
                               between this class and its nearest neighbor
                               class. Answers "does this axis matter exactly
                               at a class boundary" -- the case §11.7 warns
                               a purely-averaged metric systematically
                               misses.

An axis low on the first and high on the second is not a contradiction --
it is the signature of a *categorical* (class-defining) parameter rather
than a graded one (§11.7), reported here as "class_boundary_marker": a
fourth outcome the classic phonemic/allophonic binary has no name for but
the doc explicitly asks for.

Significance without a fixed magic threshold:
    Both sensitivities are scored against a matched-magnitude null: the same
    perturbation size applied along a random (isotropic) direction instead
    of the axis-aligned one. An axis counts as significant only if it moves
    the outcome *more than a generic direction of the same size would* --
    the classification threshold is this per-axis, per-class null band
    (`ConfidenceBlock.interval`), not an arbitrary constant.

§11.5 caveat carried through explicitly: every `AxisAnnotation` records
`tested_context_range` (the perturbation step actually used). This module
is only entitled to say "silent_in_tested_range", never "silent absolutely"
-- the kinetic-isotope-effect counterexample (§11.5) is exactly the case of
a parameter silent almost everywhere and decisive in one narrow context this
scaffold did not test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .corpus import LetterCandidate
from .schema import AxisAnnotation, AxisAnnotations, ConfidenceBlock, LetterClass


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class CommutationConfig:
    """Perturbation scale and trial counts. All of this is part of §11.5's
    "tested range" and ends up recorded on the output (`tested_context_range`),
    not just consumed internally -- a null result is only ever a claim about
    this specific configuration."""

    n_continuous_trials: int = 200
    n_categorical_trials: int = 200
    step_in_axis_std: float = 1.0
    # perturbation size for axis i = step_in_axis_std * (corpus-wide std of axis i);
    # scales the probe per-axis so axes with naturally larger/smaller ranges
    # (e.g. l=0 power-spectrum terms vs. high-l bispectrum terms) are probed
    # proportionally rather than with one absolute delta for all of them.
    boundary_jitter_std: float = 0.05
    # jitter of the boundary trial point along the inter-centroid hyperplane,
    # relative to the inter-centroid distance -- keeps categorical trials from
    # all landing on the exact same midpoint (§11.7 implies plural trials).
    n_null_bootstrap: int = 500
    rng_seed: int = 0


# --------------------------------------------------------------------------
# the fixed decision rule ("outcome") a commutation trial is tested against
# --------------------------------------------------------------------------


def _nearest_class(vector: np.ndarray, centroids: Sequence[np.ndarray]) -> int:
    """Index of the class whose centroid is nearest `vector` (Euclidean).
    Centroids are held fixed during a trial -- exactly like the rest of a
    minimal pair's context is held fixed in the linguistic original."""
    dists = [np.linalg.norm(vector - c) for c in centroids]
    return int(np.argmin(dists))


def _perturb_axis(vector: np.ndarray, axis: int, delta: float) -> np.ndarray:
    out = vector.copy()
    out[axis] += delta
    return out


def _random_unit_direction(dim: int, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=dim)
    return d / np.linalg.norm(d)


def _percentile_ci(values: Sequence[float], n_boot: int, rng: np.random.Generator) -> tuple:
    arr = np.asarray(values, dtype=float)
    boot_means = [
        float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# §11.3 continuous sensitivity
# --------------------------------------------------------------------------


def continuous_sensitivity(
    axis: int,
    vectors: np.ndarray,
    centroids: Sequence[np.ndarray],
    axis_std: float,
    config: CommutationConfig,
    rng: np.random.Generator,
) -> ConfidenceBlock:
    """§11.3: over random points pooled across the *whole* corpus (the doc's
    "случайные минимальные пары" is a corpus-wide average, deliberately not
    conditioned on any one class), the rate at which perturbing axis `axis`
    alone by `config.step_in_axis_std * axis_std` changes which class's
    centroid is nearest.

    `score` is this raw rate; `interval` is the 2.5/97.5 percentile band of
    the *null* rate produced by perturbing the same amount along a random
    direction instead -- so a caller compares `score` to `interval[1]`
    directly to ask "is this axis more disruptive than a generic direction
    of the same size", with no separately-chosen constant involved.
    """
    n = len(vectors)
    size = config.n_continuous_trials
    replace = size > n
    idx = rng.choice(n, size=size, replace=replace)
    delta = config.step_in_axis_std * axis_std

    axis_flips: List[int] = []
    null_flips: List[int] = []
    for i in idx:
        v = vectors[i]
        original = _nearest_class(v, centroids)

        v_axis = _perturb_axis(v, axis, delta)
        axis_flips.append(int(_nearest_class(v_axis, centroids) != original))

        direction = _random_unit_direction(len(v), rng)
        v_null = v + delta * direction
        null_flips.append(int(_nearest_class(v_null, centroids) != original))

    score = float(np.mean(axis_flips))
    boot_rng = np.random.default_rng(config.rng_seed + 1 + axis)
    lo, hi = _percentile_ci(null_flips, config.n_null_bootstrap, boot_rng)

    return ConfidenceBlock(
        n=len(axis_flips),
        method="permutation",
        score=score,
        interval=(lo, hi),
        corpus_version="n/a",
        stale=False,
    )


# --------------------------------------------------------------------------
# §11.7 categorical sensitivity
# --------------------------------------------------------------------------


def _boundary_point(
    centroid_a: np.ndarray, centroid_b: np.ndarray, jitter_std: float, rng: np.random.Generator
) -> np.ndarray:
    """A point on the Euclidean decision boundary between two centroids
    (the midpoint, which is equidistant under nearest-centroid assignment),
    jittered within the boundary hyperplane -- §11.7 calls for testing
    "deliberately constructed pairs crossing the boundary", plural, not one
    single fixed point."""
    mid = 0.5 * (centroid_a + centroid_b)
    normal = centroid_b - centroid_a
    inter_dist = np.linalg.norm(normal)
    if inter_dist < 1e-12:
        return mid
    normal = normal / inter_dist

    raw = rng.normal(size=len(mid))
    raw = raw - (raw @ normal) * normal
    raw_norm = np.linalg.norm(raw)
    if raw_norm < 1e-12:
        return mid

    tangent = raw / raw_norm
    jitter_scale = jitter_std * inter_dist * rng.normal()
    return mid + jitter_scale * tangent


def categorical_sensitivity(
    axis: int,
    class_index: int,
    centroids: Sequence[np.ndarray],
    axis_std: float,
    config: CommutationConfig,
    rng: np.random.Generator,
) -> ConfidenceBlock:
    """§11.7: effect of perturbing axis `axis` alone at points placed
    deliberately on the boundary between this class (`centroids[class_index]`)
    and its nearest neighboring class -- not at random corpus points (that is
    `continuous_sensitivity`'s job). Same matched-magnitude random-direction
    null as `continuous_sensitivity`, applied at the same boundary points, so
    the two scores are comparable and differ only in *where* trial points
    are drawn from.

    With fewer than 2 classes in the alphabet there is no boundary to test
    yet -- returns an explicit n=0 block rather than fabricating a trial,
    which `classify_axis` reads as "indeterminate", the honest answer.
    """
    if len(centroids) < 2:
        return ConfidenceBlock(
            n=0, method="permutation", score=0.0, interval=None,
            corpus_version="n/a", stale=False,
        )

    this_centroid = centroids[class_index]
    others = [c for i, c in enumerate(centroids) if i != class_index]
    nearest_other = min(others, key=lambda c: np.linalg.norm(c - this_centroid))

    delta = config.step_in_axis_std * axis_std
    axis_flips: List[int] = []
    null_flips: List[int] = []
    for _ in range(config.n_categorical_trials):
        point = _boundary_point(this_centroid, nearest_other, config.boundary_jitter_std, rng)
        original = _nearest_class(point, centroids)

        point_axis = _perturb_axis(point, axis, delta)
        axis_flips.append(int(_nearest_class(point_axis, centroids) != original))

        direction = _random_unit_direction(len(point), rng)
        point_null = point + delta * direction
        null_flips.append(int(_nearest_class(point_null, centroids) != original))

    score = float(np.mean(axis_flips))
    boot_rng = np.random.default_rng(config.rng_seed + 1001 + axis)
    lo, hi = _percentile_ci(null_flips, config.n_null_bootstrap, boot_rng)

    return ConfidenceBlock(
        n=len(axis_flips),
        method="permutation",
        score=score,
        interval=(lo, hi),
        corpus_version="n/a",
        stale=False,
    )


# --------------------------------------------------------------------------
# combining the two into one classification
# --------------------------------------------------------------------------


def classify_axis(continuous: ConfidenceBlock, categorical: ConfidenceBlock) -> str:
    """§11.7's four-way outcome, thresholded against each block's own null
    band rather than a fixed constant (see module docstring)."""
    if continuous.n == 0 or categorical.n == 0:
        return "indeterminate"

    continuous_significant = (
        continuous.interval is not None and continuous.score > continuous.interval[1]
    )
    categorical_significant = (
        categorical.interval is not None and categorical.score > categorical.interval[1]
    )

    if continuous_significant:
        return "significant"
    if categorical_significant:
        return "class_boundary_marker"
    return "silent_in_tested_range"


def _margin_above_null(block: ConfidenceBlock) -> float:
    """How far `score` sits above its own null upper bound; 0 if at or
    below it. Left unbounded/un-rescaled deliberately -- squashing it into
    [0,1] with an arbitrary constant would be exactly the kind of heuristic
    fudge the rest of this scaffold tries to avoid."""
    if block.interval is None:
        return 0.0
    return max(0.0, block.score - block.interval[1])


def result_confidence(continuous: ConfidenceBlock, categorical: ConfidenceBlock) -> Optional[ConfidenceBlock]:
    """Confidence in the classification itself, not in either raw score
    alone (doc's `axis_annotations.result_confidence`). `n` is the smaller
    of the two trial counts, since the discrete label depended on comparing
    both; `score` is the larger of the two above-null margins -- how far
    the *deciding* test cleared its own null band."""
    if continuous.n == 0 or categorical.n == 0:
        return None
    n = min(continuous.n, categorical.n)
    margin = max(_margin_above_null(continuous), _margin_above_null(categorical))
    return ConfidenceBlock(
        n=n, method="permutation", score=margin, interval=None,
        corpus_version="n/a", stale=False,
    )


# --------------------------------------------------------------------------
# top-level entry point
# --------------------------------------------------------------------------


def build_axis_annotations(
    letter_class: LetterClass,
    all_classes: Sequence[LetterClass],
    candidates: Optional[Sequence[LetterCandidate]] = None,
    config: Optional[CommutationConfig] = None,
    *,
    vectors: Optional[np.ndarray] = None,
    continuous_cache: Optional[Dict[int, ConfidenceBlock]] = None,
) -> AxisAnnotations:
    """Doc §11 / §15 step 4, full per-class output: run both sensitivity
    tests on every axis of the descriptor vector and classify each one.

    `all_classes` must include `letter_class` itself (its centroid is looked
    up by `class_id`); it should also include the rest of the currently
    discovered alphabet, since `categorical_sensitivity` needs a neighboring
    class to define a boundary against. With only one class in the whole
    alphabet, every axis correctly comes back "indeterminate" on the
    categorical side -- there is no boundary yet, which is the honest
    answer, not a gap to paper over.

    ``candidates`` supplies the corpus-wide pool ``continuous_sensitivity``
    draws its random points from, and sets the per-axis perturbation scale.
    A caller that evaluates several classes over the same candidate partition
    may instead pass the pre-stacked ``vectors`` and a mutable
    ``continuous_cache``.  Continuous sensitivity is a property of an axis
    and the fixed comparison alphabet, not of the particular class being
    annotated; recomputing it for every class merely repeats the same random
    experiment and was the largest avoidable cost in the master runner.
    """
    config = config or CommutationConfig()
    rng = np.random.default_rng(config.rng_seed)

    class_ids = [c.class_id for c in all_classes]
    if letter_class.class_id not in class_ids:
        raise ValueError(
            f"letter_class.class_id={letter_class.class_id!r} not found in all_classes"
        )
    class_index = class_ids.index(letter_class.class_id)
    centroids = [c.invariant_centroid for c in all_classes]

    if vectors is None:
        if not candidates:
            raise ValueError("need at least one candidate to set axis perturbation scale")
        vectors = np.stack([c.descriptor.to_vector() for c in candidates])
    if len(vectors) == 0:
        raise ValueError("need at least one vector to set axis perturbation scale")
    axis_stds = vectors.std(axis=0)
    dim = vectors.shape[1]

    axes: List[AxisAnnotation] = []
    for axis in range(dim):
        axis_std = float(axis_stds[axis])
        if axis_std == 0.0:
            # constant across the entire corpus -- no perturbation scale is
            # even definable; report honestly instead of dividing by zero
            axes.append(
                AxisAnnotation(
                    axis_index=axis,
                    continuous_sensitivity=None,
                    categorical_sensitivity=None,
                    commutation_test_result="indeterminate",
                    result_confidence=None,
                    tested_context_range=None,
                )
            )
            continue

        if continuous_cache is not None and axis in continuous_cache:
            cont = continuous_cache[axis]
        else:
            cont = continuous_sensitivity(axis, vectors, centroids, axis_std, config, rng)
            if continuous_cache is not None:
                continuous_cache[axis] = cont
        cat = categorical_sensitivity(axis, class_index, centroids, axis_std, config, rng)
        result = classify_axis(cont, cat)
        conf = result_confidence(cont, cat)
        delta = config.step_in_axis_std * axis_std

        axes.append(
            AxisAnnotation(
                axis_index=axis,
                continuous_sensitivity=cont,
                categorical_sensitivity=cat,
                commutation_test_result=result,
                result_confidence=conf,
                tested_context_range=(-delta, delta),
            )
        )

    return AxisAnnotations(class_id=letter_class.class_id, axes=axes)
