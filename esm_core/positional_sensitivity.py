"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
positional_sensitivity.py
==========================
Doc §17 ("Устойчивость смысла к позднему возмущению") -- explicitly a
LATER analytical layer, applicable only once part of the alphabet is
already decoded (§17: "определение «позиции в предложении» и «порядка
добавления слов»... само требует, чтобы единицы алфавита уже были
выделены"). This module therefore consumes already-discovered
`schema.LetterClass` identities (from `corpus.py`) as the alphabet its
"positions" are positions OF, rather than discovering anything new about
what a letter is.

§17.4's explicit method: for a matched series of molecules differing in
the ORDER or COMPOSITION of accumulated substituents (build steps), measure
not the raw property change from the last substitution, but that change
NORMALIZED by how much structural context has already accumulated --
distinguishing three outcomes the doc keeps conceptually separate:

    - canalized_background (§17.1/17.2): typical, low normalized
      sensitivity -- most positions, once enough prior context exists, are
      robust to further perturbation (Waddington canalization, funnel
      theory, §9.2).
    - hot_spot (§17.1): significantly ABOVE the matched null at a late,
      otherwise-canalized position -- alanine-scanning/activity-cliff
      style: a small minority of positions carry almost all the
      information load (Clackson & Wells 1995).
    - bifurcation_point (§17.3): a position where the effect is large NOT
      because the position is special in an already-settled structure, but
      because the system genuinely has not yet resolved between two (or
      more) competing outcomes at that point in the build order (garden-
      path sentences; SN1/SN2 mechanism forks). Operationally distinguished
      from a hot spot by whether the DOWNSTREAM outcome distribution at
      that position is itself multimodal (bifurcation) versus a smooth
      unimodal shift (hot spot) -- §17.3's own distinction, made testable.

Pipeline:
    1. `BuildTrajectory` -- one matched series: an ordered sequence of
       (`LetterClass.class_id` added, outcome measured after that step).
       Caller-supplied, since this scaffold has no source of matched
       build-order series itself (same honest-scaffold posture the rest of
       this codebase takes toward data it does not produce -- e.g.
       `environment_response.py`'s caller-supplied `EnvironmentResponseSample`s).
    2. `accumulated_context_volume` -- §17.2's canalization proxy: number
       of prior steps already committed (the simplest, doc-literal reading
       of "объём уже накопленного структурного контекста"); a caller with
       a richer physical proxy (e.g. cumulative free-energy commitment
       along a funnel-theory landscape) can substitute their own via
       `context_volume_fn`.
    3. `positional_sensitivity_profile` -- the §17.4 entry point: per
       position, the mean |outcome change| across every matched-pair
       trajectory that shares the same prefix up to that position, divided
       by that position's accumulated context volume, tested against a
       matched null of position-label permutation (same "beat a matched
       null" idiom as commutation.py's random-direction null).
    4. `classify_position` -- folds the significance test plus a downstream
       multimodality check (Hartigan-style: compare 1- vs 2-component
       Gaussian-mixture BIC on the position's downstream outcome values)
       into the three-way §17.1 vs §17.3 verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .schema import ConfidenceBlock, PositionSensitivity, PositionalSensitivityProfile


# --------------------------------------------------------------------------
# input data
# --------------------------------------------------------------------------


@dataclass
class BuildStep:
    """One step of a matched-series build order: which letter class was
    added, and the outcome measured immediately after this step (e.g.
    binding affinity, activation energy -- any scalar property tracked
    along the accumulation)."""

    class_id_added: str
    outcome_after_step: float


@dataclass
class BuildTrajectory:
    """One molecule's full accumulation history, e.g. a scaffold with
    substituents added one at a time in a recorded order. `series_id`
    groups trajectories that are meant to be compared position-by-position
    (a "matched series" in the MMPA/§15-step-5 sense -- differing only in
    which class was added at one or more positions, not in overall
    length/scaffold)."""

    trajectory_id: str
    series_id: str
    steps: List[BuildStep]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class PositionalSensitivityConfig:
    """n_null_permutations -- trials for the position-label permutation
    null. min_trajectories_per_position -- below this many trajectories
    contributing an outcome-change observation at a given position, that
    position is reported "insufficient_data" rather than a statistic
    computed on too few points (same idiom as
    `environment_response.py`'s `len(vectors) < 3` guard).
    multimodality_alpha -- p-value threshold below which the downstream
    outcome distribution at a position is called multimodal (§17.3).
    rng_seed -- top-level seed.
    """

    n_null_permutations: int = 500
    min_trajectories_per_position: int = 4
    multimodality_alpha: float = 0.05
    rng_seed: int = 0


# --------------------------------------------------------------------------
# step 2: accumulated context volume
# --------------------------------------------------------------------------


def accumulated_context_volume(
    trajectory: BuildTrajectory,
    position: int,
    context_volume_fn: Optional[Callable[[BuildTrajectory, int], float]] = None,
) -> float:
    """§17.2 canalization proxy for one trajectory at `position` (0-indexed
    step about to be perturbed): default is simply the number of steps
    already committed before this one (`position`), floored at 1 so
    position 0 (no prior context at all -- the doc's own §17.3 case of
    "context still objectively ambiguous") does not divide by zero
    downstream. A caller with a physically richer canalization measure
    (e.g. cumulative funnel depth) supplies `context_volume_fn(trajectory,
    position) -> float` instead.
    """
    if context_volume_fn is not None:
        return float(context_volume_fn(trajectory, position))
    return float(max(1, position))


# --------------------------------------------------------------------------
# step 3: per-position normalized sensitivity + permutation null
# --------------------------------------------------------------------------


def _outcome_changes_at_position(
    trajectories: Sequence[BuildTrajectory], position: int
) -> List[Tuple[float, float]]:
    """(|outcome change|, context_volume) pairs contributed by every
    trajectory long enough to have a step at `position` and a prior step
    to compare against (position >= 1)."""
    out = []
    for traj in trajectories:
        if position < 1 or position >= len(traj.steps):
            continue
        delta = abs(traj.steps[position].outcome_after_step - traj.steps[position - 1].outcome_after_step)
        out.append(delta)
    return out


def _normalized_sensitivity_stat(
    trajectories: Sequence[BuildTrajectory],
    position: int,
    context_volume_fn: Optional[Callable[[BuildTrajectory, int], float]],
) -> Optional[float]:
    deltas = []
    for traj in trajectories:
        if position < 1 or position >= len(traj.steps):
            continue
        delta = abs(traj.steps[position].outcome_after_step - traj.steps[position - 1].outcome_after_step)
        vol = accumulated_context_volume(traj, position, context_volume_fn)
        deltas.append(delta / vol if vol > 0 else delta)
    if not deltas:
        return None
    return float(np.mean(deltas))


def position_sensitivity(
    trajectories: Sequence[BuildTrajectory],
    position: int,
    config: Optional[PositionalSensitivityConfig] = None,
    context_volume_fn: Optional[Callable[[BuildTrajectory, int], float]] = None,
    corpus_version: str = "unversioned",
) -> PositionSensitivity:
    """§17.4: normalized sensitivity at one position, tested against a
    matched null built by permuting WHICH position each trajectory's
    outcome-change observation is attributed to (pooling every
    (position, |delta|/volume) observation across the whole series,
    reshuffling the position labels, and recomputing the mean at this
    position under the shuffle) -- exactly the "beat a matched null of the
    same magnitude/shape, not an arbitrary constant" idiom used everywhere
    else in this codebase (commutation.py's random-direction null,
    fundamentality.py's label-permutation null).
    """
    config = config or PositionalSensitivityConfig()

    contributing = [
        t for t in trajectories if position >= 1 and position < len(t.steps)
    ]
    if len(contributing) < config.min_trajectories_per_position:
        return PositionSensitivity(
            position_index=position,
            normalized_sensitivity=ConfidenceBlock(
                n=len(contributing), method="permutation", score=0.0, corpus_version=corpus_version
            ),
            classification="insufficient_data",
        )

    observed = _normalized_sensitivity_stat(contributing, position, context_volume_fn)
    if observed is None:
        return PositionSensitivity(
            position_index=position,
            normalized_sensitivity=ConfidenceBlock(
                n=0, method="permutation", score=0.0, corpus_version=corpus_version
            ),
            classification="insufficient_data",
        )

    # Pool (delta/volume) observations from every position with enough data,
    # so the null reflects the overall spread of normalized sensitivities
    # across the whole build order, not just this position's own values.
    max_len = max(len(t.steps) for t in trajectories)
    pooled: List[float] = []
    for p in range(1, max_len):
        deltas_p = []
        for t in trajectories:
            if p < len(t.steps):
                vol = accumulated_context_volume(t, p, context_volume_fn)
                d = abs(t.steps[p].outcome_after_step - t.steps[p - 1].outcome_after_step)
                deltas_p.append(d / vol if vol > 0 else d)
        pooled.extend(deltas_p)

    rng = np.random.default_rng(config.rng_seed + position)
    pooled_arr = np.array(pooled)
    n_at_position = len(contributing)
    null_means = []
    for _ in range(config.n_null_permutations):
        sample = rng.choice(pooled_arr, size=n_at_position, replace=len(pooled_arr) < n_at_position)
        null_means.append(float(np.mean(sample)))
    lo, hi = np.percentile(null_means, [2.5, 97.5])

    return PositionSensitivity(
        position_index=position,
        normalized_sensitivity=ConfidenceBlock(
            n=n_at_position,
            method="permutation",
            score=observed,
            interval=(float(lo), float(hi)),
            corpus_version=corpus_version,
            stale=False,
        ),
        # Placeholder, not a verdict -- real classification is
        # `classify_position`'s job. Distinct from the true
        # "insufficient_data" early-returns above (which have
        # normalized_sensitivity.n == 0): this branch always has
        # n = n_at_position >= config.min_trajectories_per_position.
        classification="canalized_background",
    )


# --------------------------------------------------------------------------
# step 4: hot spot vs bifurcation point vs background
# --------------------------------------------------------------------------


def _gmm_bimodality_p(values: np.ndarray, rng: np.random.Generator, n_boot: int = 200) -> Optional[float]:
    """§17.3 downstream-multimodality test: BIC(2-component GMM) vs
    BIC(1-component) on the raw downstream outcome VALUES (not deltas) at
    this position, with a parametric-bootstrap p-value -- resample from the
    fitted 1-component null, refit both models, and ask how often the null
    produces a BIC improvement at least as large as observed (same
    "compare an observed statistic to its own null-generated distribution"
    idiom used throughout this codebase, applied here to a bimodality test
    in the absence of a dedicated Hartigan-dip-test dependency).

    Returns `None` if there are too few points (< 6) to fit two components
    meaningfully.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if len(values) < 6:
        return None
    from sklearn.mixture import GaussianMixture

    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(values)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(values)
    observed_gain = gmm1.bic(values) - gmm2.bic(values)  # positive = 2-component better

    mu, sigma = float(gmm1.means_[0, 0]), float(np.sqrt(gmm1.covariances_[0, 0, 0]))
    sigma = max(sigma, 1e-9)
    gains = []
    for _ in range(n_boot):
        sample = rng.normal(mu, sigma, size=len(values)).reshape(-1, 1)
        g1 = GaussianMixture(n_components=1, random_state=0).fit(sample)
        g2 = GaussianMixture(n_components=2, random_state=0).fit(sample)
        gains.append(g1.bic(sample) - g2.bic(sample))
    gains = np.array(gains)
    p = float(np.mean(gains >= observed_gain))
    return p


def classify_position(
    sensitivity: PositionSensitivity,
    trajectories: Sequence[BuildTrajectory],
    position: int,
    config: Optional[PositionalSensitivityConfig] = None,
) -> PositionSensitivity:
    """§17.1 vs §17.3: significant normalized sensitivity is necessary but
    not sufficient to call a position a "hot spot" -- it is a hot spot only
    if the downstream outcome distribution at this position stays
    essentially unimodal (a consistent, large shift); it is instead a
    "bifurcation_point" if that distribution is itself multimodal (the
    system genuinely forks at this point, per §17.3, rather than
    consistently landing on one shifted value)."""
    config = config or PositionalSensitivityConfig()
    block = sensitivity.normalized_sensitivity
    if block.n == 0:
        sensitivity.classification = "insufficient_data"
        return sensitivity

    significant = block.interval is not None and block.score > block.interval[1]

    if not significant:
        sensitivity.classification = "canalized_background"
        return sensitivity

    downstream = [t.steps[position].outcome_after_step for t in trajectories if position < len(t.steps)]
    rng = np.random.default_rng(config.rng_seed + 10_000 + position)
    p = _gmm_bimodality_p(np.array(downstream), rng)
    sensitivity.downstream_multimodality_p = p

    if p is not None and p < config.multimodality_alpha:
        sensitivity.classification = "bifurcation_point"
    else:
        sensitivity.classification = "hot_spot"
    return sensitivity


# --------------------------------------------------------------------------
# top-level entry point
# --------------------------------------------------------------------------


def positional_sensitivity_profile(
    trajectories: Sequence[BuildTrajectory],
    series_id: Optional[str] = None,
    config: Optional[PositionalSensitivityConfig] = None,
    context_volume_fn: Optional[Callable[[BuildTrajectory, int], float]] = None,
    corpus_version: str = "unversioned",
) -> PositionalSensitivityProfile:
    """§17.4, full pipeline for one matched series: `trajectories` should
    all share `series_id` (filtered here if a mix is passed, so a caller
    can hand this function a whole corpus of series and call it once per
    `series_id` without pre-filtering by hand).

    Positions run from 1 to the shortest trajectory's length - 1 (position
    0 has no prior step to measure a change against, per
    `accumulated_context_volume`'s docstring -- it is the doc's own §17.3
    "context still objectively ambiguous" case in the extreme, not
    measurable by this delta-based method at all).
    """
    config = config or PositionalSensitivityConfig()
    if series_id is not None:
        trajectories = [t for t in trajectories if t.series_id == series_id]
    else:
        series_ids = {t.series_id for t in trajectories}
        if len(series_ids) > 1:
            raise ValueError(
                "trajectories span multiple series_id values; pass series_id "
                "explicitly to select one, or call this once per series"
            )
        series_id = next(iter(series_ids), "unspecified")

    if not trajectories:
        return PositionalSensitivityProfile(series_id=series_id, positions=[], canalization_curve=[])

    min_len = min(len(t.steps) for t in trajectories)
    positions: List[PositionSensitivity] = []
    canalization_curve: List[float] = []

    for pos in range(1, min_len):
        raw = position_sensitivity(trajectories, pos, config, context_volume_fn, corpus_version)
        classified = classify_position(raw, trajectories, pos, config)
        positions.append(classified)
        mean_vol = float(
            np.mean([accumulated_context_volume(t, pos, context_volume_fn) for t in trajectories])
        )
        canalization_curve.append(mean_vol)

    return PositionalSensitivityProfile(
        series_id=series_id, positions=positions, canalization_curve=canalization_curve
    )
