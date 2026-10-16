"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
expectation_profiles.py
========================
Doc §19 ("Целевые данные: ожидаемые профили как критерий фальсификации").
Direct continuation of §16: a validation claim without a pre-declared
failure profile "risks accepting any result as confirmation" (§19 intro).
Every builder below states BOTH what counts as support and what counts as
failure before looking at data, and reports `schema.ExpectationCheck.verdict`
accordingly -- never a bare pass/fail collapsed post hoc.

Representation choice: the doc's pseudocode (§19.2-§19.9) nests
`ExpectationCheck` fields inside named profile blocks
(`ExpectedLetterCorpusProfile`, `ExpectedWordCorpusProfile`, ...). Rather
than adding eight more rigid dataclasses to schema.py (whose fields would
just be `ExpectationCheck` again, keyed by the doc's own field names), each
`build_*_profile` function here returns a `Dict[str, ExpectationCheck]`
keyed EXACTLY by the doc's field names (`"alphabet_size_curve"`,
`"cluster_stability_distribution"`, ...) -- schema-faithful without a
second, redundant layer of near-empty dataclasses.

Data-source honesty, matching the rest of this scaffold: every check below
that CAN be computed from objects the existing pipeline already produces
(`schema.LetterClass`, `schema.CooccurrenceEntry`, `schema.TransitionModel`,
`schema.EnvironmentResponseProfile`, `schema.SemanticStatus`, this
scaffold's own `paradigm.py`/`distributional_test.py` outputs) is wired
directly to those objects. Checks whose doc-mandated inputs belong to a
later phase this scaffold does not populate (§19.3's `word_class`,
§19.6's sentence/execution data) take plain arrays instead, with a
docstring stating exactly what the caller must supply -- the same posture
`environment_response.py` takes toward `EnvironmentResponseSample` and
`schema.py`'s module docstring takes toward "correctly shaped, not
populated by this scaffold" containers. §19.7 (alphabet discovery) is the
one profile the doc calls out as logically prior to the rest (§19.10) and
gets the most self-contained implementation here since it only needs
repeated `corpus.cluster_candidates_bootstrap` runs, already fully
supported.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .corpus import LetterCandidate, class_members
from .fundamentality import DictionaryFit, FundamentalityConfig, fit_shared_dictionary
from .schema import (
    ConfidenceBlock,
    CooccurrenceEntry,
    EnvironmentResponseProfile,
    ExpectationCheck,
    ForbiddenMapEntry,
    LetterClass,
    Paradigm,
    SemanticStatus,
    SynonymRelation,
    TransitionModel,
)


# ==========================================================================
# generic statistical primitives, shared across profiles (§19.1)
# ==========================================================================


def _confidence_from_bootstrap(
    values: Sequence[float], n_boot: int, rng: np.random.Generator, corpus_version: str
) -> ConfidenceBlock:
    """§19.1's `confidence` field: how reliably `observed_shape`'s summary
    statistic (usually its mean) was itself measured, independent of
    whether it matches the theory's prediction."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return ConfidenceBlock(n=0, method="bootstrap", score=0.0, corpus_version=corpus_version)
    boot = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return ConfidenceBlock(
        n=len(arr), method="bootstrap", score=float(np.mean(arr)),
        interval=(float(lo), float(hi)), corpus_version=corpus_version, stale=False,
    )


def _bimodality_check(
    values: Sequence[float],
    expected_shape: str,
    corpus_version: str = "unversioned",
    n_boot: int = 200,
    rng_seed: int = 0,
) -> ExpectationCheck:
    """dip_test proxy (no dedicated Hartigan-dip dependency in this
    scaffold's requirements): compares BIC of a 1-component vs 2-component
    1D Gaussian mixture, with a parametric-bootstrap p-value from the
    1-component null (same construction as
    `positional_sensitivity._gmm_bimodality_p`, reused here for a
    corpus-level distribution instead of one position's downstream
    outcomes). `p_value` low => reject "unimodal continuum", i.e. supports
    a genuine bimodal split when `expected_shape` calls for one.
    """
    from sklearn.mixture import GaussianMixture

    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(rng_seed)
    if len(arr) < 6:
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape="too few points to test (<6)",
            test_method="dip_test", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(arr)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(arr)
    observed_gain = gmm1.bic(arr) - gmm2.bic(arr)

    mu, sigma = float(gmm1.means_[0, 0]), max(float(np.sqrt(gmm1.covariances_[0, 0, 0])), 1e-9)
    gains = []
    for _ in range(n_boot):
        sample = rng.normal(mu, sigma, size=len(arr)).reshape(-1, 1)
        g1 = GaussianMixture(n_components=1, random_state=0).fit(sample)
        g2 = GaussianMixture(n_components=2, random_state=0).fit(sample)
        gains.append(g1.bic(sample) - g2.bic(sample))
    p = float(np.mean(np.array(gains) >= observed_gain))

    conf = _confidence_from_bootstrap(values, n_boot, rng, corpus_version)
    verdict = "supports_theory" if p < 0.05 and observed_gain > 0 else "against_theory"
    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"BIC gain (2-comp over 1-comp)={observed_gain:.2f}, means~{gmm2.means_.ravel()}",
        test_method="dip_test", p_value=p, confidence=conf,
        corpus_version=corpus_version, verdict=verdict,
    )


def _powerlaw_ks_check(
    frequencies: Sequence[float],
    expected_shape: str,
    corpus_version: str = "unversioned",
    rng_seed: int = 0,
) -> ExpectationCheck:
    """§19.2 `occurrence_frequency`: fit a Zipfian rank-frequency slope via
    log-log linear regression, then KS-test the empirical frequency
    distribution against the fitted discrete power law. Failure mode
    (§19.2): a near-zero slope (flat log-log) reads as uniform, not
    Zipfian."""
    from scipy.stats import kstest, zipf

    freqs = np.sort(np.asarray(frequencies, dtype=float))[::-1]
    freqs = freqs[freqs > 0]
    if len(freqs) < 5:
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape="too few classes to test (<5)",
            test_method="KS", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    ranks = np.arange(1, len(freqs) + 1)
    slope, intercept = np.polyfit(np.log(ranks), np.log(freqs), 1)
    r2 = float(np.corrcoef(np.log(ranks), np.log(freqs))[0, 1] ** 2)

    a = max(1.01, -slope + 1.0)  # zipf shape parameter s.t. rank^-slope ~ zipf pmf tail
    try:
        stat, p = kstest(freqs / freqs.sum(), zipf(a).pmf, args=())
    except Exception:
        stat, p = float("nan"), None

    verdict = "supports_theory" if slope < -0.3 and r2 > 0.5 else "against_theory"
    rng = np.random.default_rng(rng_seed)
    conf = _confidence_from_bootstrap(freqs.tolist(), 200, rng, corpus_version)
    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"log-log slope={slope:.2f}, R^2={r2:.2f}",
        test_method="KS", p_value=p, confidence=conf,
        corpus_version=corpus_version, verdict=verdict,
    )


def _sublinear_vs_linear_check(
    corpus_sizes: Sequence[float],
    counts: Sequence[float],
    expected_shape: str,
    corpus_version: str = "unversioned",
) -> ExpectationCheck:
    """§19.2 `alphabet_size_curve` / §19.7's growth checks: compare a
    saturating power-law fit (count ~ a * size^b, b < 1) against a linear
    fit (count ~ a * size) by AIC on log-count residuals. Lower AIC for
    the sublinear fit AND a fitted exponent visibly below 1 both required
    for "supports_theory" -- a power-law fit with b close to 1 is
    observationally indistinguishable from linear and should not be read
    as saturation.
    """
    x = np.asarray(corpus_sizes, dtype=float)
    y = np.asarray(counts, dtype=float)
    if len(x) < 4 or np.any(y <= 0) or np.any(x <= 0):
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape="too few / invalid snapshots to fit (<4)",
            test_method="AIC_comparison", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    log_x, log_y = np.log(x), np.log(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    resid_power = log_y - (log_a + b * log_x)
    rss_power = float(np.sum(resid_power**2))

    slope_lin, intercept_lin = np.polyfit(x, y, 1)
    resid_lin = y - (slope_lin * x + intercept_lin)
    rss_lin_in_logspace = float(np.sum((np.log(np.clip(slope_lin * x + intercept_lin, 1e-9, None)) - log_y) ** 2))

    k = 2  # both models have 2 free parameters
    n = len(x)
    aic_power = n * np.log(rss_power / n + 1e-12) + 2 * k
    aic_linear = n * np.log(rss_lin_in_logspace / n + 1e-12) + 2 * k

    verdict = "supports_theory" if (aic_power < aic_linear and b < 0.85) else "against_theory"
    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"power-law exponent b={b:.2f} (AIC={aic_power:.1f} vs linear AIC={aic_linear:.1f})",
        test_method="AIC_comparison", p_value=None,
        confidence=ConfidenceBlock(n=n, method="analytic_CI", score=float(b), corpus_version=corpus_version),
        corpus_version=corpus_version, verdict=verdict,
    )


def _uniform_chi_square_check(
    counts: Sequence[float],
    expected_shape: str,
    corpus_version: str = "unversioned",
    reject_uniform: bool = True,
) -> ExpectationCheck:
    """§19.4: tests whether `counts` (e.g. node degrees, transition
    out-degrees) deviate from a uniform/equal-frequency null.
    `reject_uniform=True` (default) means the THEORY predicts a
    non-uniform, structured shape (hub/periphery, sparse transitions) --
    "supports_theory" iff chi-square rejects uniformity. Set False for a
    check where the theory instead predicts near-uniformity."""
    from scipy.stats import chisquare

    arr = np.asarray(counts, dtype=float)
    arr = arr[arr >= 0]
    if len(arr) < 3 or arr.sum() <= 0:
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape="too few bins to test (<3)",
            test_method="chi_square", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    expected = np.full(len(arr), arr.sum() / len(arr))
    stat, p = chisquare(arr, expected)
    non_uniform = p < 0.05
    verdict = "supports_theory" if (non_uniform == reject_uniform) else "against_theory"
    gini = float(np.sum(np.abs(arr[:, None] - arr[None, :]))) / (2 * len(arr) * arr.sum()) if arr.sum() else 0.0
    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"gini={gini:.2f} over {len(arr)} bins",
        test_method="chi_square", p_value=float(p), confidence=None,
        corpus_version=corpus_version, verdict=verdict,
    )


def _range_check(
    value: float,
    low: float,
    high: float,
    expected_shape: str,
    n: int,
    corpus_version: str = "unversioned",
) -> ExpectationCheck:
    """§19.5/§19.4's recurring "content but not extreme" pattern
    (named_correlates coverage, forbidden_map coverage): a healthy result
    lies strictly BETWEEN two suspicious extremes, not merely above/below
    one threshold. No p-value -- this is a declared-band check, not a
    significance test against a null, so `test_method="range_check"` and
    `p_value=None` by construction.
    """
    verdict = "supports_theory" if low < value < high else "against_theory"
    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"value={value:.3f} (band checked: {low:.2f}-{high:.2f})",
        test_method="range_check", p_value=None,
        confidence=ConfidenceBlock(n=n, method="analytic_CI", score=value, corpus_version=corpus_version),
        corpus_version=corpus_version, verdict=verdict,
    )


def _trend_check(
    ordered_group_success_rates: Sequence[Tuple[str, float, int]],
    expected_shape: str,
    corpus_version: str = "unversioned",
    n_permutations: int = 2000,
    rng_seed: int = 0,
) -> ExpectationCheck:
    """§19.6 `execution_success_rate_by_grammaticality_status`: tests that
    success rate STRICTLY decreases across `ordered_group_success_rates`
    (already given in the theory-predicted order, e.g. attested >
    combinatorially_valid_untested > physically_forbidden), via a
    label-permutation null on the monotone-trend statistic (Spearman rho
    between group rank and success rate)."""
    from scipy.stats import spearmanr

    if len(ordered_group_success_rates) < 2:
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape="fewer than 2 groups",
            test_method="trend_test", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    ranks = np.arange(len(ordered_group_success_rates))
    rates = np.array([r for _, r, _ in ordered_group_success_rates])
    rho, _ = spearmanr(ranks, rates)

    rng = np.random.default_rng(rng_seed)
    null_rhos = []
    for _ in range(n_permutations):
        perm = rng.permutation(rates)
        r, _ = spearmanr(ranks, perm)
        null_rhos.append(r if not np.isnan(r) else 0.0)
    p = float(np.mean(np.array(null_rhos) <= rho)) if rho <= 0 else float(np.mean(np.array(null_rhos) >= rho))
    # strictly monotone decreasing across the doc-declared order == rho <= -1 + eps
    strictly_decreasing = all(
        rates[i] > rates[i + 1] for i in range(len(rates) - 1)
    )
    verdict = "supports_theory" if strictly_decreasing and rho < -0.5 else "against_theory"
    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"rates={[round(r,3) for _,r,_ in ordered_group_success_rates]}, rho={rho:.2f}",
        test_method="trend_test", p_value=p, confidence=None,
        corpus_version=corpus_version, verdict=verdict,
    )


def _correlation_check(
    x: Sequence[float],
    y: Sequence[float],
    expected_sign: str,
    expected_shape: str,
    corpus_version: str = "unversioned",
    n_permutations: int = 2000,
    equivalence_margin: float = 0.2,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> ExpectationCheck:
    """Spearman correlation with a permutation p-value, used wherever §19
    asks for a signed relationship (e.g. §19.8's graph_distance <->
    invariant_distance / invariant_distance <-> function_distance pattern).
    `expected_sign` in {"positive", "negative", "none"}.

    CORRECTNESS NOTE (fixed): `expected_sign="none"` used to read `p >=
    0.05` (failure to reject the null of zero correlation) directly as
    "supports_theory". That is the classic absence-of-evidence /
    evidence-of-absence error: with the `len(xs) < 4` floor this function
    otherwise allows, a genuinely correlated pair easily fails to reach
    significance just from low power, and would have been counted as
    confirming "no relationship" rather than as an inconclusive test. This
    now runs an actual equivalence test instead -- the same idiom
    `semantics.orphan_confidence_from_tests` already uses elsewhere in this
    codebase (check a bootstrap CI's bound against a threshold, not a bare
    p-value): resample pairs with replacement, recompute rho each time, and
    require the WHOLE bootstrap CI to sit inside
    [-equivalence_margin, +equivalence_margin]. A wide CI (too little data
    or a genuinely uncertain relationship) no longer defaults to
    "supports_theory" -- it fails the margin check and returns
    "inconclusive", the same way `semantics.py`'s equivalent orphan check
    does when its CI is too wide to be conclusive. The `len(xs) < 4` floor
    below is a hard minimum for computing rho at all, not a claim that 4
    points give an equivalence test any real power -- see `_range_check`-style
    callers upstream (§19.8) for the doc's own, larger minimum-pairs
    thresholds gating whether this function is even called.
    """
    from scipy.stats import spearmanr

    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xs) < 4:
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape="too few pairs (<4)",
            test_method="correlation", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    rho, _ = spearmanr(xs, ys)
    rho = 0.0 if np.isnan(rho) else float(rho)
    rng = np.random.default_rng(rng_seed)

    if expected_sign in ("positive", "negative"):
        null = []
        for _ in range(n_permutations):
            perm = rng.permutation(ys)
            r, _ = spearmanr(xs, perm)
            null.append(0.0 if np.isnan(r) else r)
        null = np.array(null)
        p = float(np.mean(np.abs(null) >= abs(rho)))
        ok = (rho > equivalence_margin and p < 0.05) if expected_sign == "positive" else (
            rho < -equivalence_margin and p < 0.05
        )
        return ExpectationCheck(
            expected_shape=expected_shape, observed_shape=f"spearman rho={rho:.2f}",
            test_method="correlation", p_value=p, confidence=None,
            corpus_version=corpus_version, verdict="supports_theory" if ok else "against_theory",
        )

    # expected_sign == "none": equivalence test, not a significance test.
    n = len(xs)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(xs[idx], ys[idx])
        boot_rhos.append(0.0 if np.isnan(r) else r)
    lo, hi = np.percentile(boot_rhos, [2.5, 97.5])

    if n < 8:
        verdict = "inconclusive"
    elif -equivalence_margin <= lo and hi <= equivalence_margin:
        verdict = "supports_theory"
    elif lo > equivalence_margin or hi < -equivalence_margin:
        # CI clears the margin on the wrong side -- a real correlation, not noise.
        verdict = "against_theory"
    else:
        # CI straddles the margin boundary -- genuinely undecided, not a pass.
        verdict = "inconclusive"

    return ExpectationCheck(
        expected_shape=expected_shape,
        observed_shape=f"spearman rho={rho:.2f}, bootstrap 95% CI=({lo:.2f}, {hi:.2f}), equivalence margin=±{equivalence_margin:.2f}",
        test_method="correlation", p_value=None,
        confidence=ConfidenceBlock(n=n, method="bootstrap", score=rho, interval=(float(lo), float(hi)), corpus_version=corpus_version),
        corpus_version=corpus_version, verdict=verdict,
    )


# ==========================================================================
# §19.2 ExpectedLetterCorpusProfile
# ==========================================================================


def build_letter_corpus_profile(
    corpus_size_snapshots: Sequence[int],
    n_classes_snapshots: Sequence[int],
    letter_classes: Sequence[LetterClass],
    class_occurrence_counts: Sequence[int],
    intrinsic_dimension: Optional[ConfidenceBlock],
    raw_dimension: Optional[int],
    within_class_entropies: Sequence[float],
    between_class_entropy: float,
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.2. `corpus_size_snapshots`/`n_classes_snapshots` -- paired
    (molecule count, alphabet size) at successive corpus checkpoints
    (e.g. from repeated `pipeline.decode_alphabet` runs -- see
    `build_alphabet_discovery_profile` for the fuller §19.7 treatment of
    the same underlying data). `intrinsic_dimension` is
    `dimensionality.levina_bickel_intrinsic_dimension`'s own output,
    reused directly rather than recomputed (staleness-avoidance, same
    reasoning as `schema.HypothesisStatus.obligatory_pairing_score`).
    """
    out: Dict[str, ExpectationCheck] = {}

    out["alphabet_size_curve"] = _sublinear_vs_linear_check(
        corpus_size_snapshots, n_classes_snapshots,
        expected_shape="sublinear, saturating growth of unique stable classes with corpus size (Heaps'-law-like)",
        corpus_version=corpus_version,
    )

    stability_scores = [c.cluster_stability.score for c in letter_classes]
    out["cluster_stability_distribution"] = _bimodality_check(
        stability_scores,
        expected_shape="bimodal: a cluster near 1.0 (real letters) separate from a cluster near 0 (segmentation noise)",
        corpus_version=corpus_version,
    )

    out["occurrence_frequency"] = _powerlaw_ks_check(
        class_occurrence_counts,
        expected_shape="Zipfian (power-law) frequency distribution -- few very frequent letters, long tail",
        corpus_version=corpus_version,
    )

    if intrinsic_dimension is not None and raw_dimension:
        ratio = intrinsic_dimension.score / raw_dimension if raw_dimension else 1.0
        out["intrinsic_dimension"] = _range_check(
            ratio, low=0.0, high=0.5,
            expected_shape="intrinsic dimension of invariant-vector manifold << raw bispectrum_vector dimension",
            n=intrinsic_dimension.n, corpus_version=corpus_version,
        )
    else:
        out["intrinsic_dimension"] = ExpectationCheck(
            expected_shape="intrinsic dimension << raw dimension", observed_shape="no data supplied",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    if within_class_entropies and between_class_entropy > 0:
        ratio = float(np.mean(within_class_entropies)) / between_class_entropy
        out["entropy_per_letter"] = _range_check(
            ratio, low=0.0, high=0.6,
            expected_shape="low within-class entropy relative to between-class entropy",
            n=len(within_class_entropies), corpus_version=corpus_version,
        )
    else:
        out["entropy_per_letter"] = ExpectationCheck(
            expected_shape="low within-class entropy relative to between-class",
            observed_shape="no data supplied", test_method="range_check", p_value=None,
            confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )
    return out


# ==========================================================================
# §19.3 ExpectedWordCorpusProfile -- generic array inputs (word_class §4.6
# is not built by this scaffold; see distributional_test.py module docstring)
# ==========================================================================


def build_word_corpus_profile(
    word_lengths: Sequence[int],
    occupancy_frequencies: Sequence[float],
    topology_variances: Sequence[float],
    invariant_distances_for_topology: Sequence[float],
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.3. Every argument is caller-supplied raw data (counts of letters
    per word; per-word occupancy of the modal constituent letter class;
    per-word connectivity-graph variance across instances) -- this
    scaffold has no `word_class` layer producing these itself (see
    `distributional_test.py`'s module docstring for exactly what is
    missing and why)."""
    out: Dict[str, ExpectationCheck] = {}

    if len(word_lengths) >= 5:
        from scipy.stats import chisquare
        arr = np.asarray(word_lengths)
        vals, counts = np.unique(arr, return_counts=True)
        stat, p = chisquare(counts)
        verdict = "supports_theory" if p < 0.05 else "against_theory"
        out["word_length_distribution"] = ExpectationCheck(
            expected_shape="a meaningful modal scale of letters-per-word, not unbounded/arbitrary length",
            observed_shape=f"mode={int(vals[np.argmax(counts)])}, std={float(arr.std()):.2f}",
            test_method="chi_square", p_value=float(p), confidence=None,
            corpus_version=corpus_version, verdict=verdict,
        )
    else:
        out["word_length_distribution"] = ExpectationCheck(
            expected_shape="a meaningful modal scale of letters-per-word",
            observed_shape="too few words (<5)", test_method="chi_square", p_value=None,
            confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )

    if occupancy_frequencies:
        arr = np.asarray(occupancy_frequencies, dtype=float)
        frac_high = float(np.mean(arr > 0.95))
        std = float(arr.std())
        # supports theory: not degenerate (std ~ 0) and not fully random
        # (frac_high near 1 for MOST words, but a meaningful minority lower)
        verdict = "supports_theory" if (std > 0.02 and 0.3 <= frac_high <= 0.95) else "against_theory"
        out["occupancy_frequency"] = ExpectationCheck(
            expected_shape="high occupancy for most words, but a meaningful fraction show low occupancy (letter synonymy within a word, §4.4)",
            observed_shape=f"fraction near-100% occupancy={frac_high:.2f}, std={std:.2f}",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict=verdict,
        )
    else:
        out["occupancy_frequency"] = ExpectationCheck(
            expected_shape="high occupancy for most words, meaningful low-occupancy tail",
            observed_shape="no data supplied", test_method="range_check", p_value=None,
            confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )

    out["connectivity_topology.topology_variance"] = _correlation_check(
        invariant_distances_for_topology, topology_variances, expected_sign="none",
        expected_shape="low topology_variance for real words; high variance should NOT concentrate at low invariant distance (would signal two words merged under one class_id)",
        corpus_version=corpus_version,
    )
    return out


# ==========================================================================
# §19.4 ExpectedGrammarProfile
# ==========================================================================


def build_grammar_profile(
    cooccurrence_entries: Sequence[CooccurrenceEntry],
    forbidden_entries: Sequence[ForbiddenMapEntry],
    n_combinatorially_valid_pairs: int,
    transition_model: Optional[TransitionModel],
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.4."""
    out: Dict[str, ExpectationCheck] = {}

    degree: Dict[str, float] = {}
    for e in cooccurrence_entries:
        a, b = e.class_id_pair
        degree[a] = degree.get(a, 0.0) + e.ppmi_value
        degree[b] = degree.get(b, 0.0) + e.ppmi_value
    out["co_occurrence_matrix.ppmi_distribution"] = _uniform_chi_square_check(
        list(degree.values()),
        expected_shape="clustered, hub/periphery structure -- not uniform connectivity",
        corpus_version=corpus_version, reject_uniform=True,
    )

    if n_combinatorially_valid_pairs > 0:
        coverage = len(forbidden_entries) / n_combinatorially_valid_pairs
        out["forbidden_map.coverage"] = _range_check(
            coverage, low=0.02, high=0.85,
            expected_shape="meaningful but non-extreme fraction of combinatorially valid pairs physically forbidden",
            n=n_combinatorially_valid_pairs, corpus_version=corpus_version,
        )
    else:
        out["forbidden_map.coverage"] = ExpectationCheck(
            expected_shape="meaningful, non-extreme forbidden-pair coverage",
            observed_shape="no combinatorially-valid-pair count supplied", test_method="range_check",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )

    if transition_model is not None and transition_model.transition_probabilities:
        by_state: Dict[str, List[float]] = {}
        for entry in transition_model.transition_probabilities:
            by_state.setdefault(entry["from"], []).append(entry["probability"])
        sparsities = [
            float(np.mean(np.array(probs) > (1.0 / len(probs)) * 0.1)) if probs else 0.0
            for probs in by_state.values()
        ]
        # fraction of a state's outgoing mass concentrated in "few" targets:
        # use Gini of per-state probability vectors as the sparsity proxy.
        ginis = []
        for probs in by_state.values():
            arr = np.sort(np.array(probs))
            n = len(arr)
            if n < 2 or arr.sum() == 0:
                continue
            ginis.append(float(np.sum((2 * np.arange(1, n + 1) - n - 1) * arr) / (n * arr.sum())))
        mean_gini = float(np.mean(ginis)) if ginis else 0.0
        out["transition_model.sparsity"] = _range_check(
            mean_gini, low=0.25, high=1.01,
            expected_shape="sparse transition matrix -- from each state, few high-probability successors (high Gini), not near-uniform",
            n=len(by_state), corpus_version=corpus_version,
        )
    else:
        out["transition_model.sparsity"] = ExpectationCheck(
            expected_shape="sparse transition matrix (high Gini of per-state outgoing probabilities)",
            observed_shape="no transition model supplied", test_method="range_check",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )
    return out


# ==========================================================================
# §19.5 ExpectedWordMeaningProfile
# ==========================================================================


def build_word_meaning_profile(
    word_classes_semantic_status: Sequence[SemanticStatus],
    descriptor_matrix: Optional[np.ndarray],
    paradigms: Sequence[Paradigm],
    paradigm_member_descriptor_shifts: Optional[Dict[str, Sequence[float]]] = None,
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.5. `descriptor_matrix` -- (n_words, n_named_descriptor_axes)
    array of already-measured functional descriptors (pKa, nucleophilicity
    index, redox potential, ...) per word, if available.
    `paradigm_member_descriptor_shifts` -- optional {paradigm_id:
    [per-member descriptor shift along the paradigm's varying axis]}, for
    the §4.4-4.6 consistency check `paradigm_correlated_shift`.
    """
    out: Dict[str, ExpectationCheck] = {}

    if word_classes_semantic_status:
        n_named = sum(1 for s in word_classes_semantic_status if not s.orphan)
        coverage = n_named / len(word_classes_semantic_status)
        out["named_correlates.coverage_fraction"] = _range_check(
            coverage, low=0.05, high=0.95,
            expected_shape="partial named-property coverage -- neither ~0%% (invariant tracks nothing chemical) nor ~100%% (pure rediscovery of known chemistry, §16.3)",
            n=len(word_classes_semantic_status), corpus_version=corpus_version,
        )
    else:
        out["named_correlates.coverage_fraction"] = ExpectationCheck(
            expected_shape="partial named-property coverage", observed_shape="no classes supplied",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    if descriptor_matrix is not None and descriptor_matrix.shape[0] >= 5 and descriptor_matrix.shape[1] >= 2:
        corr = np.corrcoef(descriptor_matrix, rowvar=False)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.clip(eigvals, 0, None)
        explained = eigvals / eigvals.sum()
        n_components_90 = int(np.searchsorted(np.cumsum(np.sort(explained)[::-1]), 0.9) + 1)
        verdict = "supports_theory" if n_components_90 >= 2 else "against_theory"
        out["descriptors"] = ExpectationCheck(
            expected_shape="meaning as a multi-axis distribution, not fully collinear with one hidden parameter",
            observed_shape=f"{n_components_90} components needed for 90% variance out of {descriptor_matrix.shape[1]} axes",
            test_method="AIC_comparison", p_value=None,
            confidence=ConfidenceBlock(n=descriptor_matrix.shape[0], method="analytic_CI", score=float(n_components_90), corpus_version=corpus_version),
            corpus_version=corpus_version, verdict=verdict,
        )
    else:
        out["descriptors"] = ExpectationCheck(
            expected_shape="meaning as a multi-axis, non-collinear distribution",
            observed_shape="insufficient descriptor data (need >=5 words, >=2 axes)",
            test_method="AIC_comparison", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    if paradigm_member_descriptor_shifts:
        all_shifts = [s for shifts in paradigm_member_descriptor_shifts.values() for s in shifts]
        within_var = float(np.mean([np.var(shifts) for shifts in paradigm_member_descriptor_shifts.values() if len(shifts) > 1])) if all_shifts else float("nan")
        overall_var = float(np.var(all_shifts)) if all_shifts else float("nan")
        if overall_var and overall_var > 0 and not np.isnan(within_var):
            ratio = within_var / overall_var
            out["paradigm_correlated_shift"] = _range_check(
                ratio, low=0.0, high=0.6,
                expected_shape="paradigm members show a CONSISTENT descriptor shift along the varying axis (low within-paradigm variance relative to overall)",
                n=len(paradigm_member_descriptor_shifts), corpus_version=corpus_version,
            )
        else:
            out["paradigm_correlated_shift"] = ExpectationCheck(
                expected_shape="consistent within-paradigm descriptor shift", observed_shape="degenerate variance",
                test_method="range_check", p_value=None, confidence=None,
                corpus_version=corpus_version, verdict="inconclusive",
            )
    else:
        out["paradigm_correlated_shift"] = ExpectationCheck(
            expected_shape="consistent within-paradigm descriptor shift along the varying axis",
            observed_shape=f"{len(paradigms)} paradigms detected, no descriptor-shift data supplied",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    return out


# ==========================================================================
# §19.6 ExpectedSentenceMeaningProfile -- Phases 6-10 data, generic inputs
# ==========================================================================


def build_sentence_meaning_profile(
    additive_prediction_errors: Sequence[float],
    full_sentence_prediction_errors: Sequence[float],
    success_by_grammaticality_status: Dict[str, Sequence[int]],
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.6. All three arguments are steps 7-11 (trajectory/execution)
    data this scaffold does not itself produce (see `__init__.py`'s
    module docstring on scope) -- shapes match exactly what the doc's
    profile needs, ready to consume once that data exists.

    `additive_prediction_errors` / `full_sentence_prediction_errors` --
    paired |predicted - actual| errors for the same set of sentences,
    under an additive (sum-of-word-meanings) model vs the full assembled-
    sentence calculation (§3/§4 emergence).
    `success_by_grammaticality_status` -- {"attested": [0/1,...],
    "combinatorially_valid_untested": [...], "physically_forbidden": [...]}
    execution outcomes, one list per `schema.RealizationExemplar.
    grammaticality_status`.
    """
    from scipy.stats import wilcoxon

    out: Dict[str, ExpectationCheck] = {}

    add = np.asarray(additive_prediction_errors, dtype=float)
    full = np.asarray(full_sentence_prediction_errors, dtype=float)
    if len(add) >= 6 and len(add) == len(full):
        try:
            stat, p = wilcoxon(add, full, alternative="greater")
        except ValueError:
            p = None
        verdict = "supports_theory" if (p is not None and p < 0.05 and np.median(add) > np.median(full)) else "against_theory"
        out["emergence_gap"] = ExpectationCheck(
            expected_shape="additive (sum-of-words) prediction error SYSTEMATICALLY worse than full-sentence calculation",
            observed_shape=f"median additive error={np.median(add):.3f}, median full error={np.median(full):.3f}",
            test_method="permutation", p_value=p, confidence=None,
            corpus_version=corpus_version, verdict=verdict,
        )
    else:
        out["emergence_gap"] = ExpectationCheck(
            expected_shape="additive prediction systematically worse than full-sentence calculation",
            observed_shape="insufficient paired data (need >=6 matched sentences)",
            test_method="permutation", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    order = ["attested", "combinatorially_valid_untested", "physically_forbidden"]
    groups = [
        (status, float(np.mean(success_by_grammaticality_status[status])), len(success_by_grammaticality_status[status]))
        for status in order
        if status in success_by_grammaticality_status and len(success_by_grammaticality_status[status]) > 0
    ]
    out["execution_success_rate_by_grammaticality_status"] = _trend_check(
        groups,
        expected_shape="attested >> combinatorially_valid_untested >> physically_forbidden execution success rate",
        corpus_version=corpus_version,
    )

    out["execution_vs_sentence_meaning"] = ExpectationCheck(
        expected_shape="repeated executions of the SAME sentence show bounded, nonzero outcome variance (felicity conditions, §7.2) -- not identical every time, not fully random",
        observed_shape="conceptual check -- supply per-sentence repeated-execution outcome variance to evaluate quantitatively",
        test_method="range_check", p_value=None, confidence=None,
        corpus_version=corpus_version, verdict="inconclusive",
    )
    return out


# ==========================================================================
# §19.7 ExpectedAlphabetDiscoveryProfile
# ==========================================================================


def _match_snapshots(
    prev_classes: Sequence[LetterClass],
    curr_classes: Sequence[LetterClass],
    distance_threshold: Union[float, Mapping[str, float]],
) -> Tuple[List[str], List[str], List[Tuple[str, List[str]]]]:
    """Greedy nearest-centroid matching between two successive alphabet
    snapshots, WITHIN each `critical_point_type` partition (§3.2
    discipline preserved here too). Returns
    (stable_ids_in_curr, novel_ids_in_curr, merges) where `merges` lists
    every prev class matched by more than one curr class (a split) or
    every curr class matched by more than one prev class (a merge) --
    both directions reported together since which one occurred is exactly
    what `merge_rate`/`split_rate` need to tell apart.
    """
    stable: List[str] = []
    novel: List[str] = []
    prev_matches: Dict[str, List[str]] = {}

    prev_by_type: Dict[str, List[LetterClass]] = {}
    for c in prev_classes:
        prev_by_type.setdefault(c.discovery_provenance.critical_point_type, []).append(c)

    for c in curr_classes:
        ft = c.discovery_provenance.critical_point_type
        candidates = prev_by_type.get(ft, [])
        if not candidates:
            novel.append(c.class_id)
            continue
        dists = [float(np.linalg.norm(c.invariant_centroid - p.invariant_centroid)) for p in candidates]
        best = int(np.argmin(dists))
        threshold = (
            float(distance_threshold[ft])
            if isinstance(distance_threshold, Mapping)
            else float(distance_threshold)
        )
        if dists[best] <= threshold:
            stable.append(c.class_id)
            prev_matches.setdefault(candidates[best].class_id, []).append(c.class_id)
        else:
            novel.append(c.class_id)

    merges = [(prev_id, curr_ids) for prev_id, curr_ids in prev_matches.items()]
    return stable, novel, merges


def build_alphabet_discovery_profile(
    snapshot_classes: Sequence[Sequence[LetterClass]],
    snapshot_sizes: Sequence[int],
    distance_threshold: Union[float, Mapping[str, float]],
    snapshot_candidate_counts: Optional[Sequence[int]] = None,
    held_out_candidates: Optional[Sequence[LetterCandidate]] = None,
    fundamentality_config: Optional[FundamentalityConfig] = None,
    snapshot_candidate_pools: Optional[Sequence[Sequence[LetterCandidate]]] = None,
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.7, the profile §19.10 calls logically prior to the rest.

    `snapshot_classes[i]` / `snapshot_sizes[i]` -- the alphabet
    (`corpus.cluster_candidates_bootstrap` output) and molecule count at
    successive corpus checkpoints (i.e. repeated `pipeline.decode_alphabet`
    runs on growing subsets of the same corpus, in order). Needs >= 3
    snapshots for the rate curves to say anything about convergence.

    `snapshot_candidate_counts[i]` is the raw critical-point population at
    the same checkpoint.  It supplies the linear candidate-growth curve and
    lets a bounded compression sample be amortised over the true prefix
    population rather than mistaken for a full prefix pool.

    `held_out_candidates` -- candidates from molecules NOT included in any
    snapshot, for `rediscovery_rate` (§19.7's "decisive point... does not
    depend on how the curve is computed on the same data used for
    training" -- literally requires data the training curve never saw).

    `snapshot_candidate_pools[i]`, if given, enables `compression_gain`
    (§12/§19.7 MDL proxy) via `fundamentality.fit_shared_dictionary` at
    each snapshot -- omitted (returns "inconclusive") without it, since
    computing it needs the raw descriptor pool, not just the resulting
    classes.
    """
    out: Dict[str, ExpectationCheck] = {}
    n_snap = len(snapshot_classes)

    class_counts = [len(sc) for sc in snapshot_classes]
    candidate_counts = list(snapshot_candidate_counts or [])
    out["candidate_count_curve"] = ExpectationCheck(
        expected_shape="linear growth of raw critical-point candidate count with corpus size (no saturation)",
        observed_shape=(
            f"raw candidate counts per snapshot: {candidate_counts}"
            if candidate_counts else "raw candidate counts were not supplied"
        ),
        test_method="AIC_comparison", p_value=None, confidence=None,
        corpus_version=corpus_version, verdict="inconclusive",
    )
    if n_snap >= 4 and len(candidate_counts) == n_snap:
        # reuse the sublinear-vs-linear machinery but invert the read: here
        # linear (b close to 1) is the theory-consistent outcome.
        check = _sublinear_vs_linear_check(
            snapshot_sizes,
            candidate_counts,
            "linear raw-candidate growth",
            corpus_version,
        )
        b = check.confidence.score if check.confidence else None
        verdict = "supports_theory" if (b is not None and b > 0.85) else "against_theory"
        out["candidate_count_curve"] = ExpectationCheck(
            expected_shape="linear growth of raw candidate count with corpus size",
            observed_shape=check.observed_shape, test_method="AIC_comparison",
            p_value=None, confidence=check.confidence, corpus_version=corpus_version, verdict=verdict,
        )

    if n_snap >= 2:
        merge_rates, split_rates, novel_rates, coverage_rates = [], [], [], []
        for i in range(1, n_snap):
            stable, novel, merges = _match_snapshots(snapshot_classes[i - 1], snapshot_classes[i], distance_threshold)
            n_curr = len(snapshot_classes[i])
            n_merged_targets = sum(1 for _, curr_ids in merges if len(curr_ids) > 1)
            # split: one prev class matched by multiple curr classes already
            # captured above; merge (several prev -> one curr) needs the
            # reverse direction, symmetric call:
            _, _, reverse_merges = _match_snapshots(snapshot_classes[i], snapshot_classes[i - 1], distance_threshold)
            n_merge_events = sum(1 for _, prev_ids in reverse_merges if len(prev_ids) > 1)

            merge_rates.append(n_merge_events / max(1, len(snapshot_classes[i - 1])))
            split_rates.append(n_merged_targets / max(1, len(snapshot_classes[i - 1])))
            novel_rates.append(len(novel) / max(1, n_curr))
            coverage_rates.append(len(stable) / max(1, n_curr))

        out["merge_rate"] = ExpectationCheck(
            expected_shape="merge_rate decreasing toward 0 with corpus size",
            observed_shape=f"per-snapshot merge_rate: {[round(r,3) for r in merge_rates]}",
            test_method="trend_test", p_value=None, confidence=None,
            corpus_version=corpus_version,
            verdict="supports_theory" if len(merge_rates) >= 2 and merge_rates[-1] <= merge_rates[0] else "against_theory",
        )
        out["split_rate"] = ExpectationCheck(
            expected_shape="split_rate decreasing toward 0 with corpus size",
            observed_shape=f"per-snapshot split_rate: {[round(r,3) for r in split_rates]}",
            test_method="trend_test", p_value=None, confidence=None,
            corpus_version=corpus_version,
            verdict="supports_theory" if len(split_rates) >= 2 and split_rates[-1] <= split_rates[0] else "against_theory",
        )
        out["novel_letter_rate"] = ExpectationCheck(
            expected_shape="novel_letter_rate decreasing with corpus size (Heaps'-law-like), not necessarily to exact zero",
            observed_shape=f"per-snapshot novel_letter_rate: {[round(r,3) for r in novel_rates]}",
            test_method="trend_test", p_value=None, confidence=None,
            corpus_version=corpus_version,
            verdict="supports_theory" if len(novel_rates) >= 2 and novel_rates[-1] <= novel_rates[0] else "against_theory",
        )
        out["corpus_coverage"] = _range_check(
            coverage_rates[-1] if coverage_rates else 0.0, low=0.3, high=0.98,
            expected_shape="coverage rises toward a plateau, but NOT 100% (Zipf tail always leaves some uncovered)",
            n=n_snap, corpus_version=corpus_version,
        )

        stable_fracs = coverage_rates
        out["stable_core_fraction"] = ExpectationCheck(
            expected_shape="Pareto-like: a small core of classes stabilizes almost immediately",
            observed_shape=f"stable-fraction curve: {[round(r,3) for r in stable_fracs]}",
            test_method="AIC_comparison", p_value=None, confidence=None,
            corpus_version=corpus_version,
            verdict="supports_theory" if stable_fracs and stable_fracs[0] > 0.5 else "inconclusive",
        )
    else:
        for key in ("merge_rate", "split_rate", "novel_letter_rate", "corpus_coverage", "stable_core_fraction"):
            out[key] = ExpectationCheck(
                expected_shape="see doc §19.7", observed_shape="need >= 2 snapshots",
                test_method="trend_test", p_value=None, confidence=None,
                corpus_version=corpus_version, verdict="inconclusive",
            )

    if held_out_candidates and snapshot_classes:
        latest = snapshot_classes[-1]
        rediscovered = 0
        n_eligible = 0
        for c in held_out_candidates:
            same_type = [lc for lc in latest if lc.discovery_provenance.critical_point_type == c.critical_point.field_type]
            if not same_type:
                continue
            n_eligible += 1
            vec = c.descriptor.to_vector()
            dists = [float(np.linalg.norm(vec - lc.invariant_centroid)) for lc in same_type]
            threshold = (
                float(distance_threshold[c.critical_point.field_type])
                if isinstance(distance_threshold, Mapping)
                else float(distance_threshold)
            )
            if min(dists) <= threshold:
                rediscovered += 1
        rate = rediscovered / n_eligible if n_eligible else 0.0
        train_coverage = coverage_rates[-1] if n_snap >= 2 and coverage_rates else None
        verdict = "inconclusive"
        if train_coverage is not None:
            verdict = "supports_theory" if rate >= train_coverage - 0.15 else "against_theory"
        out["rediscovery_rate"] = _range_check(
            rate, low=0.3, high=1.01,
            expected_shape="high, close to training-corpus coverage -- held-out critical points mostly land in already-known classes",
            n=n_eligible, corpus_version=corpus_version,
        )
        out["rediscovery_rate"].verdict = verdict
    else:
        out["rediscovery_rate"] = ExpectationCheck(
            expected_shape="high rediscovery rate on held-out molecules, close to training coverage",
            observed_shape="no held-out candidates supplied", test_method="range_check",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )

    if snapshot_candidate_pools and fundamentality_config is not None and n_snap >= 3:
        gains = []
        sampled_pools = False
        for pool_index, pool in enumerate(snapshot_candidate_pools):
            if len(pool) < 2:
                gains.append(0.0)
                continue
            try:
                fit = fit_shared_dictionary(pool, fundamentality_config)
            except Exception:
                gains.append(0.0)
                continue
            vectors = np.stack([c.descriptor.to_vector() for c in pool])
            raw_bits = vectors.size * 32
            n_atoms, dim = fit.dictionary.shape
            dict_bits = n_atoms * dim * 32
            # A reservoir pool estimates per-candidate coding cost, but the
            # shared dictionary is amortised over the true prefix size.  The
            # old implementation used len(pool) for both quantities, making
            # a bounded sample masquerade as a tiny corpus and biasing the
            # growth curve.  Actual OMP sparsity (not its requested maximum)
            # is used because linear dependence may validly stop OMP early.
            population_size = (
                int(candidate_counts[pool_index])
                if pool_index < len(candidate_counts) else len(pool)
            )
            sampled_pools = sampled_pools or population_size != len(pool)
            mean_active = float(fit.codes.nnz) / max(1, len(pool))
            full_raw_bits = population_size * dim * 32
            code_bits = population_size * mean_active * (np.log2(max(2, n_atoms)) + 32)
            gains.append(max(0.0, 1.0 - (dict_bits + code_bits) / full_raw_bits))
        deltas = [gains[i + 1] - gains[i] for i in range(len(gains) - 1)]
        plateauing = len(deltas) >= 2 and abs(deltas[-1]) <= abs(deltas[0])
        out["compression_gain"] = ExpectationCheck(
            expected_shape="compression gain rising then plateauing with corpus size (quantitative MDL version of 'the alphabet compresses')",
            observed_shape=(
                f"gain curve: {[round(float(g), 3) for g in gains]}"
                + (" (dictionary fit on bounded reservoir samples; coding cost scaled to raw prefix size)"
                   if sampled_pools else "")
            ),
            test_method="AIC_comparison", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="supports_theory" if plateauing else "against_theory",
        )
    else:
        out["compression_gain"] = ExpectationCheck(
            expected_shape="rising-then-plateauing compression gain (MDL) with corpus size",
            observed_shape="need >=3 snapshots' raw candidate pools + a FundamentalityConfig",
            test_method="AIC_comparison", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )
    return out


# ==========================================================================
# §19.8 ExpectedSynonymProfile
# ==========================================================================


def build_synonym_profile(
    synonym_relations: Sequence[SynonymRelation],
    realization_counts_per_member: Optional[Sequence[int]] = None,
    typical_within_class_realization_count: Optional[float] = None,
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.8. `synonym_relations` should be filtered to
    `classification == "idiosyncratic_synonym"` entries from
    `paradigm.build_synonym_relations` -- declension-paradigm members are a
    different phenomenon (§4.4) and do not belong in this profile."""
    synonyms = [r for r in synonym_relations if r.classification == "idiosyncratic_synonym"]
    out: Dict[str, ExpectationCheck] = {}

    if realization_counts_per_member and typical_within_class_realization_count:
        mean_count = float(np.mean(realization_counts_per_member))
        ratio = mean_count / typical_within_class_realization_count if typical_within_class_realization_count else 0.0
        out["realization_diversity"] = _range_check(
            ratio, low=1.0, high=1e9,
            expected_shape="higher structural heterogeneity of realizations across synonym-cluster members than typical within one ordinary class",
            n=len(realization_counts_per_member), corpus_version=corpus_version,
        )
    else:
        out["realization_diversity"] = ExpectationCheck(
            expected_shape="higher realization heterogeneity across synonym members than within one class",
            observed_shape="no realization-count data supplied", test_method="range_check",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )

    graph_d = [r.graph_distance for r in synonyms if r.graph_distance is not None]
    inv_d_for_graph = [r.invariant_distance for r in synonyms if r.graph_distance is not None]
    if len(graph_d) >= 4:
        out["graph_distance"] = _range_check(
            float(np.mean(graph_d)), low=0.4, high=1.01,
            expected_shape="large atomic-graph distance (e.g. Tanimoto on Morgan fingerprints) between synonym-cluster members",
            n=len(graph_d), corpus_version=corpus_version,
        )
    else:
        out["graph_distance"] = ExpectationCheck(
            expected_shape="large graph distance between synonym members", observed_shape="too few pairs with graph_distance (<4)",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    func_d = [r.function_distance for r in synonyms if r.function_distance is not None]
    if len(func_d) >= 4:
        out["function_distance"] = _range_check(
            float(np.mean(func_d)), low=0.0, high=0.4,
            expected_shape="small distance on named functional properties -- the basis for calling the cluster synonyms at all",
            n=len(func_d), corpus_version=corpus_version,
        )
    else:
        out["function_distance"] = ExpectationCheck(
            expected_shape="small functional-property distance", observed_shape="too few pairs with function_distance (<4)",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    inv_d = [r.invariant_distance for r in synonyms]
    if len(inv_d) >= 4:
        out["invariant_distance"] = _range_check(
            float(np.mean(inv_d)), low=0.0, high=0.5,
            expected_shape="small ESM-invariant distance between synonym members, generalizing the single -COOH/tetrazole calibration pair (§15 step 5) to the whole corpus",
            n=len(inv_d), corpus_version=corpus_version,
        )
    else:
        out["invariant_distance"] = ExpectationCheck(
            expected_shape="small invariant distance between synonym members", observed_shape="too few pairs (<4)",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    if len(graph_d) >= 4 and len(inv_d_for_graph) >= 4:
        graph_inv_check = _correlation_check(
            graph_d, inv_d_for_graph, expected_sign="none",
            expected_shape="graph_distance NOT positively correlated with invariant_distance (decisive joint pattern, §19.8)",
            corpus_version=corpus_version,
        )
    else:
        graph_inv_check = None
    if len(inv_d) >= 4 and len(func_d) >= 4 and len(inv_d) == len(func_d):
        inv_func_check = _correlation_check(
            inv_d, func_d, expected_sign="positive",
            expected_shape="invariant_distance POSITIVELY correlated with function_distance (decisive joint pattern, §19.8)",
            corpus_version=corpus_version,
        )
    else:
        inv_func_check = None

    if graph_inv_check is not None and inv_func_check is not None:
        joint_verdict = (
            "supports_theory"
            if graph_inv_check.verdict == "supports_theory" and inv_func_check.verdict == "supports_theory"
            else "against_theory"
        )
        out["decisive_joint_pattern"] = ExpectationCheck(
            expected_shape="no/negative graph<->invariant correlation AND positive invariant<->function correlation, jointly (§19.8 decisive signal)",
            observed_shape=f"graph<->invariant: {graph_inv_check.observed_shape}; invariant<->function: {inv_func_check.observed_shape}",
            test_method="correlation", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict=joint_verdict,
        )
    else:
        out["decisive_joint_pattern"] = ExpectationCheck(
            expected_shape="no/negative graph<->invariant correlation AND positive invariant<->function correlation, jointly",
            observed_shape="insufficient paired data for one or both correlations", test_method="correlation",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )
    return out


# ==========================================================================
# §19.9 ExpectedPolysemyProfile
# ==========================================================================


def build_polysemy_profile(
    environment_response_profiles: Sequence[EnvironmentResponseProfile],
    realization_similarities: Optional[Sequence[float]] = None,
    function_divergences: Optional[Sequence[float]] = None,
    context_dependence_r2: Optional[Sequence[float]] = None,
    corpus_version: str = "unversioned",
) -> Dict[str, ExpectationCheck]:
    """§19.9. `environment_response_profiles` reuses
    `environment_response.py`'s already-populated
    `ResponseSmoothness.classification` directly for
    `environment_switch_rate` -- no recomputation, same
    staleness-avoidance discipline as `HypothesisStatus.
    obligatory_pairing_score`. The other three fields are optional
    caller-supplied per-class arrays (§13's `RealizationRegistry` and
    §19.5's descriptor layer are not populated by this scaffold, so this
    scaffold cannot derive them itself -- see schema.py's module
    docstring).
    """
    out: Dict[str, ExpectationCheck] = {}

    classified = [
        p.response_smoothness.classification
        for p in environment_response_profiles
        if p.response_smoothness is not None
        and p.response_smoothness.classification != "insufficient_data"
    ]
    if classified:
        switch_rate = sum(1 for c in classified if c == "discontinuous_polysemy") / len(classified)
        out["environment_switch_rate"] = _range_check(
            switch_rate, low=0.02, high=0.8,
            expected_shape="a meaningful but non-dominant fraction of classes show discontinuous_polysemy rather than smooth indexicality",
            n=len(classified), corpus_version=corpus_version,
        )
    else:
        out["environment_switch_rate"] = ExpectationCheck(
            expected_shape="meaningful, non-dominant discontinuous_polysemy fraction",
            observed_shape="no classified environment-response profiles supplied", test_method="range_check",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )

    if realization_similarities:
        out["realization_similarity"] = _range_check(
            float(np.mean(realization_similarities)), low=0.6, high=1.01,
            expected_shape="high realization similarity -- same class_id, not an artifact of two classes merged (§15 step 3)",
            n=len(realization_similarities), corpus_version=corpus_version,
        )
    else:
        out["realization_similarity"] = ExpectationCheck(
            expected_shape="high realization similarity within a polysemous class", observed_shape="no data supplied",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    if function_divergences:
        out["function_divergence"] = _range_check(
            float(np.mean(function_divergences)), low=0.3, high=1e9,
            expected_shape="substantial functional divergence across contexts despite high realization similarity",
            n=len(function_divergences), corpus_version=corpus_version,
        )
    else:
        out["function_divergence"] = ExpectationCheck(
            expected_shape="substantial cross-context functional divergence", observed_shape="no data supplied",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict="inconclusive",
        )

    if context_dependence_r2 is not None and function_divergences is not None and len(context_dependence_r2) == len(function_divergences) and len(context_dependence_r2) >= 4:
        # decisive interpretive fork (§19.9): high context_dependence +
        # large divergence => polysemy; low context_dependence + large
        # divergence => aliasing artifact.
        r2 = np.asarray(context_dependence_r2, dtype=float)
        div = np.asarray(function_divergences, dtype=float)
        large_div = div > np.median(div)
        mean_r2_large_div = float(np.mean(r2[large_div])) if large_div.any() else 0.0
        verdict = "supports_theory" if mean_r2_large_div > 0.4 else "against_theory"
        out["context_dependence"] = ExpectationCheck(
            expected_shape="high context_dependence (R^2 / mutual information vs environment) specifically among high-function_divergence classes -- polysemy, not descriptor aliasing",
            observed_shape=f"mean context_dependence among high-divergence classes = {mean_r2_large_div:.2f}",
            test_method="range_check", p_value=None, confidence=None,
            corpus_version=corpus_version, verdict=verdict,
        )
    else:
        out["context_dependence"] = ExpectationCheck(
            expected_shape="high context_dependence among high-function_divergence classes (polysemy vs aliasing fork)",
            observed_shape="insufficient paired data", test_method="range_check",
            p_value=None, confidence=None, corpus_version=corpus_version, verdict="inconclusive",
        )
    return out
