"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
semantics.py
============
Doc §16.1, §16.3 ("Эпистемологическая проблема" / "Поиск сирот"): bridge
between isostere calibration (step 5) and the final `LetterClass` output.

This module implements the logic to answer whether a structurally stable
class correlates with known functional properties, or if it is an "orphan" --
a genuine candidate for undescribed function.

Crucially, per §16.3: "насколько уверенно установлено отсутствие корреляции,
а не просто 'пока не найдена'". A class is only a confident orphan if we
have statistically powerful tests showing that its correlation with known
properties is actively low, not just tests that failed to reach significance
due to small sample sizes. This is operationalized via equivalence testing:
checking if the *upper bound* of the bootstrap confidence interval for the
correlation coefficient falls strictly below the `min_correlation` threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .isostere import (
    CalibrationItem,
    MantelResult,
    build_functional_dissimilarity_matrix,
    build_invariant_distance_matrix,
    mantel_test,
)
from .schema import ConfidenceBlock, LetterClass, NamedCorrelate, SemanticStatus


@dataclass
class NamedPropertyTest:
    """The result of testing one specific functional property against the
    invariant space distances for a set of class exemplars."""
    property_name: str
    mantel_result: MantelResult
    correlation_ci: tuple  # (lo, hi) bootstrap confidence interval


def mantel_bootstrap_interval(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    method: Literal["pearson", "spearman"] = "spearman",
    n_boot: int = 500,
    rng_seed: int = 0,
) -> tuple:
    """Bootstrap confidence interval for the Mantel correlation.
    
    Resamples items (rows/cols simultaneously) to preserve the distance
    matrix structure. Pairs of identical items (i == j in the resampled
    indices) are excluded from the correlation calculation to avoid
    artificial zero-distance inflations that occur when an item is sampled
    multiple times in the same bootstrap draw.
    """
    n = matrix_a.shape[0]
    rng = np.random.default_rng(rng_seed)
    corrs = []
    # Computed once, not per bootstrap draw: the upper-triangle (i<j) index
    # pairs are the same n*(n-1)/2 pairs on every iteration; only which of
    # them get excluded by idx[i]==idx[j] changes.
    iu, ju = np.triu_indices(n, k=1)

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a_boot = matrix_a[np.ix_(idx, idx)]
        b_boot = matrix_b[np.ix_(idx, idx)]

        # Vectorised replacement for what was a pure-Python O(n^2) double
        # loop run once per bootstrap draw (so O(n^2 * n_boot) Python-level
        # iterations overall) -- at this function's own `max_molecules_per_class`
        # cap of 2000 and the default n_boot=500, that was ~1e9 Python-level
        # loop iterations for a single class, entirely on one core (hence
        # low overall CPU% despite being fully busy) and the dominant cost
        # of the whole per-class semantics step. Same exclusion rule (drop
        # pairs where the resampled indices happen to coincide), same
        # values, computed as array operations instead of appends.
        keep = idx[iu] != idx[ju]
        a_vals = a_boot[iu, ju][keep]
        b_vals = b_boot[iu, ju][keep]

        if len(np.unique(a_vals)) > 1 and len(np.unique(b_vals)) > 1:
            if method == "pearson":
                # safeguard against constant vectors within a bootstrap draw
                if np.std(a_vals) == 0.0 or np.std(b_vals) == 0.0:
                    c = 0.0
                else:
                    c = float(np.corrcoef(a_vals, b_vals)[0, 1])
            else:
                from scipy.stats import spearmanr
                stat = spearmanr(a_vals, b_vals).correlation
                c = float(stat) if stat is not None and not np.isnan(stat) else 0.0
            corrs.append(c)
        else:
            corrs.append(0.0)

    if not corrs:
        return (0.0, 0.0)
    lo, hi = np.percentile(corrs, [2.5, 97.5])
    return float(lo), float(hi)


def build_property_test(
    property_name: str,
    items: Sequence[CalibrationItem],
    method: Literal["pearson", "spearman"] = "spearman",
    n_boot: int = 500,
    rng_seed: int = 0,
    corpus_version: str = "unversioned",
) -> NamedPropertyTest:
    """Doc §16.1: Run a structural/functional Mantel test for a specific named
    property and compute the bootstrap CI for its correlation magnitude.
    """
    func_matrix = build_functional_dissimilarity_matrix(items)
    inv_matrix = build_invariant_distance_matrix(items)

    mantel = mantel_test(
        func_matrix, inv_matrix, method=method, rng_seed=rng_seed,
        corpus_version=corpus_version,
    )

    ci = mantel_bootstrap_interval(
        func_matrix, inv_matrix, method=method, n_boot=n_boot, rng_seed=rng_seed + 1
    )

    # Attach the explicit CI directly to the Mantel confidence block
    mantel.confidence.interval = ci

    return NamedPropertyTest(
        property_name=property_name,
        mantel_result=mantel,
        correlation_ci=ci,
    )


def mantel_significant(
    test: NamedPropertyTest, alpha: float = 0.05, min_correlation: float = 0.3
) -> bool:
    """True if the test passes both the significance threshold (p_value <= alpha)
    and the minimum effect size (correlation >= min_correlation).
    """
    return (test.mantel_result.p_value <= alpha) and (test.mantel_result.correlation >= min_correlation)


def orphan_confidence_from_tests(
    tests: Sequence[NamedPropertyTest],
    min_correlation: float = 0.3,
    corpus_version: str = "unversioned"
) -> ConfidenceBlock:
    """Doc §16.3: Equivalence testing for orphan status.
    
    A class is a confident orphan if its correlation with ALL tested properties 
    is demonstrably below `min_correlation`. We check the upper bound of each
    property's bootstrap CI. If the test lacked statistical power (n too small), 
    the CI will be wide, the upper bound will exceed `min_correlation`, and the 
    orphan confidence will rightly drop to 0.
    """
    if not tests:
        # No properties tested -> zero confidence it's an orphan (just untested)
        return ConfidenceBlock(n=0, method="bootstrap", score=0.0, interval=None, corpus_version=corpus_version)

    max_hi = max(t.correlation_ci[1] for t in tests)
    min_hi = min(t.correlation_ci[1] for t in tests)

    if min_correlation > 0:
        # Score scales from 1.0 (max_hi <= 0) down to 0.0 (max_hi >= min_correlation)
        score = max(0.0, 1.0 - max(0.0, max_hi) / min_correlation)
    else:
        score = 0.0 if max_hi > 0 else 1.0

    n_items_min = min(t.mantel_result.n_items for t in tests)

    return ConfidenceBlock(
        n=n_items_min,
        method="bootstrap",
        score=float(score),
        interval=(float(min_hi), float(max_hi)),
        corpus_version=corpus_version,
        stale=False,
    )


def update_semantic_status(
    letter_class: LetterClass,
    tests: Sequence[NamedPropertyTest],
    alpha: float = 0.05,
    min_correlation: float = 0.3,
    corpus_version: str = "unversioned"
) -> LetterClass:
    """Doc §16.1/§16.3: Fold step 5's property testing results into the
    LetterClass's semantic status. Resolves whether it has named correlates 
    or remains a confident orphan.
    """
    significant_tests = [t for t in tests if mantel_significant(t, alpha, min_correlation)]

    correlates = [
        NamedCorrelate(
            property=t.property_name,
            correlation_strength=t.mantel_result.correlation,
            confidence=t.mantel_result.confidence
        )
        for t in significant_tests
    ]

    is_orphan = len(correlates) == 0
    orphan_conf = orphan_confidence_from_tests(tests, min_correlation, corpus_version) if is_orphan else None

    letter_class.semantic_status = SemanticStatus(
        named_correlates=correlates,
        orphan=is_orphan,
        orphan_confidence=orphan_conf
    )
    return letter_class