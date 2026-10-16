"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
isostere.py
===========
Doc §15, step 5 ("Тест на изостерных парах — контролируемый дизайн, а не
единичная пара"): the first point in the program where the invariant
descriptor (invariants.py) is checked against something *external* --
independently measured functional data -- rather than only against itself
(rotation invariance) or against the corpus's own clustering (steps 1-4).
Step 4 (commutation.py) explicitly deferred this: "there is no
external, named property to test commutation against yet -- that only
enters at step 5". This module is that entry.

The doc is explicit that a single positive example (-COOH / tetrazole) is
not sufficient evidence: the design must cover all four combinations of
(bond-graph similarity, functional similarity), by analogy with Campbell &
Fiske's multitrait-multimethod matrix --

    dissimilar graph / similar function     -- the actual hypothesis test
                                                (positive case: does the
                                                descriptor see past a
                                                different atomic ensemble
                                                to a shared function?)
    similar graph / dissimilar function      -- MANDATORY negative control
                                                (e.g. a carboxylic acid vs.
                                                a structurally close amide
                                                must give differing, not
                                                converged, invariants -- or
                                                the method is only picking
                                                up trivial graph similarity,
                                                not physics)
    similar graph / similar function         -- calibration point
    dissimilar graph / dissimilar function   -- calibration point

Significance, for the small number of examples such a hand-built design
realistically has, is assessed with a Mantel test: the matrix of
independently known functional similarity (measured pKa, binding
constants, literature MMPA data) against the matrix of invariant-space
distances, with significance from *exact* permutation enumeration where
feasible (falling back to a large, explicitly-labeled Monte Carlo
permutation count otherwise) -- not an asymptotic normal-theory p-value,
per the doc's explicit instruction.

Success criteria (`SuccessCriteria`) are recorded on the design *before*
`run_calibration` is ever called, exactly to prevent the post-hoc
reinterpretation the doc warns against ("критерии успеха фиксируются до
вычислений во избежание постфактумной подгонки интерпретации").

Explicit scope limit carried through from the doc's own closing caveat on
this step: passing this calibration on already-known, textbook isosterism
is a necessary sanity check of the *pipeline*, not evidence that the
theory sees something a classical chemist's eye would miss. The
substantive test the doc actually asks for -- applying the calibrated
invariant to a held-out group not used for calibration, with an explicit
quantitative prediction checked against independent data afterward -- needs
data this scaffold has no source for. `predict_functional_dissimilarity`
produces the falsifiable number; checking it against reality is a step
this module cannot take for the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import factorial
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from .invariants import InvariantDescriptor
from .schema import ConfidenceBlock, NamedCorrelate


# --------------------------------------------------------------------------
# the 2x2 design (§15 step 5 / Campbell-Fiske)
# --------------------------------------------------------------------------

PairCategory = Literal[
    "dissimilar_graph_similar_function",
    "similar_graph_dissimilar_function",
    "similar_graph_similar_function",
    "dissimilar_graph_dissimilar_function",
]

_ALL_CATEGORIES: Tuple[PairCategory, ...] = (
    "dissimilar_graph_similar_function",
    "similar_graph_dissimilar_function",
    "similar_graph_similar_function",
    "dissimilar_graph_dissimilar_function",
)

# doc: only these two quadrants are mandatory ("test" and "control"); the
# other two are "additional calibration points" -- present but not gating.
_MANDATORY_CATEGORIES: Tuple[PairCategory, ...] = (
    "dissimilar_graph_similar_function",
    "similar_graph_dissimilar_function",
)


@dataclass
class CalibrationItem:
    """One molecular motif entering the calibration design -- e.g. the
    descriptor of one functional-group critical point (a LetterCandidate's
    `descriptor`, corpus.py) plus the independently measured functional
    property used as this design's ground truth (§15 step 5: "measured
    pKa, binding constants, literature MMPA data").

    All items in one `CalibrationDesign` must report `functional_property`
    on the *same* measured scale (e.g. all pKa, or all binding constants
    from the same assay) -- the Mantel test below compares raw differences
    of this field, which is only meaningful within one comparable series.
    A design mixing incommensurable properties should instead supply a
    precomputed functional dissimilarity matrix directly to `mantel_test`.
    """

    item_id: str
    descriptor: InvariantDescriptor
    functional_property: float
    property_name: str = "unspecified"
    source: str = "unspecified"
    label: str = ""

    def vector(self) -> np.ndarray:
        return self.descriptor.to_vector()


@dataclass
class CalibrationPair:
    """One entry of the 2x2 design: two items plus the quadrant this pair
    is meant to occupy. `category` is assigned by the experimenter *a
    priori*, from known chemistry (bond-graph identity/similarity) and
    known measured function -- never derived from the invariant descriptor
    itself, or the design would be testing itself circularly."""

    item_a: str
    item_b: str
    category: PairCategory
    note: str = ""


@dataclass
class SuccessCriteria:
    """§15 step 5: "критерии успеха фиксируются до вычислений во избежание
    постфактумной подгонки интерпретации" -- fixed once, before
    `run_calibration` is ever called, so results can't be reinterpreted
    after the fact. Every field here is a falsifiable claim, not a knob to
    retune once the numbers are in.

    `max_distance_for_similar_function` / `min_distance_for_dissimilar_function`
    are on the same invariant-descriptor Euclidean-distance scale as
    `InvariantDescriptor.to_vector()` distances -- set them from prior
    knowledge of that scale (e.g. `corpus.suggest_distance_threshold` on a
    representative pool), not from this design's own results.
    """

    max_distance_for_similar_function: float
    min_distance_for_dissimilar_function: float
    mantel_alpha: float = 0.05
    mantel_min_correlation: float = 0.3

    def __post_init__(self) -> None:
        if self.max_distance_for_similar_function >= self.min_distance_for_dissimilar_function:
            raise ValueError(
                "max_distance_for_similar_function must be strictly below "
                "min_distance_for_dissimilar_function, or the two criteria "
                "overlap and nothing could ever fail either test"
            )
        if not (0.0 < self.mantel_alpha < 1.0):
            raise ValueError("mantel_alpha must be in (0, 1)")


@dataclass
class CalibrationDesign:
    items: Dict[str, CalibrationItem]
    pairs: List[CalibrationPair]
    success_criteria: SuccessCriteria

    def category_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in _ALL_CATEGORIES}
        for p in self.pairs:
            counts[p.category] += 1
        return counts

    def missing_categories(self) -> List[str]:
        counts = self.category_counts()
        return [c for c in _ALL_CATEGORIES if counts[c] == 0]


def build_calibration_design(
    items: Sequence[CalibrationItem],
    pairs: Sequence[CalibrationPair],
    success_criteria: SuccessCriteria,
    require_full_coverage: bool = True,
) -> CalibrationDesign:
    """Assemble and validate a `CalibrationDesign`.

    `require_full_coverage=True` (default) enforces the doc's explicit
    requirement that all four quadrants of the 2x2 design be present --
    "одной положительной пары... недостаточно" ("one positive pair is not
    enough"). Set it False only to build a design incrementally in stages;
    `run_calibration` still expects the two mandatory quadrants (positive
    test, negative control) to be non-empty and will raise otherwise.
    """
    items_by_id = {it.item_id: it for it in items}
    if len(items_by_id) != len(items):
        raise ValueError("duplicate item_id among calibration items")

    for p in pairs:
        if p.item_a not in items_by_id:
            raise ValueError(f"pair references unknown item_a={p.item_a!r}")
        if p.item_b not in items_by_id:
            raise ValueError(f"pair references unknown item_b={p.item_b!r}")
        if p.item_a == p.item_b:
            raise ValueError(f"pair references the same item twice: {p.item_a!r}")

    design = CalibrationDesign(items=items_by_id, pairs=list(pairs), success_criteria=success_criteria)

    missing = design.missing_categories()
    if require_full_coverage and missing:
        raise ValueError(
            "calibration design is missing quadrant(s) "
            f"{missing!r} -- doc §15 step 5 requires all four combinations "
            "of graph/function similarity, not a single positive pair"
        )
    return design


# --------------------------------------------------------------------------
# per-pair evaluation
# --------------------------------------------------------------------------


@dataclass
class PairResult:
    item_a: str
    item_b: str
    category: PairCategory
    invariant_distance: float
    passed: Optional[bool]
    note: str = ""


def _invariant_distance(a: CalibrationItem, b: CalibrationItem) -> float:
    va, vb = a.vector(), b.vector()
    if va.shape != vb.shape:
        raise ValueError(
            f"descriptor vectors for {a.item_id!r} and {b.item_id!r} have "
            f"different shapes {va.shape} vs {vb.shape} -- they must be "
            "built with the same l_max / shell count to be comparable"
        )
    return float(np.linalg.norm(va - vb))


def evaluate_pairs(design: CalibrationDesign) -> List[PairResult]:
    """Distance + pass/fail for every pair in the design, against the
    design's pre-registered `SuccessCriteria`. The pass/fail rule is keyed
    on the *function* half of each category label, independent of the
    *graph* half -- this is deliberate, not a simplification: the whole
    point of §15 step 5 is that the descriptor should track function
    regardless of graph similarity, so a "similar_function" pair (whichever
    graph quadrant it's in) is expected to land close, and a
    "dissimilar_function" pair is expected to land far, uniformly across
    all four quadrants.
    """
    sc = design.success_criteria
    results = []
    for p in design.pairs:
        a, b = design.items[p.item_a], design.items[p.item_b]
        dist = _invariant_distance(a, b)
        if p.category.endswith("similar_function"):
            passed: Optional[bool] = dist <= sc.max_distance_for_similar_function
        else:
            passed = dist >= sc.min_distance_for_dissimilar_function
        results.append(
            PairResult(item_a=p.item_a, item_b=p.item_b, category=p.category, invariant_distance=dist, passed=passed, note=p.note)
        )
    return results


# --------------------------------------------------------------------------
# Mantel test: exact permutation where feasible, labeled Monte Carlo otherwise
# --------------------------------------------------------------------------


def _upper_tri(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def _correlate(a: np.ndarray, b: np.ndarray, method: Literal["pearson", "spearman"]) -> float:
    if method == "pearson":
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    else:
        stat = spearmanr(a, b).correlation
        return float(stat) if stat is not None and not np.isnan(stat) else 0.0


@dataclass
class MantelResult:
    correlation: float
    correlation_method: Literal["pearson", "spearman"]
    p_value: float
    alternative: Literal["greater", "two-sided", "less"]
    n_items: int
    n_permutations_used: int
    exact: bool
    confidence: ConfidenceBlock


def mantel_test(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    method: Literal["pearson", "spearman"] = "spearman",
    alternative: Literal["greater", "two-sided", "less"] = "greater",
    max_exact_permutations: int = 40320,  # 8! -- exact enumeration budget
    n_monte_carlo: int = 20000,
    rng_seed: int = 0,
    corpus_version: str = "unversioned",
) -> MantelResult:
    """Doc §15 step 5: "значимость оценивается тестом Мантеля... со
    значимостью, оцениваемой точным перебором перестановок, а не
    асимптотической статистикой" ("significance is assessed by Mantel's
    test... with significance assessed by exact permutation, not
    asymptotic statistics").

    `matrix_a`, `matrix_b` are square, symmetric, zero-diagonal matrices
    over the *same* ordered set of n items (e.g. functional dissimilarity
    and invariant-space distance respectively -- see
    `build_functional_dissimilarity_matrix` / `build_invariant_distance_matrix`).
    The Mantel statistic is the correlation between their upper-triangle
    entries; its null distribution is generated by permuting the row/column
    labeling of `matrix_b` (equivalent to relabeling which item is which)
    and recomputing the same correlation against the fixed `matrix_a`.

    `alternative="greater"` (default) tests the doc's actual hypothesis --
    that functional dissimilarity and invariant distance are *positively*
    associated (physically similar function -> small invariant distance) --
    rather than a generic two-sided test of any association.

    Exact enumeration (all n! permutations, including the identity, which
    is what makes the observed statistic itself) is used whenever n! does
    not exceed `max_exact_permutations`; `MantelResult.exact` reports
    honestly whether that happened. Above that, a Monte Carlo sample of
    `n_monte_carlo` random permutations is used instead (plus the identity,
    via the standard +1/+1 correction so p can never come out as exactly
    0), and `exact=False` -- this is the labeled approximation the doc's
    "not asymptotic statistics" instruction still permits: a permutation
    p-value from an incomplete but large, explicit permutation sample, as
    opposed to a closed-form Z/normal-theory formula.
    """
    matrix_a = np.asarray(matrix_a, dtype=float)
    matrix_b = np.asarray(matrix_b, dtype=float)
    n = matrix_a.shape[0]
    if matrix_a.shape != matrix_b.shape or matrix_a.shape[0] != matrix_a.shape[1]:
        raise ValueError("matrix_a and matrix_b must be square matrices of the same shape")
    if n < 3:
        raise ValueError("Mantel test needs at least 3 items (>=3 upper-triangle entries)")

    a_flat = _upper_tri(matrix_a)
    observed = _correlate(a_flat, _upper_tri(matrix_b), method)

    total_perms = factorial(n)
    exact = total_perms <= max_exact_permutations
    rng = np.random.default_rng(rng_seed)

    if exact:
        perm_stats = []
        for perm in permutations(range(n)):
            b_perm = matrix_b[np.ix_(perm, perm)]
            perm_stats.append(_correlate(a_flat, _upper_tri(b_perm), method))
        perm_stats = np.array(perm_stats)
        n_used = total_perms
        if alternative == "greater":
            p_value = float(np.mean(perm_stats >= observed))
        elif alternative == "less":
            p_value = float(np.mean(perm_stats <= observed))
        else:
            p_value = float(np.mean(np.abs(perm_stats) >= abs(observed)))
    else:
        perm_stats = np.empty(n_monte_carlo)
        for i in range(n_monte_carlo):
            perm = rng.permutation(n)
            b_perm = matrix_b[np.ix_(perm, perm)]
            perm_stats[i] = _correlate(a_flat, _upper_tri(b_perm), method)
        n_used = n_monte_carlo
        if alternative == "greater":
            count = int(np.sum(perm_stats >= observed))
        elif alternative == "less":
            count = int(np.sum(perm_stats <= observed))
        else:
            count = int(np.sum(np.abs(perm_stats) >= abs(observed)))
        # +1/+1 Monte Carlo correction: never report p=0 from a finite sample
        p_value = float((count + 1) / (n_monte_carlo + 1))

    confidence = ConfidenceBlock(
        n=n_used,
        method="permutation",
        score=p_value,
        interval=None,
        corpus_version=corpus_version,
        stale=False,
    )

    return MantelResult(
        correlation=observed,
        correlation_method=method,
        p_value=p_value,
        alternative=alternative,
        n_items=n,
        n_permutations_used=n_used,
        exact=exact,
        confidence=confidence,
    )


def build_functional_dissimilarity_matrix(items: Sequence[CalibrationItem]) -> np.ndarray:
    """n x n matrix of |functional_property_i - functional_property_j|.
    Only meaningful when every item's `functional_property` is on the same
    measured scale (see `CalibrationItem` docstring) -- for a design mixing
    incommensurable property types, build the matrix independently and
    pass it to `mantel_test` directly instead of using this helper."""
    values = np.array([it.functional_property for it in items], dtype=float)
    return np.abs(values[:, None] - values[None, :])


def build_invariant_distance_matrix(items: Sequence[CalibrationItem]) -> np.ndarray:
    """n x n matrix of Euclidean distances between descriptor vectors."""
    vectors = np.stack([it.vector() for it in items])
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(diff, axis=-1)


# --------------------------------------------------------------------------
# top-level report
# --------------------------------------------------------------------------

_CALIBRATION_CAVEAT = (
    "Doc \u00a715 step 5: passing this calibration on already-known, textbook "
    "isosterism is a necessary sanity check of the pipeline, not evidence "
    "the theory sees something a classical chemist's eye would miss. "
    "Substantive validation requires applying the calibrated invariant to a "
    "group NOT used in this design, with an explicit quantitative "
    "prediction (see predict_functional_dissimilarity) checked against "
    "independent data afterward -- that check is outside this module's "
    "scope."
)


@dataclass
class CalibrationReport:
    design: CalibrationDesign
    pair_results: List[PairResult]
    mantel: MantelResult
    positive_test_passed: Optional[bool]
    negative_control_passed: Optional[bool]
    overall_passed: bool
    caveat: str = _CALIBRATION_CAVEAT


def _category_passed(pair_results: Sequence[PairResult], category: PairCategory) -> Optional[bool]:
    relevant = [r for r in pair_results if r.category == category]
    if not relevant:
        return None
    return all(r.passed for r in relevant)


def run_calibration(
    design: CalibrationDesign,
    mantel_method: Literal["pearson", "spearman"] = "spearman",
    max_exact_permutations: int = 40320,
    n_monte_carlo: int = 20000,
    rng_seed: int = 0,
) -> CalibrationReport:
    """Doc §15 step 5, full pipeline: per-pair threshold checks against the
    design's pre-registered `SuccessCriteria`, plus the Mantel test of the
    whole item set's functional-dissimilarity matrix against its
    invariant-distance matrix. `overall_passed` requires both mandatory
    quadrants (positive test, negative control) to pass AND the Mantel
    correlation to clear both its significance (`mantel_alpha`) and
    minimum-effect-size (`mantel_min_correlation`) pre-registered bars --
    the two "additional calibration point" quadrants are still scored and
    reported (`pair_results`), but per the doc's own wording they are not
    gating.
    """
    if design.missing_categories():
        raise ValueError(
            f"design is missing quadrant(s) {design.missing_categories()!r}; "
            "cannot run a §15 step 5 calibration without full 2x2 coverage"
        )

    pair_results = evaluate_pairs(design)

    items = list(design.items.values())
    func_matrix = build_functional_dissimilarity_matrix(items)
    inv_matrix = build_invariant_distance_matrix(items)
    mantel = mantel_test(
        func_matrix,
        inv_matrix,
        method=mantel_method,
        alternative="greater",
        max_exact_permutations=max_exact_permutations,
        n_monte_carlo=n_monte_carlo,
        rng_seed=rng_seed,
    )

    positive_passed = _category_passed(pair_results, "dissimilar_graph_similar_function")
    negative_passed = _category_passed(pair_results, "similar_graph_dissimilar_function")

    mantel_ok = (mantel.p_value <= design.success_criteria.mantel_alpha) and (
        mantel.correlation >= design.success_criteria.mantel_min_correlation
    )
    overall = bool(positive_passed) and bool(negative_passed) and mantel_ok

    return CalibrationReport(
        design=design,
        pair_results=pair_results,
        mantel=mantel,
        positive_test_passed=positive_passed,
        negative_control_passed=negative_passed,
        overall_passed=overall,
    )


# --------------------------------------------------------------------------
# held-out prediction (§15 step 5 closing requirement) and schema bridge
# --------------------------------------------------------------------------


def predict_functional_dissimilarity(design: CalibrationDesign, query_distance: float) -> float:
    """Monotonic (isotonic) fit of invariant distance -> functional
    dissimilarity from the calibration design's own pairs, applied to a
    NEW distance value from a pair NOT in the calibration set. This is the
    "явное количественное предсказание, сверяемое впоследствии с
    независимыми данными" the doc's closing sentence on step 5 requires --
    the number this function returns is the falsifiable prediction;
    checking it against reality is outside this module's scope.

    Isotonic (monotone non-decreasing) regression is used rather than a
    linear fit because the only assumption the theory actually licenses is
    the direction of the relationship (larger invariant distance -> at
    least as large functional dissimilarity), not linearity -- and the
    calibration design is, by construction, a handful of points (§15 step
    5's "малое число примеров"), too few to justify a stronger functional
    form.
    """
    from sklearn.isotonic import IsotonicRegression

    pair_results = evaluate_pairs(design)
    if len(pair_results) < 2:
        raise ValueError("need at least 2 calibration pairs to fit a distance -> dissimilarity mapping")

    distances = np.array([r.invariant_distance for r in pair_results])
    dissimilarities = np.array(
        [
            abs(design.items[r.item_a].functional_property - design.items[r.item_b].functional_property)
            for r in pair_results
        ]
    )

    model = IsotonicRegression(out_of_bounds="clip", increasing=True)
    model.fit(distances, dissimilarities)
    return float(model.predict([query_distance])[0])


def to_named_correlate(mantel: MantelResult, property_name: str) -> NamedCorrelate:
    """Bridge back into the main schema: a §15 step 5 Mantel result becomes one
    entry of `schema.SemanticStatus.named_correlates` for the letter class
    the calibration items were drawn from -- `correlation_strength` is the
    raw Mantel correlation, `confidence` carries the permutation p-value
    (see `mantel_test`'s ConfidenceBlock)."""
    return NamedCorrelate(property=property_name, correlation_strength=mantel.correlation, confidence=mantel.confidence)
