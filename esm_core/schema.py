"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
schema.py
=========
Dataclasses mirroring the theory document's final schema block ("Переиспользуемый
блок уверенности" onward) field-for-field, so pipeline output is directly
schema-compliant rather than an ad-hoc dict.

Scope note: this module defines the *whole* documented schema (letter_class,
environment_response_profile, topological_skeleton, realization_registry,
axis_annotations, and the corpus-level entries) because the doc specifies them
as one interlocking structure and getting the shape wrong would make the
letter-decoding output schema-incompatible with the rest of the program.
The `letter_class`-producing path (doc §15 steps 1-3) is backed by working
code in critical_points.py / invariants.py / corpus.py;
`letter_class.semantic_status` specifically is
populated by `semantics.py` (§16.1/§16.3, folding step 5's calibration
results in); and `axis_annotations` (§11) is populated by `commutation.py`.
The remaining containers (environment response, topological skeleton,
realization registry, corpus-level co-occurrence/forbidden-map/
transition-model/positional-sensitivity) are later-phase concerns (doc §13,
§17, Phases 7-11) and are included here as correctly-shaped, empty-by-default
containers so a `LetterClass` produced today has a stable place to
accumulate that data later -- they are not populated by this scaffold.

`HypothesisStatus` (§3.1/§3.2) is the one exception to "not populated by this
scaffold" among the later additions: it IS populated, by
`distributional_test.py`. It is the doc's `word_class.hypothesis_status`
schema block lifted out on its own, without the rest of `word_class`
(`constituent_letters`, `connectivity_topology`, `paradigm`) -- building
those requires a design decision this scaffold does not make (deriving
`slot_role` from `connectivity_topology.typical_graph` automorphism orbits),
so `distributional_test.py`'s `WordCandidateHypothesisResult` carries a
`HypothesisStatus` as the field that will slot unchanged into a future
`word_class.hypothesis_status` once that larger feature exists, without
itself being a `word_class`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

ConfidenceMethod = Literal[
    "bootstrap", "permutation", "held-out", "analytic_CI", "expert_provisional"
]


@dataclass
class ConfidenceBlock:
    """Reusable confidence block. Attached to every *derived* (non-raw) field.

    n              -- sample size / number of observations behind the estimate
    method         -- how it was computed
    score          -- stability score in [0,1], OR a p-value / CI depending on `method`
    interval       -- (low, high) bound, if applicable
    corpus_version -- version tag of the corpus used; the doc is explicit that
                       conclusions must be re-evaluated as the corpus grows
    stale          -- True once corpus_version is known to be behind the
                       current corpus. This is a flag to recompute, not to discard.
    """

    n: int
    method: ConfidenceMethod
    score: float
    interval: Optional[tuple] = None
    corpus_version: str = "unversioned"
    stale: bool = False


@dataclass
class DiscoveryProvenance:
    """§15 step 2 / step 3, extended per §3.2: the hyperparameters AND the
    partition key that materially change which candidates crystallize into
    a class, so every class records what it was discovered under.

    persistence_threshold_used -- segmentation threshold at class formation.
    l_max_bispectrum           -- spherical-harmonic order the class converged at.
    critical_point_type        -- §3.1/§3.2: the `critical_points.CriticalPoint
                                   .field_type` this class was clustered
                                   within (e.g. "nuclear_attractor",
                                   "bond_critical_point"). §3.2 is explicit
                                   that pooling different critical-point
                                   types into one flat alphabet before
                                   clustering would make the distributional
                                   test distinguishing Hypothesis A from
                                   Hypothesis B (§3.1) impossible to run --
                                   corpus.cluster_candidates_bootstrap never
                                   does that pooling, and this field records
                                   which type-partition a given class came
                                   from.
    """

    persistence_threshold_used: float
    l_max_bispectrum: int
    critical_point_type: str


@dataclass
class NamedCorrelate:
    """One external-property correlation for a letter class (§16.1-16.3):
    each correlation carries its own confidence, not a blanket one for the class."""

    property: str  # e.g. "pKa", "binding_affinity", "activation_energy"
    correlation_strength: float
    confidence: ConfidenceBlock


@dataclass
class SemanticStatus:
    """§16.3: whether a structurally stable class correlates with anything
    already named, or is an 'orphan' -- a candidate for undescribed function."""

    named_correlates: list = field(default_factory=list)  # list[NamedCorrelate]
    orphan: bool = True
    orphan_confidence: Optional[ConfidenceBlock] = None


HypothesisDecision = Literal[
    "hypothesis_B_consistent", "hypothesis_A_consistent", "indeterminate"
]


@dataclass
class HypothesisStatus:
    """§3.1/§3.2, doc's revised final-schema block (`word_class.hypothesis_status`):
    per-candidate-word verdict of the distributional test that distinguishes
    Hypothesis A (flat alphabet -- every critical-point type is an
    independent "letter") from Hypothesis B (only nuclear-attractor basins
    are letters; bond/ring/cage critical points are word-level relations
    between them, expected to show "obligatory pairing" with a fixed,
    predictable set of attractor letters rather than combining freely).

    Built by `distributional_test.evaluate_hypothesis_for_class` -- see that
    module for the actual test (which this scaffold's `corpus.py` and
    `commutation.py` explicitly did not implement; only the type-partitioning
    precondition the test depends on, §3.2's "practical consequence", was
    already in place in `corpus.py`).

    obligatory_pairing_score -- deliberately NOT a number computed
                                 independently for this field. Per the doc's
                                 explicit correction: storing a separately
                                 computed copy would silently diverge from
                                 the corpus-level `co_occurrence_matrix`
                                 (level="letter") as the corpus grows
                                 (`corpus_version`/`stale` would then
                                 disagree between the two, with no way to
                                 tell which to trust). This field instead
                                 IS one of those `CooccurrenceEntry.confidence`
                                 blocks -- specifically the entry (or, when a
                                 word's dominant attractor tuple has more
                                 than 2 members, the weakest of several
                                 entries) for the pair(s) of attractor letter
                                 classes this word's members were most
                                 commonly linked to. `None` when there was
                                 not enough data to identify a dominant
                                 attractor-letter pairing at all (see
                                 `decision == "indeterminate"`).
    decision                  -- "hypothesis_B_consistent" when this word
                                 candidate's attractor-letter pairing is
                                 both concentrated (within-class) and
                                 significant (corpus-wide PPMI) beyond what a
                                 matched permutation null would produce;
                                 "hypothesis_A_consistent" when neither
                                 signal clears its null (free, variable
                                 combination, as Hypothesis A predicts);
                                 "indeterminate" when there isn't enough
                                 data, or the two signals disagree -- an
                                 honest non-answer rather than a forced
                                 choice, per §3.1's own caution that
                                 intermediate outcomes (or outcomes that
                                 differ molecule-class by molecule-class) are
                                 explicitly not ruled out by the theory.
    """

    obligatory_pairing_score: Optional[ConfidenceBlock] = None
    decision: HypothesisDecision = "indeterminate"


@dataclass
class LetterClass:
    """The core output of letter decoding (doc's final schema, `letter_class:` block).

    class_id            -- stable identifier for this equivalence class ("letter")
    discovery_provenance-- hyperparameters the class was discovered under
    invariant_centroid  -- fixed-length rotation-invariant descriptor vector
                            (raw -- doesn't need its own confidence)
    cluster_stability   -- confidence that this class survives resampling of
                            the corpus (bootstrap, §15 step 3)
    semantic_status      -- correlation with known properties, or orphan status
    """

    class_id: str
    discovery_provenance: DiscoveryProvenance
    invariant_centroid: np.ndarray
    cluster_stability: ConfidenceBlock
    semantic_status: SemanticStatus = field(default_factory=SemanticStatus)


# ---------------------------------------------------------------------------
# Later-phase schema containers (§11, §13, §16, Phases 6-10).
# Correctly shaped, NOT populated by this scaffold -- see module docstring.
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentResponseSample:
    condition: str  # e.g. "vacuum", "water(eps=80)", "enzyme_pocket"
    eps_or_explicit_residues: str
    invariant_vector: Optional[np.ndarray]
    dense_tensor_ref: Optional[str]
    level_of_theory: str

@dataclass
class ResponseSmoothness:
    """§4.3, §13.4: Nested block tracking indexicality vs polysemy."""
    derivative_profile: list[float]  # Step differences
    monotonicity_score: float
    classification: Literal["indexical", "discontinuous_polysemy", "insufficient_data"]
    classification_confidence: ConfidenceBlock


@dataclass
class EnvironmentResponseProfile:
    """§4.3, §13.4: response of a class's realization to the indexical
    environment parameter. `response_smoothness.classification` distinguishes
    a smooth indexical shift from a discontinuous jump (= a different 'word',
    not a modulation).

    `response_smoothness` is nested (not flattened onto this dataclass)
    because that is exactly the doc's own schema shape --
    `environment_response_profile: {class_id, samples, response_smoothness:
    {...}}` -- and because it is a single derived unit (built together by
    `environment_response.analyze_response_smoothness` from `samples`) that
    should be replaced atomically, not four separately-settable fields that
    can drift out of sync with each other. `None` until
    `environment_response.update_environment_response_profile` has been run
    on this profile's `samples`.
    """

    class_id: str
    samples: list = field(default_factory=list)  # list[EnvironmentResponseSample]
    response_smoothness: Optional[ResponseSmoothness] = None


@dataclass
class CriticalPointRecord:
    position: np.ndarray
    field_type: str
    poincare_hopf_index: int
    value: float
    basis_set_convergence: Optional[ConfidenceBlock] = None


@dataclass
class TopologicalSkeleton:
    """§15 extraction precaution: per-environment topology of a class, kept
    separate from the (cheaper) geometry level of theory."""

    class_id: str
    per_environment_condition: list = field(default_factory=list)
    # each entry: {"condition": str, "critical_points": list[CriticalPointRecord],
    #              "morse_smale_graph_edges": list}


@dataclass
class RealizationExemplar:
    atomic_structure: str  # e.g. SMILES / structure ref
    source_molecule_id: str
    environment_context: str
    distance_to_centroid: float
    barrier_to_decay: Optional[float]
    synthetic_accessibility: Optional[float]
    stability_confidence: Optional[ConfidenceBlock]
    grammaticality_status: Literal[
        "attested", "combinatorially_valid_untested", "physically_forbidden"
    ]
    grammaticality_confidence: Optional[ConfidenceBlock]


@dataclass
class RealizationRegistry:
    """§13: the 'phonetics' layer -- concrete atomic realizations of an
    abstract class, needed for generation, not just recognition."""

    class_id: str
    exemplars: list = field(default_factory=list)  # list[RealizationExemplar]


@dataclass
class AxisAnnotation:
    axis_index: int
    continuous_sensitivity: Optional[ConfidenceBlock]
    categorical_sensitivity: Optional[ConfidenceBlock]
    commutation_test_result: Literal[
        "significant", "silent_in_tested_range", "class_boundary_marker", "indeterminate"
    ]
    result_confidence: Optional[ConfidenceBlock]
    tested_context_range: Optional[tuple]


@dataclass
class AxisAnnotations:
    """§11: which descriptor axes inside a class are meaningful (phonemic)
    vs silent (allophonic). Produced by the commutation test, §11.3/§11.7 --
    not implemented in this scaffold (see commutation.py stub)."""

    class_id: str
    axes: list = field(default_factory=list)  # list[AxisAnnotation]


# --- corpus-level (§15 steps 6-11) -----------------------------------------


@dataclass
class CooccurrenceEntry:
    """§15 step 6 co-occurrence entry, extended per the doc's closing revision
    block ("co_occurrence_matrix, forbidden_map, transition_model: level:
    'letter' | 'word'"):

    level                       -- which tier of unit `class_id_pair` refers
                                    to. "letter" (default) -- the only tier
                                    populated by this scaffold today -- means
                                    both ids are `LetterClass.class_id`s.
                                    "word" is reserved for once `word_class`
                                    exists (a separate, larger feature than
                                    this module builds -- see
                                    `distributional_test.py`'s module
                                    docstring); nothing in this codebase
                                    currently produces a "word"-level entry.
    mediating_critical_point_type -- §3.1/§3.2: which `critical_points.
                                    CriticalPoint.field_type` (e.g.
                                    "bond_critical_point",
                                    "ring_critical_point") the co-occurrence
                                    was measured THROUGH, when `level ==
                                    "letter"` and both ids are
                                    nuclear-attractor classes linked by some
                                    bond/ring/cage candidate. Not part of the
                                    doc's literal schema text, but required
                                    to keep the same §3.2 partition
                                    discipline `corpus.py` already enforces
                                    at clustering time: a letter-level
                                    co-occurrence entry mixing evidence from
                                    bond-mediated and ring-mediated links
                                    would silently blur exactly the
                                    distinction the distributional test is
                                    for. `None` for co-occurrence not
                                    produced by that pathway (e.g. a future
                                    "word"-level entry, or spatial-proximity
                                    co-occurrence unrelated to a specific
                                    mediating critical point).
    """

    class_id_pair: tuple
    ppmi_value: float
    confidence: ConfidenceBlock
    level: Literal["letter", "word"] = "letter"
    mediating_critical_point_type: Optional[str] = None


@dataclass
class ForbiddenMapEntry:
    class_id_pair_or_path: tuple
    forbidden_type: Literal["combinatorial", "geometric", "orbital", "conditional_kinetic"]
    evidence_strength: ConfidenceBlock
    replicated: bool


@dataclass
class TransitionModel:
    """§9.2 Markov State Model: alphabet + first-order grammar."""

    states: list = field(default_factory=list)  # list[class_id]
    transition_probabilities: list = field(default_factory=list)
    # each entry: {"from": class_id, "to": class_id, "probability": float,
    #              "confidence": ConfidenceBlock}
    condition: str = "unspecified"


# ---------------------------------------------------------------------------
# §19: ExpectationCheck -- the falsification-profile framework.
# Populated by expectation_profiles.py.
# ---------------------------------------------------------------------------

TestMethod = Literal[
    "dip_test", "AIC_comparison", "KS", "chi_square", "permutation",
    "trend_test", "range_check", "correlation",
]

ExpectationVerdict = Literal["supports_theory", "against_theory", "inconclusive"]


@dataclass
class ExpectationCheck:
    """§19.1: the reusable comparison unit -- NOT a single number. Doc is
    explicit that `p_value` ("does observed match expected -- a claim about
    the HYPOTHESIS") and `confidence` ("how reliably was observed_shape
    itself measured -- a claim about the MEASUREMENT") must stay two
    separate fields; collapsing them loses exactly the distinction §19.1
    exists to make. `verdict` is this module's own explicit, pre-declared
    reading of the two (see each builder in expectation_profiles.py for its
    stated pass/fail band) -- never silently reinterpreted post hoc
    (§15 step 5 / §19.10's own caution against exactly that).
    """

    expected_shape: str
    observed_shape: str
    test_method: TestMethod
    p_value: Optional[float]
    confidence: Optional[ConfidenceBlock]
    corpus_version: str = "unversioned"
    verdict: ExpectationVerdict = "inconclusive"


# ---------------------------------------------------------------------------
# §4.4-4.6: Paradigm (declension) vs idiosyncratic synonymy.
# Populated by paradigm.py.
# ---------------------------------------------------------------------------

PairRelationKind = Literal[
    "declension_paradigm_member", "idiosyncratic_synonym", "insufficient_evidence"
]


@dataclass
class Paradigm:
    """§4.6: a group of `class_id`s linked by ONE reusable transformation
    rule along a common varying axis (protonation state, isotope,
    oxidation state, conformer, ...) -- the doc's missing grouping layer
    above `RealizationRegistry`'s flat exemplar list (§4.6: word-forms of
    one lexeme are not interchangeable "pronunciations" of the same
    content, unlike true allophonic variation within a class).

    transformation_rule -- mean normalized displacement direction (unit
        vector, invariant-descriptor space) shared by every consecutive
        member pair along `class_ids`' declared order. Reusability (§4.5:
        "правило переиспользуемо... а не специфично для одной пары") is
        what `rule_reusability` actually certifies -- see paradigm.py.
    varying_axis -- caller-supplied semantic label for what the rule
        physically varies (this scaffold has no chemistry-aware way to
        derive the label itself from the displacement direction alone).
    """

    paradigm_id: str
    class_ids: list  # ordered list[str], the paradigm's declension sequence
    varying_axis: str
    transformation_rule: np.ndarray
    rule_reusability: ConfidenceBlock


@dataclass
class SynonymRelation:
    """§4.4/§4.5/§19.8: verdict for one pair of functionally-close classes
    -- declension-paradigm member (reusable rule, §4.5) vs idiosyncratic
    synonymy (§4.1(1), no generalizing rule) vs not enough evidence either
    way. `graph_distance` is, like `isostere.CalibrationPair.category`,
    assigned a priori from known bond-graph similarity (e.g. Tanimoto on
    Morgan fingerprints) -- never derived from the invariant descriptor
    itself, for the same reason isostere.py gives (testing the descriptor
    against a similarity measure derived from the descriptor would be
    circular). `None` when the caller has no graph-similarity source for
    this pair yet.
    """

    class_id_pair: tuple
    graph_distance: Optional[float]
    function_distance: Optional[float]
    invariant_distance: float
    classification: PairRelationKind


# ---------------------------------------------------------------------------
# §17: positional sensitivity to late perturbation (hot spots / bifurcation
# points). Populated by positional_sensitivity.py.
# ---------------------------------------------------------------------------

PositionClassification = Literal[
    "canalized_background", "hot_spot", "bifurcation_point", "insufficient_data"
]


@dataclass
class PositionSensitivity:
    """§17.1-17.4: one position in a build-order sequence (e.g. the k-th
    substituent added to a scaffold across a matched series of molecules).

    normalized_sensitivity -- score = mean |outcome change| at this
        position, normalized by accumulated context volume (§17.2's
        canalization) so a late position isn't automatically flagged
        "hot" merely because canalization has made most late positions
        quiet -- interval is the matched-null band from position-label
        permutation (same "beat a matched null" idiom as commutation.py).
    classification -- "canalized_background" (typical, low, matches most
        positions -- §17.1/17.2), "hot_spot" (significant effect at an
        otherwise-canalized, late position -- §17.1), "bifurcation_point"
        (significant effect concentrated at a position where the
        downstream outcome distribution is itself multimodal, i.e. the
        position sits at a genuine unresolved fork rather than a
        structured but single context -- §17.3), or "insufficient_data".
    """

    position_index: int
    normalized_sensitivity: ConfidenceBlock
    classification: PositionClassification
    downstream_multimodality_p: Optional[float] = None


@dataclass
class PositionalSensitivityProfile:
    """§17.4: one matched series' full positional-sensitivity map."""

    series_id: str
    positions: list  # list[PositionSensitivity]
    canalization_curve: list  # list[float], one accumulated-context-volume value per position
