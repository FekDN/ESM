"""
Energy Semiotics of Molecules -- letter-decoding scaffold.

Implements doc §15 steps 1-3 (legacy candidate "letter" extraction:
topological critical points, rotation-invariant descriptors, and clustering
into named, confidence-scored letter classes), plus a separate field-native
multi-scale region proposal archive that does not use atoms, atomic basins,
or critical-point seeds. The two proposal families are retained side by side
until a field-region alphabet can be validated and integrated into the
downstream class/visualization schema. Also implements step 4 (commutation
test, §11: which descriptor axes inside an already-found class are
signifying vs silent), and step 5 (isostere/Mantel calibration). See each
module's docstring for exactly
what is and isn't implemented, and schema.py for the full documented data
model this pipeline's output fits into.

Also implements §16.1/§16.3 (`semantics.py` -- folding step 5's calibration
results into a class's `semantic_status`, with an honest, equivalence-tested
`orphan_confidence` rather than plain non-significance), §16.2
(`unsupervised_validation.py` -- testing whether the invariant space is
more compressible, by §12's own criterion, than a scrambled/random-field
control of the same scale, requiring no new data beyond what the real
corpus already used), and §12/§15 step 10 (`fundamentality.py` -- the
compressibility/reuse-across-corpus criterion for telling a fundamental
"letter" apart from a derived combination, via a corpus-wide shared
dictionary rather than any per-parameter magnitude cutoff).

Steps 7-11 (dynamics/trajectory analysis, §8-10, §17) are now partially in
scope: `paradigm.py` (§4.4-4.6, declension vs synonymy), `distributional_test.py`
(§3.1/§3.2, the letter-vs-wordform hypothesis test), `positional_sensitivity.py`
(§17, hot spots / bifurcation points), and `expectation_profiles.py` (§19, the
falsification-profile framework) are implemented and exported below -- see each
module's docstring for exactly what is and isn't covered. `corpus.py`'s
docstring still lists what remains genuinely unbuilt (the full `word_class`
schema block: `constituent_letters.slot_role`, `connectivity_topology`).
"""

from .fields import ScalarField, GaussianMixtureDensity, CubeFileField, ScrambledFieldControl
from .field_patterns import (
    FieldRegionCandidate,
    FieldPartWholeRelation,
    FieldPatternBatch,
    extract_field_pattern_candidates,
    write_field_pattern_archive,
)
from .critical_points import (
    CriticalPoint,
    find_critical_points,
    find_critical_points_blind,
    compare_seeded_vs_blind,
    classify,
    basis_convergence_discrepancy,
    check_basis_convergence,
)
from .invariants import InvariantDescriptor, build_invariant_descriptor
from .corpus import (
    LetterCandidate,
    assign_candidate_labels,
    cluster_candidates_bootstrap,
    dpmeans_incremental_cluster,
    select_threshold_by_mdl,
    cluster_labels,
    suggest_distance_threshold,
    class_members,
)
from .commutation import CommutationConfig, build_axis_annotations
from .semantics import (
    NamedPropertyTest,
    build_property_test,
    mantel_bootstrap_interval,
    mantel_significant,
    orphan_confidence_from_tests,
    update_semantic_status,
)
from .fundamentality import (
    DictionaryFit,
    FundamentalityConfig,
    FundamentalityResult,
    active_atoms,
    compute_fundamentality,
    fit_shared_dictionary,
    diagnose_dictionary_fit,
    select_dictionary_size_by_cv,
    reconstruction_fidelity,
    reuse_breadth,
    reuse_breadth_confidences,
    suggest_n_atoms,
)
from .unsupervised_validation import (
    build_control_corpus,
    compressibility_statistic,
    validate_structure_against_control,
)
from .isostere import (
    CalibrationItem,
    CalibrationPair,
    CalibrationDesign,
    SuccessCriteria,
    PairResult,
    MantelResult,
    CalibrationReport,
    build_calibration_design,
    evaluate_pairs,
    mantel_test,
    run_calibration,
    predict_functional_dissimilarity,
    to_named_correlate,
)
from .environment_response import analyze_response_smoothness, update_environment_response_profile
from .dimensionality import levina_bickel_intrinsic_dimension
from .paradigm import (
    ParadigmConfig,
    PairTransformation,
    pairwise_transformation,
    all_pairwise_transformations,
    group_transformations_by_direction,
    rule_reusability_confidence,
    detect_declension_paradigms,
    classify_pair,
    build_synonym_relations,
)
from .distributional_test import (
    HypothesisTestConfig,
    AttractorLink,
    WordCandidateHypothesisResult,
    link_candidates_to_attractors,
    build_letter_cooccurrence,
    evaluate_hypothesis_for_class,
    evaluate_all_word_candidates,
    run_distributional_test,
    summarize_hypothesis_evidence,
)
from .positional_sensitivity import (
    BuildStep,
    BuildTrajectory,
    PositionalSensitivityConfig,
    accumulated_context_volume,
    position_sensitivity,
    classify_position,
    positional_sensitivity_profile,
)
from .expectation_profiles import (
    build_letter_corpus_profile,
    build_word_corpus_profile,
    build_grammar_profile,
    build_word_meaning_profile,
    build_sentence_meaning_profile,
    build_alphabet_discovery_profile,
    build_synonym_profile,
    build_polysemy_profile,
)
from .pipeline import extract_letter_candidates, decode_alphabet
from .schema import (
    ConfidenceMethod,
    ConfidenceBlock,
    DiscoveryProvenance,
    NamedCorrelate,
    SemanticStatus,
    LetterClass,
    HypothesisDecision,
    HypothesisStatus,
    EnvironmentResponseSample,
    ResponseSmoothness,
    EnvironmentResponseProfile,
    CriticalPointRecord,
    TopologicalSkeleton,
    RealizationExemplar,
    RealizationRegistry,
    AxisAnnotation,
    AxisAnnotations,
    CooccurrenceEntry,
    ForbiddenMapEntry,
    TransitionModel,
    TestMethod,
    ExpectationVerdict,
    ExpectationCheck,
    PairRelationKind,
    Paradigm,
    SynonymRelation,
    PositionClassification,
    PositionSensitivity,
    PositionalSensitivityProfile,
)

__all__ = [
    # fields.py
    "ScalarField", "GaussianMixtureDensity", "CubeFileField", "ScrambledFieldControl",
    # field_patterns.py (field-only multi-scale proposals)
    "FieldRegionCandidate", "FieldPartWholeRelation", "FieldPatternBatch",
    "extract_field_pattern_candidates", "write_field_pattern_archive",
    # critical_points.py
    "CriticalPoint", "find_critical_points", "find_critical_points_blind", "compare_seeded_vs_blind", "classify", "basis_convergence_discrepancy", "check_basis_convergence",
    # invariants.py
    "InvariantDescriptor", "build_invariant_descriptor",
    # corpus.py
    "LetterCandidate", "assign_candidate_labels", "cluster_candidates_bootstrap", "dpmeans_incremental_cluster", "select_threshold_by_mdl", "cluster_labels", "suggest_distance_threshold", "class_members",
    # commutation.py
    "CommutationConfig", "build_axis_annotations",
    # semantics.py
    "NamedPropertyTest", "build_property_test", "mantel_bootstrap_interval", "mantel_significant", "orphan_confidence_from_tests", "update_semantic_status",
    # fundamentality.py
    "DictionaryFit", "FundamentalityConfig", "FundamentalityResult", "active_atoms", "compute_fundamentality", "fit_shared_dictionary", "diagnose_dictionary_fit", "select_dictionary_size_by_cv", "reconstruction_fidelity", "reuse_breadth", "reuse_breadth_confidences", "suggest_n_atoms",
    # unsupervised_validation.py
    "build_control_corpus", "compressibility_statistic", "validate_structure_against_control",
    # isostere.py
    "CalibrationItem", "CalibrationPair", "CalibrationDesign", "SuccessCriteria", "PairResult", "MantelResult", "CalibrationReport", "build_calibration_design", "evaluate_pairs", "mantel_test", "run_calibration", "predict_functional_dissimilarity", "to_named_correlate",
    # environment_response.py
    "analyze_response_smoothness", "update_environment_response_profile",
    # dimensionality.py
    "levina_bickel_intrinsic_dimension",
    # paradigm.py
    "ParadigmConfig", "PairTransformation", "pairwise_transformation", "all_pairwise_transformations", "group_transformations_by_direction", "rule_reusability_confidence", "detect_declension_paradigms", "classify_pair", "build_synonym_relations",
    # distributional_test.py
    "HypothesisTestConfig", "AttractorLink", "WordCandidateHypothesisResult", "link_candidates_to_attractors", "build_letter_cooccurrence", "evaluate_hypothesis_for_class", "evaluate_all_word_candidates", "run_distributional_test", "summarize_hypothesis_evidence",
    # positional_sensitivity.py
    "BuildStep", "BuildTrajectory", "PositionalSensitivityConfig", "accumulated_context_volume", "position_sensitivity", "classify_position", "positional_sensitivity_profile",
    # expectation_profiles.py
    "build_letter_corpus_profile", "build_word_corpus_profile", "build_grammar_profile", "build_word_meaning_profile", "build_sentence_meaning_profile", "build_alphabet_discovery_profile", "build_synonym_profile", "build_polysemy_profile",
    # pipeline.py
    "extract_letter_candidates", "decode_alphabet",
    # schema.py
    "ConfidenceMethod", "ConfidenceBlock", "DiscoveryProvenance", "NamedCorrelate", "SemanticStatus", "LetterClass",
    "HypothesisDecision", "HypothesisStatus",
    "EnvironmentResponseSample", "ResponseSmoothness", "EnvironmentResponseProfile", "CriticalPointRecord", "TopologicalSkeleton", "RealizationExemplar", "RealizationRegistry", "AxisAnnotation", "AxisAnnotations", "CooccurrenceEntry", "ForbiddenMapEntry", "TransitionModel",
    "TestMethod", "ExpectationVerdict", "ExpectationCheck",
    "PairRelationKind", "Paradigm", "SynonymRelation",
    "PositionClassification", "PositionSensitivity", "PositionalSensitivityProfile",
]
