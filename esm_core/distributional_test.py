"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
distributional_test.py
=======================
Doc §3.1/§3.2 ("Открытый вопрос: что физически соответствует «букве»" /
"Как гипотезы фальсифицируемы"): the actual distributional test that
decides, per candidate bond/ring/cage `LetterClass`, whether it behaves as
Hypothesis A predicts (an independent "letter", freely combining with a
wide and variable set of neighbouring attractor basins) or as Hypothesis B
predicts (not a letter at all, but a word-level *relation* that shows
"obligatory pairing" -- it almost never occurs outside a fixed, predictable
connection to specific attractor basins).

Audit context (why this module exists): `corpus.py` already implements
§3.2's *precondition* correctly -- candidates are partitioned by
`critical_point.field_type` before any clustering, so nuclear-attractor
candidates never compete for cluster membership against bond/ring/cage
candidates (see `corpus._partition_by_field_type` and its docstring). What
neither `corpus.py` nor `commutation.py` implements is the test itself: the
Ventris-method distributional grid (doc §6.1, explicitly invoked by name in
§3.2) that would show whether a given bond/ring/cage class's neighbours are
"free and variable" or "fixed and predictable". `commutation.py` is a
different, complementary check one level *below* this one -- it asks which
axes *inside* an already-found class carry meaning (§11), not whether the
class itself is a letter or a relation between letters (§3.1). And the
`co_occurrence_matrix` built by the external `esm_corpus_miner.py` script is
a coarse spatial-distance-binned tally over *any* pair of classes -- useful
corpus telemetry, but not partitioned by mediating critical-point type and
not read by any hypothesis test, so it cannot answer §3.2's question either.

Scope, explicitly: this module implements the *check* -- it does not build
the doc's `word_class` schema block in full. `word_class` also carries
`constituent_letters.slots[...].slot_role` (to be derived from
`connectivity_topology.typical_graph` automorphism orbits) and `paradigm`
(§4.4-4.6, declension vs synonymy) -- both are a separate, larger feature
requiring their own design decisions (chiefly: what counts as one instance
of "the typical graph" for a ring/cage word candidate, and how automorphism
orbits of that graph assign `slot_role` when two structurally-symmetric
positions are in practice filled by different attractor letter classes
across instances). This module produces exactly what §3.1/§3.2 asks for --
a falsifiable, per-word-candidate hypothesis verdict -- via
`schema.HypothesisStatus`, the doc's schema block for that verdict, so it
slots unchanged into `word_class.hypothesis_status` whenever that larger
feature is built.

Pipeline, mirroring the doc's own two-part framing of §3.2:
    1. `link_candidates_to_attractors` -- for every bond/ring/cage
       candidate, identify which nuclear-attractor LETTER CLASSES it
       connects, via nearest-position lookup within the same molecule (the
       critical-point-finder's own bond/ring search already seeds these
       candidates near specific attractor pairs/triples -- see
       `critical_points.find_critical_points`'s Stage 2 -- so "nearest
       attractors by position" recovers that same correspondence without
       needing the finder to have recorded it explicitly).
    2. `build_letter_cooccurrence` -- the doc §6.1 "positional grid" made
       concrete: a PPMI co-occurrence table over PAIRS of attractor letter
       classes, built *separately per mediating critical-point type* (never
       pooling bond-mediated and ring-mediated evidence -- the same §3.2
       partition discipline `corpus.py` already enforces at clustering
       time), with significance from a label-permutation null (the same
       "beat a matched null, not an arbitrary constant" idiom used
       throughout this codebase -- `commutation.py`'s random-direction
       null, `fundamentality.py`'s molecule-label permutation null,
       `unsupervised_validation.py`'s scrambled-field null).
    3. `evaluate_hypothesis_for_class` -- per bond/ring/cage `LetterClass`,
       two independent significance tests, both required to agree before
       a firm verdict is returned (an honest "indeterminate" when they
       don't, rather than picking one arbitrarily):
         - within-class CONCENTRATION: of this class's own members, what
           fraction connect the exact same attractor-letter tuple, versus a
           matched-size random subsample of every same-type candidate in
           the corpus? High, significant concentration = this "letter" is
           not freely combining -- direct evidence against Hypothesis A.
         - corpus-wide PAIRING STRENGTH: is that dominant attractor tuple's
           `co_occurrence_matrix` entry (step 2) itself significant, i.e.
           read directly off the table built in step 2, never recomputed
           independently (see `schema.HypothesisStatus.obligatory_pairing_score`'s
           docstring for why duplicating this number would be a staleness
           bug waiting to happen).
    4. `evaluate_all_word_candidates` -- runs 1-3 across every discovered
       bond/ring/cage class in a corpus at once, reusing the attractor
       linking and per-type co-occurrence tables (each built once, not once
       per class).

What this module deliberately does NOT decide: one global "Hypothesis A or
B" verdict for the whole corpus. §3.1 explicitly keeps that door open --
"любые промежуточными вариантами, включая возможность, что ни одна не
верна для всех классов молекул одновременно" ("...including the
possibility that neither holds for all classes of molecules
simultaneously"). `summarize_hypothesis_evidence` reports the per-type
tally of verdicts and stops there; collapsing that into one yes/no answer
is a claim the theory itself declines to make in general, and is left to
the caller to make explicitly, with their own stated reasoning, rather than
have this module make it silently.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .corpus import LetterCandidate, class_members
from .schema import (
    ConfidenceBlock,
    CooccurrenceEntry,
    HypothesisStatus,
    LetterClass,
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class HypothesisTestConfig:
    """§3.1/§3.2 test configuration.

    attractor_arity -- for each bond/ring/cage `critical_point.field_type`,
        how many nearest nuclear-attractor candidates (by position, within
        the same molecule) constitute "the attractor context" of one
        candidate of that type. Deliberately covers only
        "bond_critical_point" (arity 2 -- a QTAIM bond critical point
        connects exactly two attractors, unambiguously) and
        "ring_critical_point" (arity 3) by default -- NOT
        "cage_critical_point". Two reasons, both about honesty rather than
        completeness:
          (a) `critical_points.find_critical_points`'s own targeted search
              (Stage 2) only ever seeds ring-critical-point search from
              ATTRACTOR TRIPLES (see that module's docstring) -- it has no
              search stage for rings larger than 3 attractors, and no
              targeted cage search at all. Any cage-critical-point
              candidates this pipeline finds are incidental grid-search
              hits (Stage 1), not a systematically searched population --
              treating them as a well-sampled class here would overstate
              what the corpus actually supports.
          (b) unlike a bond (always exactly 2 endpoints) or the triples this
              pipeline actually searches for, a real cage's attractor count
              is not fixed (cubane's is 8, adamantane's is 4, ...) -- a
              single constant arity is not even the right model. A caller
              working with a corpus that DOES have deliberate cage search
              (and knows its typical member count) can add
              `"cage_critical_point": k` explicitly; this default declines
              to guess.
    min_members_for_decision -- a word-candidate class with fewer linked
        members than this returns `decision="indeterminate"` rather than a
        concentration statistic computed on too few points to mean
        anything (mirrors `environment_response.py`'s `len(vectors) < 3`
        "insufficient_data" guard, and `commutation.py`'s `len(centroids) <
        2` "no boundary yet" guard -- same idiom, applied here).
    n_null_permutations -- shared trial budget for BOTH permutation nulls
        this module runs (the letter-level co-occurrence label-permutation
        null in `build_letter_cooccurrence`, and the within-class
        concentration null in `evaluate_hypothesis_for_class`) -- both are
        the same computational shape (draw a random relabelling/resample,
        recompute one scalar statistic, repeat), so one shared knob is
        used rather than two independently-tunable ones with no principled
        reason to differ.
    rng_seed -- top-level seed. `evaluate_all_word_candidates` spawns one
        independent child seed per word-candidate class from this (via
        `numpy.random.SeedSequence.spawn`, the same pattern
        `unsupervised_validation.build_control_corpus` already uses for an
        identical reason: reproducible, non-overlapping randomness per unit
        of work, without ad hoc seed arithmetic).
    """

    attractor_arity: Dict[str, int] = field(
        default_factory=lambda: {"bond_critical_point": 2, "ring_critical_point": 3}
    )
    min_members_for_decision: int = 3
    n_null_permutations: int = 500
    rng_seed: int = 0

    def __post_init__(self) -> None:
        for field_type, arity in self.attractor_arity.items():
            if arity < 2:
                raise ValueError(
                    f"attractor_arity[{field_type!r}]={arity} -- co-occurrence "
                    "needs at least 2 attractor endpoints to form a pair"
                )
        if self.min_members_for_decision < 1:
            raise ValueError("min_members_for_decision must be >= 1")


# --------------------------------------------------------------------------
# step 1: link bond/ring/cage candidates to their attractor letter classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AttractorLink:
    """Which attractor `LetterClass`es one bond/ring/cage `LetterCandidate`
    connects. `attractor_class_ids` is sorted so the same physical pair/
    triple always hashes and compares equal regardless of discovery order.
    `attractor_distances` is kept alongside (same order as the *unsorted*
    nearest-neighbour search, not `attractor_class_ids`'s sorted order) only
    for diagnostics -- e.g. spotting a link whose nearest "attractor" is
    suspiciously far away, a sign the molecule's attractor context was
    ambiguous or the candidate itself is spurious.
    """

    candidate_index: int
    attractor_class_ids: Tuple[str, ...]
    attractor_distances: Tuple[float, ...]


def _molecule_groups(candidates: Sequence[LetterCandidate]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        groups[c.source_molecule_id].append(i)
    return groups


def _nearest_letter_class_id(
    vector: np.ndarray, classes_of_type: Sequence[LetterClass]
) -> Optional[str]:
    """Nearest-centroid `LetterClass.class_id` to `vector`, restricted to
    the (already type-filtered) classes given -- the same nearest-centroid
    idiom `corpus.class_members` uses, just returning the winning id instead
    of a distance-thresholded membership test, since here we want "which
    attractor letter is this" rather than "is this within threshold of one
    specific letter"."""
    if not classes_of_type:
        return None
    dists = [float(np.linalg.norm(vector - c.invariant_centroid)) for c in classes_of_type]
    return classes_of_type[int(np.argmin(dists))].class_id


def link_candidates_to_attractors(
    candidates: Sequence[LetterCandidate],
    letter_classes: Sequence[LetterClass],
    config: Optional[HypothesisTestConfig] = None,
) -> Dict[int, AttractorLink]:
    """Doc §3.2 step 1: for every candidate whose `critical_point.field_type`
    is a key of `config.attractor_arity`, find its nearest attractor
    candidates *by position* within the same molecule (not by descriptor
    distance -- position is what QTAIM bond-path topology is actually about;
    two candidates can be geometrically adjacent while having very different
    local field shapes, and vice versa), then map each of those attractor
    candidates onto its already-discovered `LetterClass` via nearest
    centroid in descriptor space.

    A candidate is silently OMITTED from the returned mapping (not given a
    partial/best-effort link) whenever:
      - its own molecule has fewer attractor candidates than the configured
        arity (not enough context to determine the full attractor tuple), or
      - no nuclear-attractor `LetterClass`es exist yet in `letter_classes`
        (alphabet not built yet -- nothing to link to).
    Callers should treat a candidate index missing from the result as "link
    undetermined", not "linked to nothing" -- the two are different claims,
    and only the second one is licensed by this function's actual guarantees.

    This is a position-proximity heuristic, not gradient-path tracing
    (following the actual bond path from the critical point to each
    attractor, which is what QTAIM analysis proper does). It is a reasonable
    approximation for the typical case the doc's toy/demo fields and the
    existing `critical_points.find_critical_points` search stage 2 are built
    for (seeding bond/ring search from attractor pairs/triples that are
    already close in Euclidean space) -- but for a strongly bent or unusually
    long bond path, nearest-by-position could disagree with the true
    topological correspondence. Not corrected here; flagged for a caller
    who needs the stronger guarantee.
    """
    config = config or HypothesisTestConfig()
    attractor_classes = [
        c for c in letter_classes if c.discovery_provenance.critical_point_type == "nuclear_attractor"
    ]

    mol_groups = _molecule_groups(candidates)
    attractor_idx_by_mol: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        if c.critical_point.field_type == "nuclear_attractor":
            attractor_idx_by_mol[c.source_molecule_id].append(i)

    attractor_class_id_cache: Dict[int, Optional[str]] = {}
    links: Dict[int, AttractorLink] = {}

    for mol_id, idxs in mol_groups.items():
        mol_attractor_idxs = attractor_idx_by_mol.get(mol_id, [])
        for i in idxs:
            c = candidates[i]
            ft = c.critical_point.field_type
            arity = config.attractor_arity.get(ft)
            if arity is None:
                continue
            if len(mol_attractor_idxs) < arity:
                continue

            ranked = sorted(
                mol_attractor_idxs,
                key=lambda j: float(
                    np.linalg.norm(c.critical_point.position - candidates[j].critical_point.position)
                ),
            )
            nearest = ranked[:arity]

            class_ids: List[str] = []
            distances: List[float] = []
            ok = True
            for j in nearest:
                if j not in attractor_class_id_cache:
                    attractor_class_id_cache[j] = _nearest_letter_class_id(
                        candidates[j].descriptor.to_vector(), attractor_classes
                    )
                cid = attractor_class_id_cache[j]
                if cid is None:
                    ok = False
                    break
                class_ids.append(cid)
                distances.append(
                    float(np.linalg.norm(c.critical_point.position - candidates[j].critical_point.position))
                )
            if not ok:
                continue

            order = sorted(range(arity), key=lambda k: class_ids[k])
            links[i] = AttractorLink(
                candidate_index=i,
                attractor_class_ids=tuple(class_ids[k] for k in order),
                attractor_distances=tuple(distances[k] for k in order),
            )

    return links


# --------------------------------------------------------------------------
# step 2: letter-level co-occurrence (the doc §6.1 "positional grid"), PPMI
# --------------------------------------------------------------------------


def _pairwise_evidence(tup: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """All C(len(tup), 2) pairwise sub-relations within one attractor tuple,
    each sorted for order-independence. A single ring/cage event of arity k
    contributes k*(k-1)/2 pieces of pairwise evidence -- e.g. a 3-membered
    ring contributes evidence for all 3 of its pairwise edges, matching a
    fully-connected reading of "these k attractors were found together in
    one topological feature" (deliberate modelling choice: this module does
    not attempt to recover which specific sub-pairs are directly bonded
    within a ring versus merely co-members of it -- that would need the
    full connectivity graph the `word_class.connectivity_topology` feature,
    out of scope here, would eventually supply)."""
    return [tuple(sorted((tup[a], tup[b]))) for a, b in combinations(range(len(tup)), 2)]


def _ppmi_from_events(events: Sequence[Tuple[str, ...]], arity: int) -> Dict[Tuple[str, str], float]:
    """Doc §6.1's distributional grid, operationalized as pointwise mutual
    information between attractor letter classes, floored at 0 (standard
    PPMI convention -- negative PMI from rare-pair noise is unstable and not
    the direction of evidence Hypothesis B needs, per-pair *avoidance*
    evidence is `schema.ForbiddenMapEntry`'s job, not this one's).

    Marginal P(A) is the "slot" frequency: each event contributes `arity`
    slot-draws, so P(A) = (slot occurrences of A across all events) /
    (n_events * arity). Joint P(A,B) is the pairwise-evidence frequency from
    `_pairwise_evidence`: P(A,B) = (pairwise co-occurrences of {A,B}) /
    (n_events * C(arity,2)). This is the standard skip-gram-style PMI
    construction (unigram frequency from raw token counts, co-occurrence
    frequency from its own, differently-normalized pair-count total) -- the
    two denominators differ but PMI only depends on the *ratio*
    P(A,B)/(P(A)P(B)), which is well-defined and comparable across pairs as
    long as both marginals are computed consistently this way for every
    pair, which they are.

    Only pairs with at least one observed co-occurrence get an entry --
    log2 of a zero joint probability is undefined, and floors to 0 (=
    "no evidence either way") regardless, so omitting them changes nothing
    about the result while avoiding an entry count that grows with the full
    attractor-class-pair combinatorics instead of with what was actually
    observed.
    """
    n_events = len(events)
    if n_events == 0:
        return {}

    slot_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    for tup in events:
        for label in tup:
            slot_counts[label] += 1
        for pair in _pairwise_evidence(tup):
            pair_counts[pair] += 1

    total_slots = n_events * arity
    n_pairs_per_event = arity * (arity - 1) // 2
    total_pairs = n_events * n_pairs_per_event

    ppmi: Dict[Tuple[str, str], float] = {}
    for pair, cnt in pair_counts.items():
        a, b = pair
        p_ab = cnt / total_pairs
        p_a = slot_counts[a] / total_slots
        p_b = slot_counts[b] / total_slots
        if p_a <= 0.0 or p_b <= 0.0 or p_ab <= 0.0:
            continue  # cannot happen given cnt > 0, guarded defensively anyway
        pmi = float(np.log2(p_ab / (p_a * p_b)))
        ppmi[pair] = max(0.0, pmi)
    return ppmi


def _label_permutation_null(
    observed_pairs: Sequence[Tuple[str, str]],
    events: Sequence[Tuple[str, ...]],
    arity: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> Dict[Tuple[str, str], List[float]]:
    """Matched null for `_ppmi_from_events`: flatten every event's attractor
    labels into one pool of `n_events * arity` slot-labels, shuffle the pool,
    reassemble into events of the same arity (preserving the number of
    events and the corpus-wide marginal frequency of every attractor label
    exactly, destroying only which labels co-occurred *within* an event),
    and recompute PPMI under that shuffled assignment. Repeated
    `n_permutations` times. A permutation that happens not to reproduce a
    given observed pair at all contributes 0.0 for it (no evidence = PPMI
    floor), the same convention `_ppmi_from_events` itself uses -- this
    keeps every observed pair's null sample the same length regardless of
    how often that specific pair reappears under shuffling.
    """
    n_events = len(events)
    flat_labels = np.array([label for tup in events for label in tup], dtype=object)
    null_samples: Dict[Tuple[str, str], List[float]] = {p: [] for p in observed_pairs}

    for _ in range(n_permutations):
        shuffled = rng.permutation(flat_labels)
        perm_events = [tuple(shuffled[i * arity : (i + 1) * arity]) for i in range(n_events)]
        perm_ppmi = _ppmi_from_events(perm_events, arity)
        for pair in observed_pairs:
            null_samples[pair].append(perm_ppmi.get(pair, 0.0))

    return null_samples


def build_letter_cooccurrence(
    candidates: Sequence[LetterCandidate],
    letter_classes: Sequence[LetterClass],
    critical_point_type: str,
    links: Optional[Dict[int, AttractorLink]] = None,
    config: Optional[HypothesisTestConfig] = None,
    corpus_version: str = "unversioned",
) -> List[CooccurrenceEntry]:
    """Doc §3.2 step 2: the letter-level (`level="letter"`) co-occurrence
    table restricted to ONE mediating `critical_point_type` at a time --
    never pooling bond-mediated and ring-mediated evidence into one table,
    for the same reason `corpus.py` never pools types before clustering
    (see module docstring). Call once per type present in the corpus (see
    `evaluate_all_word_candidates`, which does exactly this and caches the
    result).

    Returns one `schema.CooccurrenceEntry` per attractor-letter-class pair
    that was actually observed together at least once via a candidate of
    `critical_point_type`, `confidence.score` = observed PPMI,
    `confidence.interval` = the 2.5/97.5 percentile band of the
    label-permutation null (`_label_permutation_null`) -- a pair is only
    good evidence of "obligatory pairing" when `confidence.score` clears
    `confidence.interval[1]`.
    """
    config = config or HypothesisTestConfig()
    arity = config.attractor_arity.get(critical_point_type)
    if arity is None:
        raise ValueError(
            f"no attractor_arity configured for critical_point_type={critical_point_type!r} "
            "-- add it to config.attractor_arity (see HypothesisTestConfig docstring "
            "for why 'cage_critical_point' is not a default)"
        )

    if links is None:
        links = link_candidates_to_attractors(candidates, letter_classes, config)

    events: List[Tuple[str, ...]] = []
    for i, c in enumerate(candidates):
        if c.critical_point.field_type != critical_point_type:
            continue
        link = links.get(i)
        if link is None:
            continue
        events.append(link.attractor_class_ids)

    observed_ppmi = _ppmi_from_events(events, arity)
    if not observed_ppmi:
        return []

    observed_pairs = sorted(observed_ppmi.keys())
    rng = np.random.default_rng(config.rng_seed)
    null_samples = _label_permutation_null(observed_pairs, events, arity, config.n_null_permutations, rng)

    entries: List[CooccurrenceEntry] = []
    for pair in observed_pairs:
        lo, hi = np.percentile(null_samples[pair], [2.5, 97.5])
        entries.append(
            CooccurrenceEntry(
                class_id_pair=pair,
                ppmi_value=observed_ppmi[pair],
                confidence=ConfidenceBlock(
                    n=len(events),
                    method="permutation",
                    score=observed_ppmi[pair],
                    interval=(float(lo), float(hi)),
                    corpus_version=corpus_version,
                    stale=False,
                ),
                level="letter",
                mediating_critical_point_type=critical_point_type,
            )
        )
    return entries


# --------------------------------------------------------------------------
# step 3: per-word-candidate hypothesis verdict
# --------------------------------------------------------------------------


@dataclass
class WordCandidateHypothesisResult:
    """Per bond/ring/cage `LetterClass` ("candidate word") output of the
    §3.1/§3.2 distributional test. `hypothesis_status` is exactly the
    doc's `word_class.hypothesis_status` schema block (`schema.HypothesisStatus`)
    -- the field that will slot in unchanged once a full `word_class` is
    built (see module docstring). The rest of this dataclass is this
    module's own supporting evidence, not part of the doc's fixed schema
    text, in the same spirit as `fundamentality.FundamentalityResult`
    (which makes the identical distinction for its own, differently-shaped
    corpus-wide diagnostic).

    dominant_attractor_letters -- the most common attractor-class tuple
        among this class's linked members; `None` if there were too few
        linked members to identify one (`decision` is then "indeterminate").
    tuple_concentration -- `score` = fraction of this class's linked members
        whose attractor link equals `dominant_attractor_letters`; `interval`
        = the null band from `evaluate_hypothesis_for_class`'s within-class
        concentration test (a matched-size random subsample of every
        same-type candidate in the corpus, not just this class -- see that
        function's docstring). This is the module's own statistic (unlike
        `hypothesis_status.obligatory_pairing_score`, it is NOT a readout of
        `co_occurrence_matrix` -- there is no corpus-level table this
        within-class concentration could be read off instead, since it is
        specific to one class's membership, not to a pair of attractor
        letters in general).
    """

    class_id: str
    critical_point_type: str
    n_members_total: int
    n_members_linked: int
    dominant_attractor_letters: Optional[Tuple[str, ...]]
    tuple_concentration: Optional[ConfidenceBlock]
    hypothesis_status: HypothesisStatus


def evaluate_hypothesis_for_class(
    word_class: LetterClass,
    candidates: Sequence[LetterCandidate],
    letter_classes: Sequence[LetterClass],
    links: Optional[Dict[int, AttractorLink]] = None,
    cooccurrence: Optional[Sequence[CooccurrenceEntry]] = None,
    distance_threshold: Optional[float] = None,
    config: Optional[HypothesisTestConfig] = None,
    corpus_version: str = "unversioned",
    rng: Optional[np.random.Generator] = None,
    membership_index: Optional[Mapping[str, Sequence[int]]] = None,
    all_type_tuples: Optional[Sequence[Tuple[str, ...]]] = None,
) -> WordCandidateHypothesisResult:
    """Doc §3.1/§3.2, the per-class verdict: is `word_class` (which must be a
    bond/ring/cage `LetterClass`, never a "nuclear_attractor" one -- see the
    `ValueError` below) behaving as an independent, freely-combining letter
    (Hypothesis A) or a word-level relation obligatorily paired to specific
    attractor letters (Hypothesis B)?

    `distance_threshold` defaults to `word_class.discovery_provenance.
    persistence_threshold_used` -- the threshold `word_class` was actually
    clustered with (`corpus.cluster_candidates_bootstrap` records this on
    every class it produces) -- rather than requiring the caller to pass a
    value that has to be kept in sync by hand. Override only when
    deliberately re-scoping membership at a different threshold than
    discovery used.

    Two independent significance tests, both required to agree for a firm
    verdict (§3.1's own caution against overclaiming when evidence is mixed
    -- see module docstring):

    1. Within-class concentration (`tuple_concentration` on the result):
       of `word_class`'s own linked members, what fraction share the single
       most common attractor-letter tuple? Null = the same statistic
       computed on a random matched-size sample drawn from EVERY candidate
       of this `critical_point_type` in the corpus (this class's own
       members included, standard permutation-test construction -- compare
       `corpus.py`'s bootstrap cohesion, `fundamentality.py`'s
       reuse-breadth permutation). Significant (`score > interval[1]`) means
       this class's members are less variable in *which attractors they
       connect* than a random same-size draw from the whole type-partition
       would be -- direct evidence against "freely combines with a wide and
       variable set of neighbours" (Hypothesis A's prediction).
    2. Corpus-wide pairing strength (`hypothesis_status.obligatory_pairing_score`):
       read directly off `cooccurrence` (built by `build_letter_cooccurrence`,
       same `critical_point_type`) for the pairwise sub-relations of
       `dominant_attractor_letters` -- the WEAKEST of those entries if there
       is more than one (an "obligatory" tuple relation needs every pairwise
       sub-relation to hold, not just the strongest one). Significant means
       this specific attractor pairing co-occurs, corpus-wide, more than the
       label-permutation null predicts.

    `decision`:
        both significant                       -> "hypothesis_B_consistent"
        neither significant                    -> "hypothesis_A_consistent"
        too few linked members, OR the two      -> "indeterminate"
        signals disagree
    """
    config = config or HypothesisTestConfig()
    ft = word_class.discovery_provenance.critical_point_type
    if ft not in config.attractor_arity:
        raise ValueError(
            f"evaluate_hypothesis_for_class is only meaningful for a bond/ring/cage "
            f"word-candidate LetterClass (§3.1/§3.2) -- got critical_point_type={ft!r}. "
            "Nuclear-attractor classes are the candidate LETTERS this test distinguishes "
            "bond/ring/cage classes FROM, not word candidates themselves."
        )
    arity = config.attractor_arity[ft]

    if links is None:
        links = link_candidates_to_attractors(candidates, letter_classes, config)
    if cooccurrence is None:
        cooccurrence = build_letter_cooccurrence(
            candidates, letter_classes, ft, links=links, config=config, corpus_version=corpus_version,
        )
    cooc_by_pair = {e.class_id_pair: e for e in cooccurrence}

    threshold = (
        distance_threshold
        if distance_threshold is not None
        else word_class.discovery_provenance.persistence_threshold_used
    )
    # The master pipeline obtains these exact nearest-centroid assignments
    # once, immediately after clustering.  Reusing them avoids one O(N)
    # scan per word class and, importantly, keeps every downstream test on
    # the same operational definition of membership.  The fallback preserves
    # the public standalone API.
    if membership_index is not None:
        member_idxs = list(membership_index.get(word_class.class_id, ()))
    else:
        member_idxs = class_members(candidates, word_class, threshold)
    n_total = len(member_idxs)
    linked_tuples = [links[i].attractor_class_ids for i in member_idxs if i in links]
    n_linked = len(linked_tuples)

    if n_linked < config.min_members_for_decision:
        return WordCandidateHypothesisResult(
            class_id=word_class.class_id,
            critical_point_type=ft,
            n_members_total=n_total,
            n_members_linked=n_linked,
            dominant_attractor_letters=None,
            tuple_concentration=None,
            hypothesis_status=HypothesisStatus(obligatory_pairing_score=None, decision="indeterminate"),
        )

    counts = Counter(linked_tuples)
    dominant_tuple, dominant_count = counts.most_common(1)[0]
    observed_concentration = dominant_count / n_linked

    if all_type_tuples is None:
        all_type_tuples = [
            links[i].attractor_class_ids
            for i, c in enumerate(candidates)
            if c.critical_point.field_type == ft and i in links
        ]

    rng = rng if rng is not None else np.random.default_rng(config.rng_seed)
    null_concentrations: List[float] = []
    n_pool = len(all_type_tuples)
    for _ in range(config.n_null_permutations):
        if n_linked <= n_pool:
            sample_idx = rng.choice(n_pool, size=n_linked, replace=False)
        else:
            sample_idx = rng.choice(n_pool, size=n_linked, replace=True)
        sample_tuples = [all_type_tuples[j] for j in sample_idx]
        sample_counts = Counter(sample_tuples)
        null_concentrations.append(sample_counts.most_common(1)[0][1] / n_linked)
    lo, hi = np.percentile(null_concentrations, [2.5, 97.5])

    tuple_concentration = ConfidenceBlock(
        n=n_linked,
        method="permutation",
        score=observed_concentration,
        interval=(float(lo), float(hi)),
        corpus_version=corpus_version,
        stale=False,
    )

    sub_pairs = _pairwise_evidence(dominant_tuple)
    sub_entries = [cooc_by_pair[p] for p in sub_pairs if p in cooc_by_pair]
    obligatory_pairing_score: Optional[ConfidenceBlock] = None
    if sub_entries:
        weakest = min(sub_entries, key=lambda e: e.confidence.score)
        obligatory_pairing_score = weakest.confidence

    concentration_significant = tuple_concentration.score > tuple_concentration.interval[1]
    pairing_significant = (
        obligatory_pairing_score is not None
        and obligatory_pairing_score.interval is not None
        and obligatory_pairing_score.score > obligatory_pairing_score.interval[1]
    )

    if concentration_significant and pairing_significant:
        decision = "hypothesis_B_consistent"
    elif not concentration_significant and not pairing_significant:
        decision = "hypothesis_A_consistent"
    else:
        decision = "indeterminate"

    return WordCandidateHypothesisResult(
        class_id=word_class.class_id,
        critical_point_type=ft,
        n_members_total=n_total,
        n_members_linked=n_linked,
        dominant_attractor_letters=dominant_tuple,
        tuple_concentration=tuple_concentration,
        hypothesis_status=HypothesisStatus(
            obligatory_pairing_score=obligatory_pairing_score, decision=decision
        ),
    )


# --------------------------------------------------------------------------
# step 4: whole-corpus driver
# --------------------------------------------------------------------------


def evaluate_all_word_candidates(
    letter_classes: Sequence[LetterClass],
    candidates: Sequence[LetterCandidate],
    config: Optional[HypothesisTestConfig] = None,
    corpus_version: str = "unversioned",
    *,
    links: Optional[Dict[int, AttractorLink]] = None,
    cooccurrence_by_type: Optional[Mapping[str, Sequence[CooccurrenceEntry]]] = None,
    membership_index: Optional[Mapping[str, Sequence[int]]] = None,
) -> List[WordCandidateHypothesisResult]:
    """Doc §3.1/§3.2, run end to end: evaluate every bond/ring/cage
    `LetterClass` already discovered by `corpus.cluster_candidates_bootstrap`
    (i.e. every class whose `discovery_provenance.critical_point_type` is a
    key of `config.attractor_arity`). Attractor linking and the per-type
    co-occurrence table are each computed once and reused across every class
    of that type, not recomputed per class.

    Composes directly with `pipeline.decode_alphabet`:

        classes, candidates = pipeline.decode_alphabet(molecules, ...)
        results = distributional_test.evaluate_all_word_candidates(classes, candidates)

    `rng_seed` (via `config`) is spawned into one independent child seed per
    word-candidate class (`numpy.random.SeedSequence.spawn`) so every
    class's within-class concentration null is reproducible and
    non-overlapping with every other class's, the same pattern
    `unsupervised_validation.build_control_corpus` uses for its per-molecule
    control-field seeds.
    """
    config = config or HypothesisTestConfig()
    links = links or link_candidates_to_attractors(candidates, letter_classes, config)

    word_classes = [
        wc for wc in letter_classes if wc.discovery_provenance.critical_point_type in config.attractor_arity
    ]

    cooc_cache: Dict[str, Sequence[CooccurrenceEntry]] = dict(cooccurrence_by_type or {})
    tuples_by_type: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for index, link in links.items():
        tuples_by_type[candidates[index].critical_point.field_type].append(
            link.attractor_class_ids
        )
    master_seed_seq = np.random.SeedSequence(config.rng_seed)
    child_seeds = master_seed_seq.spawn(len(word_classes))

    results: List[WordCandidateHypothesisResult] = []
    for wc, seed in zip(word_classes, child_seeds):
        ft = wc.discovery_provenance.critical_point_type
        if ft not in cooc_cache:
            cooc_cache[ft] = build_letter_cooccurrence(
                candidates, letter_classes, ft, links=links, config=config, corpus_version=corpus_version,
            )
        results.append(
            evaluate_hypothesis_for_class(
                wc,
                candidates,
                letter_classes,
                links=links,
                cooccurrence=cooc_cache[ft],
                config=config,
                corpus_version=corpus_version,
                rng=np.random.default_rng(seed),
                membership_index=membership_index,
                all_type_tuples=tuples_by_type.get(ft, ()),
            )
        )
    return results


def run_distributional_test(
    classes: Sequence[LetterClass],
    all_candidates: Sequence[LetterCandidate],
    config: Optional[HypothesisTestConfig] = None,
    corpus_version: str = "unversioned",
) -> List[WordCandidateHypothesisResult]:
    """Thin alias over `evaluate_all_word_candidates` matching
    `pipeline.decode_alphabet`'s `(classes, all_candidates)` return-value
    order, so the §3.1/§3.2 test reads as the natural next pipeline step:

        classes, all_candidates = pipeline.decode_alphabet(molecules, ...)
        results = distributional_test.run_distributional_test(classes, all_candidates)
    """
    return evaluate_all_word_candidates(classes, all_candidates, config=config, corpus_version=corpus_version)


def summarize_hypothesis_evidence(
    results: Sequence[WordCandidateHypothesisResult],
) -> Dict[str, Dict[str, int]]:
    """Per-`critical_point_type` tally of `decision` outcomes -- deliberately
    NOT collapsed into one corpus-wide "Hypothesis A or B" verdict. §3.1 is
    explicit that a mixed outcome (or an outcome that differs molecule-class
    by molecule-class) is not ruled out by the theory; forcing one global
    answer here would be a claim the theory itself declines to make. A
    caller wanting a single yes/no answer should make that collapse
    explicitly, with their own stated reasoning for why it's warranted on
    their corpus, rather than have this function make it silently.
    """
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        out[r.critical_point_type][r.hypothesis_status.decision] += 1
    return {k: dict(v) for k, v in out.items()}
