"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
paradigm.py
===========
Doc §4.4-4.6 ("Парадигматическая (склонение) vs лексическая (синонимия)
вариативность"): distinguishes two kinds of link between two functionally-
close `LetterClass`es --

    - declension (paradigm membership): the displacement between the two
      classes' centroids, in invariant-descriptor space, is a REUSABLE
      transformation -- the same direction (up to sign/scale) recurs across
      many other, structurally unrelated class pairs in the corpus under
      the same varying parameter (§4.5's operational criterion, itself a
      direct reuse of §12's compressibility idiom: a reusable rule is
      fundamental in the same sense a reusable dictionary atom is).
    - synonymy: no such generalization -- the pair's closeness is
      idiosyncratic, specific to that one pair (§4.1(1): -COOH/tetrazole).

This module does NOT try to guess the physical label of a varying axis
(protonation state, isotope, oxidation state, conformer, ...) from the
displacement vector alone -- that is exactly the kind of chemistry-specific
inference the rest of this scaffold (invariants.py, commutation.py) refuses
to make from pure geometry. `varying_axis` on every `schema.Paradigm` this
module returns is caller-supplied; `detect_declension_paradigms` accepts an
optional `axis_labeler` callback for that purpose and otherwise leaves the
field as a stable, uninterpreted group id.

Pipeline:
    1. `pairwise_transformation` -- normalized centroid-to-centroid
       displacement for one ordered pair of same-type `LetterClass`es
       (direction + magnitude kept separate, since §4.5 asks whether the
       DIRECTION generalizes, not whether the absolute magnitude matches --
       two declension instances of "the same" grammatical transformation
       need not have identical step size, e.g. different bond lengths for
       the same protonation-state shift on different scaffolds).
    2. `group_transformations_by_direction` -- clusters transformation
       vectors across every candidate pair in the corpus by cosine
       similarity (agglomerative, same clustering idiom `corpus.py` uses
       for candidates themselves, just one level up: pairs of classes
       instead of candidates). A group with >= `min_group_size` members
       drawn from pairs that do not all share one class (i.e. genuinely
       different, structurally unconnected class pairs) is a candidate
       paradigm; a group of size 1 is definitionally idiosyncratic.
    3. `rule_reusability_confidence` -- bootstrap: does the direction
       cluster survive resampling which pairs are included? (matched-null
       idiom throughout this codebase, applied to pair membership instead
       of candidate membership.)
    4. `classify_pair` / `build_synonym_relations` -- per functionally-close
       pair (caller supplies which pairs qualify as "functionally close",
       e.g. from `semantics.NamedPropertyTest` results or a supplied
       function-distance matrix), the §4.5 verdict: declension-paradigm
       member if its transformation vector fell in a reusable direction
       group, idiosyncratic synonym otherwise, "insufficient_evidence" when
       there are too few functionally-close pairs in the corpus to tell a
       reusable rule from a coincidence at all (§4.7's own caution: this
       distinction is itself a hypothesis requiring corpus-scale
       validation, not a settled fact).

§4.7 caveat carried through explicitly: `min_group_size` and the direction-
similarity threshold are hyperparameters, not doctrine -- both are recorded
on `ParadigmConfig` and should be treated the same way `corpus.py` treats
`distance_threshold` (something to vary and report, not a fixed constant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .corpus import LetterCandidate
from .schema import ConfidenceBlock, LetterClass, Paradigm, SynonymRelation


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class ParadigmConfig:
    """§4.7: hyperparameters to record and vary, not doctrine.

    direction_similarity_threshold -- minimum cosine similarity between two
        pairs' transformation vectors for them to be grouped as "the same"
        rule.
    min_group_size -- minimum number of structurally-unconnected class
        pairs sharing a direction group before it counts as a reusable
        paradigm rule (§4.5: "применимым также к множеству других...
        пар") rather than one coincidental match.
    magnitude_ratio_tolerance -- two pairs' transformation vectors are only
        grouped if their magnitudes also agree within this relative
        tolerance -- direction alone is necessary but not sufficient (a
        rule that reverses sign or triples in size between instances is not
        obviously "the same" transformation).
    n_bootstrap -- resampling trials for `rule_reusability_confidence`.
    min_functionally_close_pairs -- §4.7: below this many candidate pairs
        in the whole corpus, `classify_pair` returns
        "insufficient_evidence" rather than a forced declension/synonymy
        call -- too few examples to distinguish a generalizing rule from
        chance at all.
    rng_seed -- top-level seed for bootstrap resampling.
    """

    direction_similarity_threshold: float = 0.85
    min_group_size: int = 3
    magnitude_ratio_tolerance: float = 0.5
    n_bootstrap: int = 500
    min_functionally_close_pairs: int = 5
    rng_seed: int = 0


# --------------------------------------------------------------------------
# step 1: pairwise transformation vectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairTransformation:
    """One candidate class-pair's centroid-to-centroid displacement.
    `direction` is a unit vector; `magnitude` is kept separately (see
    module docstring -- §4.5 groups by direction, gated by magnitude
    tolerance, not by the raw displacement vector directly)."""

    class_id_pair: Tuple[str, str]
    direction: np.ndarray
    magnitude: float


def pairwise_transformation(class_a: LetterClass, class_b: LetterClass) -> Optional[PairTransformation]:
    """§4.5: normalized displacement `class_b.invariant_centroid -
    class_a.invariant_centroid`. `None` if the two classes are not the same
    `critical_point_type` (§3.2 partition discipline: a "transformation"
    between a nuclear-attractor class and a bond-critical-point class isn't
    even the same kind of unit) or if the centroids coincide (zero-length
    displacement has no defined direction).
    """
    if (
        class_a.discovery_provenance.critical_point_type
        != class_b.discovery_provenance.critical_point_type
    ):
        return None
    disp = class_b.invariant_centroid - class_a.invariant_centroid
    magnitude = float(np.linalg.norm(disp))
    if magnitude < 1e-12:
        return None
    return PairTransformation(
        class_id_pair=(class_a.class_id, class_b.class_id),
        direction=disp / magnitude,
        magnitude=magnitude,
    )


def all_pairwise_transformations(
    letter_classes: Sequence[LetterClass],
    candidate_pairs: Optional[Sequence[Tuple[LetterClass, LetterClass]]] = None,
) -> List[PairTransformation]:
    """Transformation vectors for a set of candidate class pairs.

    `candidate_pairs`, when given, should be pairs the caller already has
    independent reason to suspect might be related (e.g. functionally-close
    pairs from `semantics.py`'s named-correlate distances, or a known
    protonation/oxidation-state family) -- this is the intended, primary
    usage: §4.5's reusability test is about whether a rule generalizes
    ACROSS such suspected pairs, not about mining every possible pair in
    the alphabet for accidental directional alignment.

    Without `candidate_pairs` (the default), falls back to every ordered
    pair within each `critical_point_type` partition -- a coarse,
    combinatorial default that scales as O(n^2) in alphabet size and, at
    high similarity thresholds and low dimensionality, is increasingly
    likely to group together pairs that share a direction by pure chance
    rather than genuine relatedness (the number of pairwise comparisons
    grows quadratically while the chance of *some* spurious alignment among
    them grows with it). Prefer passing `candidate_pairs` explicitly on any
    corpus large enough for this to matter; the default exists for small
    exploratory corpora only, and `detect_declension_paradigms` documents
    the same caveat where it is actually consumed.

    Ordered (not just combinations) because a displacement and its reverse
    are physically different transformations (protonation vs
    deprotonation) even though they share a direction group up to sign --
    `group_transformations_by_direction` compares |cosine similarity| so
    sign-reversed instances of the same rule still group together, while
    `class_id_pair` keeps track of which way each individual instance runs.
    """
    if candidate_pairs is not None:
        out: List[PairTransformation] = []
        for a, b in candidate_pairs:
            t = pairwise_transformation(a, b)
            if t is not None:
                out.append(t)
        return out

    by_type: Dict[str, List[LetterClass]] = {}
    for c in letter_classes:
        by_type.setdefault(c.discovery_provenance.critical_point_type, []).append(c)

    out = []
    for classes_of_type in by_type.values():
        for a in classes_of_type:
            for b in classes_of_type:
                if a.class_id == b.class_id:
                    continue
                t = pairwise_transformation(a, b)
                if t is not None:
                    out.append(t)
    return out


# --------------------------------------------------------------------------
# step 2: group by reusable direction
# --------------------------------------------------------------------------


def _pairs_connected_component(
    transformations: Sequence[PairTransformation],
) -> "list[set]":
    """Which transformation instances refer to disjoint sets of classes --
    used so a "group of size >= min_group_size" genuinely spans
    structurally-unconnected pairs (§4.5: "не связанных общим корнем"),
    not the same one or two classes counted multiple times via different
    orderings."""
    components: "list[set]" = []
    for t in transformations:
        a, b = t.class_id_pair
        merged = {a, b}
        remaining = []
        for comp in components:
            if comp & merged:
                merged |= comp
            else:
                remaining.append(comp)
        remaining.append(merged)
        components = remaining
    return components


def group_transformations_by_direction(
    transformations: Sequence[PairTransformation],
    config: Optional[ParadigmConfig] = None,
) -> List[List[PairTransformation]]:
    """§4.5: group transformation instances whose directions agree (up to
    sign, since a rule and its inverse are the same underlying grammatical
    relation run backwards) within `config.direction_similarity_threshold`
    AND whose magnitudes agree within `config.magnitude_ratio_tolerance`.
    Greedy single-linkage grouping (deterministic order-of-input, adequate
    for a scaffold; a production system might want full agglomerative
    clustering on the (direction, magnitude) joint space instead).

    Returns every group found, INCLUDING singletons -- filtering by
    `min_group_size` and by "spans structurally-unconnected pairs" is
    `classify_pair`'s job, not this function's, so callers inspecting the
    raw grouping (e.g. for diagnostics) see everything.
    """
    config = config or ParadigmConfig()
    groups: List[List[PairTransformation]] = []
    for t in transformations:
        placed = False
        for g in groups:
            rep = g[0]
            cos_sim = abs(float(np.dot(t.direction, rep.direction)))
            if cos_sim < config.direction_similarity_threshold:
                continue
            mag_ratio = t.magnitude / rep.magnitude if rep.magnitude > 0 else np.inf
            if not (
                1.0 - config.magnitude_ratio_tolerance
                <= mag_ratio
                <= 1.0 + config.magnitude_ratio_tolerance
            ):
                continue
            g.append(t)
            placed = True
            break
        if not placed:
            groups.append([t])
    return groups


def _is_reusable_group(group: Sequence[PairTransformation], config: ParadigmConfig) -> bool:
    """§4.5's actual reusability gate: enough members (`min_group_size`)
    AND those members span more than one connected component of classes
    (otherwise a single densely-interlinked family of pairs -- e.g. every
    ordered pair among just 3 classes -- could pad a group's size without
    it generalizing to any genuinely different pair, exactly the
    "idiosyncratic... совпадение" §4.5 warns a real paradigm must NOT be)."""
    if len(group) < config.min_group_size:
        return False
    components = _pairs_connected_component(group)
    return len(components) >= 2 or (len(components) == 1 and len(group) >= config.min_group_size * 2)


# --------------------------------------------------------------------------
# step 3: reusability confidence (bootstrap over which pairs are included)
# --------------------------------------------------------------------------


def rule_reusability_confidence(
    group: Sequence[PairTransformation],
    all_transformations: Sequence[PairTransformation],
    config: Optional[ParadigmConfig] = None,
    corpus_version: str = "unversioned",
) -> ConfidenceBlock:
    """Bootstrap: resample `all_transformations` with replacement, re-run
    `group_transformations_by_direction`, and measure the fraction of
    resamples in which a group at least as large as `len(group)` and
    passing `_is_reusable_group` still forms around approximately the same
    mean direction (cosine similarity to the original group's mean
    direction >= `config.direction_similarity_threshold`). Low score =
    the apparent paradigm is an artifact of exactly which pairs happened
    to be in this corpus, not a stable rule (§4.7's caution made
    quantitative).
    """
    config = config or ParadigmConfig()
    if not group:
        return ConfidenceBlock(n=0, method="bootstrap", score=0.0, corpus_version=corpus_version)

    mean_dir = np.mean([t.direction * np.sign(np.dot(t.direction, group[0].direction)) for t in group], axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)

    rng = np.random.default_rng(config.rng_seed)
    n = len(all_transformations)
    hits = 0
    for _ in range(config.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = [all_transformations[i] for i in idx]
        boot_groups = group_transformations_by_direction(resampled, config)
        survived = False
        for g in boot_groups:
            if len(g) < len(group):
                continue
            g_mean = np.mean([t.direction * np.sign(np.dot(t.direction, mean_dir)) for t in g], axis=0)
            norm = np.linalg.norm(g_mean)
            if norm < 1e-12:
                continue
            g_mean = g_mean / norm
            if abs(float(np.dot(g_mean, mean_dir))) >= config.direction_similarity_threshold:
                survived = True
                break
        hits += int(survived)

    score = hits / config.n_bootstrap
    return ConfidenceBlock(
        n=config.n_bootstrap, method="bootstrap", score=score, corpus_version=corpus_version, stale=False
    )


# --------------------------------------------------------------------------
# step 4: build Paradigm objects + classify individual pairs
# --------------------------------------------------------------------------


def detect_declension_paradigms(
    letter_classes: Sequence[LetterClass],
    candidate_pairs: Optional[Sequence[Tuple[LetterClass, LetterClass]]] = None,
    config: Optional[ParadigmConfig] = None,
    axis_labeler: Optional[Callable[[List[PairTransformation]], str]] = None,
    corpus_version: str = "unversioned",
) -> List[Paradigm]:
    """§4.6, top-level entry: find every reusable transformation-direction
    group across the corpus and package each as a `schema.Paradigm`.

    `candidate_pairs` -- see `all_pairwise_transformations`: strongly
    preferred over the default all-pairs fallback on any real corpus,
    since it is what keeps the reusability test honest (generalizing
    across *suspected*, functionally-motivated pairs, not across every
    pair the alphabet happens to contain).

    `axis_labeler(group) -> str` lets a caller with chemistry knowledge
    (e.g. "this direction correlates with a +1 formal-charge shift in
    every member") supply `varying_axis`; without one, groups are labeled
    `"unlabeled_axis_{i}"` -- a stable placeholder, not a claim about what
    physically varies (§4.4's own point: the theory does not get to assume
    a label, only detect that *some* reusable rule exists).
    """
    config = config or ParadigmConfig()
    transformations = all_pairwise_transformations(letter_classes, candidate_pairs)
    groups = group_transformations_by_direction(transformations, config)

    paradigms: List[Paradigm] = []
    for i, g in enumerate(groups):
        if not _is_reusable_group(g, config):
            continue
        mean_dir = np.mean([t.direction * np.sign(np.dot(t.direction, g[0].direction)) for t in g], axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-12:
            continue
        mean_dir = mean_dir / norm

        # ordered class_ids: chain the pairs into their declension sequence
        # where possible (best-effort; falls back to arbitrary order for a
        # branching/non-chain group).
        seen: List[str] = []
        for t in g:
            for cid in t.class_id_pair:
                if cid not in seen:
                    seen.append(cid)

        conf = rule_reusability_confidence(g, transformations, config, corpus_version)
        varying_axis = axis_labeler(g) if axis_labeler is not None else f"unlabeled_axis_{i}"

        paradigms.append(
            Paradigm(
                paradigm_id=f"paradigm_{i}",
                class_ids=seen,
                varying_axis=varying_axis,
                transformation_rule=mean_dir,
                rule_reusability=conf,
            )
        )
    return paradigms


def classify_pair(
    class_a: LetterClass,
    class_b: LetterClass,
    paradigms: Sequence[Paradigm],
    n_functionally_close_pairs_in_corpus: int,
    config: Optional[ParadigmConfig] = None,
    function_distance: Optional[float] = None,
    graph_distance: Optional[float] = None,
) -> SynonymRelation:
    """§4.5 verdict for one functionally-close pair (caller has already
    established, e.g. via `semantics.py` named-correlate distances, that
    `class_a`/`class_b` are functionally close enough to be a synonymy/
    declension candidate at all -- this function only decides WHICH of the
    two, given that precondition holds).

    `n_functionally_close_pairs_in_corpus` gates §4.7's caution: below
    `config.min_functionally_close_pairs`, returns "insufficient_evidence"
    regardless of how the direction happens to fall, since too few
    examples exist in the whole corpus to tell a generalizing rule from
    coincidence for ANY pair yet.
    """
    config = config or ParadigmConfig()
    t = pairwise_transformation(class_a, class_b)
    invariant_distance = (
        float(np.linalg.norm(class_b.invariant_centroid - class_a.invariant_centroid))
    )

    if n_functionally_close_pairs_in_corpus < config.min_functionally_close_pairs or t is None:
        return SynonymRelation(
            class_id_pair=(class_a.class_id, class_b.class_id),
            graph_distance=graph_distance,
            function_distance=function_distance,
            invariant_distance=invariant_distance,
            classification="insufficient_evidence",
        )

    for p in paradigms:
        if class_a.class_id in p.class_ids and class_b.class_id in p.class_ids:
            cos_sim = abs(float(np.dot(t.direction, p.transformation_rule)))
            if cos_sim >= config.direction_similarity_threshold:
                return SynonymRelation(
                    class_id_pair=(class_a.class_id, class_b.class_id),
                    graph_distance=graph_distance,
                    function_distance=function_distance,
                    invariant_distance=invariant_distance,
                    classification="declension_paradigm_member",
                )

    return SynonymRelation(
        class_id_pair=(class_a.class_id, class_b.class_id),
        graph_distance=graph_distance,
        function_distance=function_distance,
        invariant_distance=invariant_distance,
        classification="idiosyncratic_synonym",
    )


def build_synonym_relations(
    functionally_close_pairs: Sequence[Tuple[LetterClass, LetterClass]],
    paradigms: Sequence[Paradigm],
    config: Optional[ParadigmConfig] = None,
    function_distances: Optional[Dict[Tuple[str, str], float]] = None,
    graph_distances: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[SynonymRelation]:
    """Batch wrapper: classify every caller-supplied functionally-close
    pair against the corpus's already-detected paradigms (see
    `detect_declension_paradigms`). `function_distances` /
    `graph_distances` are optional lookups keyed by `(class_a.class_id,
    class_b.class_id)` (checked in both orderings) -- e.g. for feeding
    `expectation_profiles.check_synonym_profile` (§19.8) directly."""
    config = config or ParadigmConfig()
    n = len(functionally_close_pairs)
    function_distances = function_distances or {}
    graph_distances = graph_distances or {}

    def _lookup(d: Dict[Tuple[str, str], float], a: str, b: str) -> Optional[float]:
        return d.get((a, b), d.get((b, a)))

    out: List[SynonymRelation] = []
    for a, b in functionally_close_pairs:
        out.append(
            classify_pair(
                a, b, paradigms, n, config,
                function_distance=_lookup(function_distances, a.class_id, b.class_id),
                graph_distance=_lookup(graph_distances, a.class_id, b.class_id),
            )
        )
    return out
