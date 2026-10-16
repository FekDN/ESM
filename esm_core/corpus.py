"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
corpus.py
=========
Doc §15, step 3 ("Кластеризация кандидатов в буквенные классы"): turn a pile
of per-molecule critical-point candidates into named equivalence classes
(`schema.LetterClass`), each with a bootstrap-estimated `cluster_stability`
confidence (§15 step 3: "была голым score — теперь полноценный
confidence-блок").

What this module does NOT do (left for later phases, see schema.py docstring):
    - correlate classes with named physical properties (§16) -- every class
      this module produces starts at the default `orphan=True`,
      `orphan_confidence=None`, since that requires external property data
      this module has no source for. That correlation step now has a home
      (see `semantics.update_semantic_status`, which takes the isostere.py
      calibration results this module's classes were never wired to and
      produces the class's real `semantic_status`) -- it is simply not run
      automatically as part of clustering, since it needs externally
      supplied measured properties per class, not just the corpus itself;
    - the isostere/Mantel calibration design of §15 step 5, which
      needs an external molecular-graph-similarity source this scaffold does
      not have;
    - anything from the trajectory/dynamics phases (§8-10, steps 7-11).

Commutation testing (§11, step 4 -- which descriptor axes inside an
already-found class are signifying vs silent) is implemented, but in
`commutation.py`, not here: it operates one level below this module (axes
*within* a unit corpus.py already decided exists), so it consumes this
module's output (`LetterClass.invariant_centroid`) rather than living beside
the clustering that produces it.

Clustering method: agglomerative clustering with a distance threshold (i.e. a
dendrogram cut, not a fixed cluster count) on Euclidean distance between
`InvariantDescriptor.to_vector()` vectors -- appropriate because we don't know
the alphabet size in advance, which is the entire premise of Ventris's method
(doc §6.1): the grammar/vocabulary is discovered, not assumed.

Type partitioning (§3.1, §3.2): before any of the above, candidates are
split by `critical_point.field_type` (nuclear attractor vs. bond/ring/cage/
etc.) and clustered *independently within each partition* -- never pooled
across types into one flat alphabet. §3.2 is explicit about why: which
concrete topological object deserves the name "letter" (Hypothesis A: every
critical point type, flat; Hypothesis B: only the nuclear-attractor basin,
with bond/ring/cage critical points as word-level relations between basins)
is an open, falsifiable question the theory defers to a distributional test
-- and that test is only runnable if bond/ring/cage critical points are
still visibly, systematically tied to particular attractor pairs/triples in
the corpus statistics, rather than having already been merged into an
undifferentiated pool at the clustering step. Mixing types before
clustering would silently destroy the evidence needed to ever resolve
§3.1, not just bias it -- so every entry point below partitions by type by
default, and `LetterClass.discovery_provenance.critical_point_type` records
which partition a given class came from.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .critical_points import CriticalPoint
from .invariants import InvariantDescriptor
from .schema import ConfidenceBlock, DiscoveryProvenance, LetterClass, SemanticStatus

logger = logging.getLogger(__name__)


@dataclass
class LetterCandidate:
    """One critical point from one molecule, with its invariant descriptor --
    the atomic unit that this module (§15 step 3) clusters into letter classes."""

    source_molecule_id: str
    critical_point: CriticalPoint
    descriptor: InvariantDescriptor
    environment_condition: str = "unspecified"


def _partition_by_field_type(
    candidates: Sequence[LetterCandidate],
) -> "dict[str, List[int]]":
    """§3.1/§3.2: group candidate indices by `critical_point.field_type`,
    preserving each group's original relative order. This is the partition
    every clustering entry point in this module applies before comparing
    any two candidates' distance -- see module docstring."""
    groups: "dict[str, List[int]]" = defaultdict(list)
    for i, c in enumerate(candidates):
        groups[c.critical_point.field_type].append(i)
    return groups


def assign_candidate_labels(
    candidates: Sequence[LetterCandidate],
    classes: Sequence[LetterClass],
    distance_threshold: Union[float, Mapping[str, float]],
    *,
    batch_size: int = 8192,
    max_distance_elements: int = 8_000_000,
    use_gpu: bool = False,
) -> List[str | None]:
    """Assign candidates to nearest classes within their own type partition.

    Rare field types can be absent from ``suggest_distance_threshold``'s map:
    it deliberately omits partitions with fewer than two examples.  For
    these types, use a unique threshold recorded in the classes' discovery
    provenance (DP-means uses 0 for an uncalibrated singleton).  If provenance
    is missing or inconsistent, use a zero threshold rather than borrowing a
    scale from another topological type.  A zero-threshold exact match is
    checked by direct vector equality so floating-point cancellation in the
    dot-product distance formula cannot drop a singleton's own class.

    The bounded matrix product replaces a Python loop allocating one distance
    list per candidate. It is exact nearest-centroid assignment for calibrated
    thresholds and keeps temporary working memory bounded independently of
    corpus size. CuPy is optional and follows the identical NumPy path when
    unavailable.
    """
    if batch_size < 1 or max_distance_elements < 1:
        raise ValueError("batch_size and max_distance_elements must be positive")

    by_type: Dict[str, List[LetterClass]] = defaultdict(list)
    candidate_indices_by_type: Dict[str, List[int]] = defaultdict(list)
    for cls in classes:
        by_type[cls.discovery_provenance.critical_point_type].append(cls)
    for i, candidate in enumerate(candidates):
        candidate_indices_by_type[candidate.critical_point.field_type].append(i)

    xp = np
    if use_gpu:
        try:
            import cupy  # type: ignore

            xp = cupy
        except Exception:
            logger.warning("GPU label assignment requested but CuPy/CUDA is unavailable; using NumPy")

    labels: List[str | None] = [None] * len(candidates)
    for field_type, candidate_indices in candidate_indices_by_type.items():
        pool = by_type.get(field_type, [])
        if not pool:
            continue

        if isinstance(distance_threshold, Mapping):
            if field_type in distance_threshold:
                threshold = float(distance_threshold[field_type])
            else:
                recorded = {
                    float(cls.discovery_provenance.persistence_threshold_used)
                    for cls in pool
                    if np.isfinite(cls.discovery_provenance.persistence_threshold_used)
                    and cls.discovery_provenance.persistence_threshold_used >= 0.0
                }
                if len(recorded) == 1:
                    threshold = recorded.pop()
                    logger.warning(
                        "No suggested assignment threshold for %s; using class provenance threshold %.6g",
                        field_type,
                        threshold,
                    )
                else:
                    threshold = 0.0
                    logger.warning(
                        "No unique valid assignment threshold for %s; using exact-match-only assignment",
                        field_type,
                    )
        else:
            threshold = float(distance_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"assignment threshold for {field_type!r} must be finite and non-negative")

        centroids = np.stack([cls.invariant_centroid for cls in pool]).astype(np.float64)
        centroid_norm_sq = np.sum(centroids * centroids, axis=1)
        centroids_xp = xp.asarray(centroids)
        centroid_norm_sq_xp = xp.asarray(centroid_norm_sq)
        effective_batch = max(
            1,
            min(batch_size, max_distance_elements // max(1, len(pool))),
        )
        for start in range(0, len(candidate_indices), effective_batch):
            batch_indices = candidate_indices[start : start + effective_batch]
            vectors = np.stack(
                [candidates[i].descriptor.to_vector() for i in batch_indices]
            ).astype(np.float64, copy=False)
            if threshold == 0.0:
                for local_i, candidate_vector in enumerate(vectors):
                    exact_matches = np.flatnonzero(
                        np.all(centroids == candidate_vector[None, :], axis=1)
                    )
                    if len(exact_matches):
                        labels[batch_indices[local_i]] = pool[int(exact_matches[0])].class_id
                continue
            vectors_xp = xp.asarray(vectors)
            distance_sq = (
                xp.sum(vectors_xp * vectors_xp, axis=1)[:, None]
                + centroid_norm_sq_xp[None, :]
                - 2.0 * vectors_xp @ centroids_xp.T
            )
            nearest = xp.argmin(distance_sq, axis=1)
            nearest_distance_sq = distance_sq[xp.arange(len(batch_indices)), nearest]
            if xp is not np:
                nearest = xp.asnumpy(nearest)
                nearest_distance_sq = xp.asnumpy(nearest_distance_sq)
            for local_i, class_index in enumerate(np.asarray(nearest, dtype=int)):
                if max(0.0, float(nearest_distance_sq[local_i])) <= threshold * threshold:
                    labels[batch_indices[local_i]] = pool[class_index].class_id
    return labels


def suggest_distance_threshold(
    candidates: Sequence[LetterCandidate],
    quantile: float = 0.25,
    group_by_field_type: bool = False,
    max_sample_size: int | None = 20000,
    rng_seed: int = 0,
):
    """Data-driven starting point for `distance_threshold`
    (= discovery_provenance.persistence_threshold_used).

    The invariant descriptor's absolute scale depends on field amplitude,
    l_max, and shell count in a way that has no universal "natural" unit, so
    a fixed constant threshold across different fields/settings is
    meaningless -- e.g. bispectrum terms scale roughly cubically with field
    amplitude, so doubling amplitude does not double typical distances. This
    returns the given quantile of the pairwise-distance distribution across
    candidates as a reasonable starting cut point: low enough to be below
    most within-class spread, high enough to be above nearest-neighbor
    sampling noise. Still a heuristic -- per §15 step 3, the threshold is
    something to record and vary (see `discovery_provenance`), not to treat
    as ground truth.

    `group_by_field_type=False` (default, unchanged from before) pools every
    candidate's pairwise distances into one quantile and returns a single
    float. This is a reasonable quick-look number, but note it is exactly
    the kind of pooling `cluster_candidates_bootstrap` itself no longer does
    (§3.2): if different critical-point types occupy systematically
    different distance scales in this corpus, a single pooled quantile can
    be a poor threshold for some or all of the per-type clusterings it gets
    used on. Set `group_by_field_type=True` to instead get a
    ``{field_type: threshold}`` dict, one quantile per partition, matching
    what `cluster_candidates_bootstrap` will actually cluster against;
    types with fewer than 2 candidates are omitted (not enough pairs yet to
    suggest anything for that type).

    SCALE NOTE: `pdist` is itself O(n^2) regardless of what it feeds --
    exactly the complexity `dpmeans_incremental_cluster` exists to avoid for
    the clustering step, so a threshold-suggestion call that silently stays
    O(n^2) would just move the same wall one function earlier. Above
    `max_sample_size` candidates (per partition, if `group_by_field_type`),
    this draws a random subsample of that size and computes the quantile
    from it instead of the full pool -- a quantile estimated from a bounded
    random subsample converges to the same population quantile as one from
    the full pool, just with somewhat more sampling noise, which is the
    right trade to make for a number this function already documents as a
    starting heuristic, not ground truth. Pass `max_sample_size=None` to
    force the exact full-corpus computation (only reasonable when you know
    the corpus is small enough for O(n^2) to be fine).
    """
    if len(candidates) < 2:
        raise ValueError("need at least 2 candidates to suggest a threshold")
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(rng_seed)

    def _quantile_of(pool: Sequence[LetterCandidate]) -> float:
        if max_sample_size is not None and len(pool) > max_sample_size:
            idx = rng.choice(len(pool), size=max_sample_size, replace=False)
            pool = [pool[i] for i in idx]
        vectors = np.stack([c.descriptor.to_vector() for c in pool])
        return float(np.quantile(pdist(vectors), quantile))

    if not group_by_field_type:
        return _quantile_of(candidates)

    out: "dict[str, float]" = {}
    for field_type, idxs in _partition_by_field_type(candidates).items():
        if len(idxs) < 2:
            continue
        out[field_type] = _quantile_of([candidates[i] for i in idxs])
    return out


def _cluster_once(vectors: np.ndarray, distance_threshold: float, linkage: str) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage=linkage
    )
    return model.fit_predict(vectors)


def cluster_labels(
    candidates: Sequence[LetterCandidate], distance_threshold: float, linkage: str = "average"
) -> np.ndarray:
    """Public, deterministic wrapper around the same clustering
    `cluster_candidates_bootstrap` uses on the full (non-resampled) data --
    exposed so callers can recover exact class membership (which candidate
    index landed in which label) instead of an approximate nearest-centroid
    lookup against `LetterClass.invariant_centroid`, which is only a mean and
    can be misleading for a class with real internal spread.

    Partitions by `critical_point.field_type` first (§3.2), exactly like
    `cluster_candidates_bootstrap`: each type is clustered independently,
    and labels are offset per partition so they stay globally unique (a
    label collision between e.g. a "nuclear_attractor" cluster and a
    "bond_critical_point" cluster would wrongly imply shared membership).
    Returned array is aligned to `candidates`' original input order, not
    grouped by partition.
    """
    n = len(candidates)
    labels = np.full(n, -1, dtype=int)
    offset = 0
    for field_type, idxs in sorted(_partition_by_field_type(candidates).items()):
        vectors = np.stack([candidates[i].descriptor.to_vector() for i in idxs])
        group_labels = _cluster_once(vectors, distance_threshold, linkage)
        for local_i, global_i in enumerate(idxs):
            labels[global_i] = group_labels[local_i] + offset
        offset += int(group_labels.max()) + 1 if len(group_labels) else 0
    return labels


def _cohesion_fraction(original_members: set, resample_indices: np.ndarray, labels: np.ndarray) -> float:
    """Of the original cluster's members that reappear in a bootstrap
    resample, what fraction end up in the single largest shared label group?
    1.0 = perfectly reproduced, low = the class doesn't survive resampling."""
    present_positions = [
        pos for pos, orig_idx in enumerate(resample_indices) if orig_idx in original_members
    ]
    if not present_positions:
        return 0.0
    present_labels = labels[present_positions]
    counts = np.bincount(present_labels - present_labels.min())
    return float(counts.max()) / float(len(present_positions))


def _cluster_group_bootstrap(
    candidates: Sequence[LetterCandidate],
    critical_point_type: str,
    distance_threshold: float,
    l_max_bispectrum: int,
    linkage: str,
    n_bootstrap: int,
    rng_seed: int,
    corpus_version: str,
    class_id_prefix: str,
) -> List[LetterClass]:
    """Doc §15 step 3 bootstrap-stability clustering, run within ONE
    critical-point-type partition (see `cluster_candidates_bootstrap`,
    which is the public entry point and does the partitioning). Unchanged
    in method from before the §3.2 fix -- only ever called on a
    single-type `candidates` pool now, never the full mixed corpus."""
    vectors = np.stack([c.descriptor.to_vector() for c in candidates])
    n = len(candidates)

    full_labels = _cluster_once(vectors, distance_threshold, linkage)
    unique_labels = sorted(set(full_labels))

    rng = np.random.default_rng(rng_seed)
    # pre-draw all bootstrap resamples once, reuse across classes
    resamples = [rng.integers(0, n, size=n) for _ in range(n_bootstrap)]
    boot_labels = [
        _cluster_once(vectors[idx], distance_threshold, linkage) for idx in resamples
    ]

    classes: List[LetterClass] = []
    for label in unique_labels:
        members = set(np.where(full_labels == label)[0].tolist())
        cohesions = [
            _cohesion_fraction(members, resamples[b], boot_labels[b])
            for b in range(n_bootstrap)
        ]
        score = float(np.mean(cohesions))
        lo, hi = np.percentile(cohesions, [2.5, 97.5])

        member_vectors = vectors[list(members)]
        centroid = member_vectors.mean(axis=0)

        classes.append(
            LetterClass(
                class_id=f"{class_id_prefix}_{label}",
                discovery_provenance=DiscoveryProvenance(
                    persistence_threshold_used=distance_threshold,
                    l_max_bispectrum=l_max_bispectrum,
                    critical_point_type=critical_point_type,
                ),
                invariant_centroid=centroid,
                cluster_stability=ConfidenceBlock(
                    n=n_bootstrap,
                    method="bootstrap",
                    score=score,
                    interval=(float(lo), float(hi)),
                    corpus_version=corpus_version,
                    stale=False,
                ),
                semantic_status=SemanticStatus(orphan=True, orphan_confidence=None),
            )
        )
    return classes


def cluster_candidates_bootstrap(
    candidates: Sequence[LetterCandidate],
    distance_threshold: float,
    l_max_bispectrum: int,
    linkage: str = "average",
    n_bootstrap: int = 200,
    rng_seed: int = 0,
    corpus_version: str = "unversioned",
    class_id_prefix: str = "letter",
) -> List[LetterClass]:
    """Cluster candidates into letter classes, then estimate each class's
    `cluster_stability` by bootstrap resampling (§15 step 3): resample
    candidates with replacement, re-cluster with the *same*
    distance_threshold, and measure how cohesively the original class's
    members regroup (see `_cohesion_fraction`). The mean cohesion across
    `n_bootstrap` resamples is the confidence score; the 2.5/97.5
    percentiles give the interval.

    §3.2 type partitioning: `candidates` is first split by
    `critical_point.field_type` (see module docstring), and every step
    above -- clustering, bootstrap resampling, cohesion scoring -- runs
    independently within each partition. A candidate never competes for
    cluster membership against a candidate of a different critical-point
    type. `class_id_prefix` is namespaced per type
    (``f"{class_id_prefix}_{field_type}_{label}"``) so IDs stay unique and
    self-describing about which structural type produced them, and
    `LetterClass.discovery_provenance.critical_point_type` records it
    machine-readably.

    `distance_threshold` is exactly the doc's
    `discovery_provenance.persistence_threshold_used`, applied identically
    within every type partition -- it is the single hyperparameter that
    determines how finely each partition's alphabet is split, and per §15
    step 3 it should be treated as something to record and vary, not a
    fixed universal constant. (If per-type distance scales differ
    substantially in your corpus, see `suggest_distance_threshold`'s
    `group_by_field_type` option for per-type starting values, and call
    this function once per type with its own threshold instead of one
    shared value.)
    """
    if not candidates:
        return []

    classes: List[LetterClass] = []
    for field_type, idxs in sorted(_partition_by_field_type(candidates).items()):
        group_candidates = [candidates[i] for i in idxs]
        classes.extend(
            _cluster_group_bootstrap(
                group_candidates,
                critical_point_type=field_type,
                distance_threshold=distance_threshold,
                l_max_bispectrum=l_max_bispectrum,
                linkage=linkage,
                n_bootstrap=n_bootstrap,
                rng_seed=rng_seed,
                corpus_version=corpus_version,
                class_id_prefix=f"{class_id_prefix}_{field_type}",
            )
        )
    return classes


def class_members(
    candidates: Sequence[LetterCandidate], letter_class: LetterClass, distance_threshold: float
) -> List[int]:
    """Convenience: which candidate indices (into the given `candidates`,
    which may be any pool, not just the one the class was clustered from)
    are within distance_threshold of a class's centroid. Useful for
    inspecting what actually landed in a class (e.g. its
    realization_registry exemplars, schema.RealizationRegistry).

    Restricted to candidates whose `critical_point.field_type` matches
    `letter_class.discovery_provenance.critical_point_type` (§3.2) before
    computing any distance: since `cluster_candidates_bootstrap` never
    pools types, a candidate of a different type landing numerically near
    this centroid would be a coincidence in an unrelated part of descriptor
    space, not real class membership.
    """
    wanted_type = letter_class.discovery_provenance.critical_point_type
    idxs = [i for i, c in enumerate(candidates) if c.critical_point.field_type == wanted_type]
    if not idxs:
        return []
    vectors = np.stack([candidates[i].descriptor.to_vector() for i in idxs])
    d = np.linalg.norm(vectors - letter_class.invariant_centroid, axis=1)
    return [idxs[j] for j in np.where(d <= distance_threshold)[0]]


# ==============================================================================
# Incremental (DP-means + periodic BIRCH-style consolidation) clustering
# ==============================================================================
"""
Everything above this line is O(n^2) in both time and memory: `pdist` in
`suggest_distance_threshold` and sklearn's `AgglomerativeClustering` in
`_cluster_once` both need the full pairwise distance structure among
whatever candidate pool they're given, and `_cluster_group_bootstrap` then
repeats that same O(n^2) clustering `n_bootstrap` more times on top. This is
fine and was never the bottleneck at the corpus sizes this scaffold was
built and tested against (thousands of candidates per type) -- but it does
not degrade gracefully; it becomes flatly infeasible, not just slow, well
before "hundreds of millions of molecules": at ~26 candidates/molecule
(this scaffold's own observed ratio) that is ~5x10^9 candidates, i.e.
~10^18-10^19 pairwise distances even after §3.2's per-type partitioning --
not a runtime that gets long, a quantity of floating-point pairs no
plausible amount of hardware computes or stores.

`dpmeans_incremental_cluster` below is a different clustering METHOD, not a
faster implementation of the same one -- it never forms a pairwise distance
matrix at all. It is DP-means (Kulis & Jordan, "Revisiting k-means: New
Algorithms via Bayesian Nonparametrics", 2012): a non-parametric
generalization of k-means where a new prototype spawns whenever a point
lands farther than `spawn_threshold` from every existing prototype, and
otherwise the point joins its nearest prototype and that prototype's
running mean is updated in O(d) (d = descriptor dimension). Each point is
compared against the CURRENT prototype set (size k), never against other
raw points -- one full pass is O(n*k), and unlike the batch method's O(n^2),
this is linear in n for any FIXED k. Periodically (every
`consolidate_every_n_batches` batches), a BIRCH-style consolidation pass
merges any prototypes that have drifted within `merge_threshold` of each
other -- O(k^2), cheap as long as k stays much smaller than n, which is
exactly the empirical question doc §19.7 ("ExpectedAlphabetDiscoveryProfile")
already asks the corpus itself to answer: if the alphabet saturates as more
molecules are processed, k stays bounded and this whole approach stays fast
indefinitely; if it does not saturate, k grows with n and NO clustering
method -- incremental or batch -- can fix that, because it would mean there
is no small stable alphabet to find. This does not decide that question; it
is what makes it possible to keep collecting the evidence to answer it
past the point where the O(n^2) method could run at all.

Known, real limitation, stated plainly: DP-means (and any streaming
clustering) is ORDER-DEPENDENT. An early point that happens to sit between
two "true" clusters can spawn a prototype that later points then
accumulate around, splitting what a batch method looking at everything at
once might call one class -- consolidation repairs most of this after the
fact, but not all of it, and re-running on the same candidates in a
different processing order can produce a slightly different final k and
slightly different centroids. This is not a bug to be fixed by a better
threshold; it is an intrinsic property of the incremental setting the O(n^2)
method never had to contend with because it always saw every point at once.
Two concrete mitigations, both cheap because they only touch the k
prototypes, not the n raw points: (1) `dpmeans_incremental_cluster` accepts
an explicit `rng_seed` used only to shuffle candidate order within the
partition before batching -- running with several different seeds and
comparing resulting k / class membership is a direct, cheap way to see how
order-sensitive a given corpus actually is; (2) the consolidation pass
itself is exactly the repair mechanism, so `consolidate_every_n_batches`
trades order-sensitivity against consolidation cost, not against
correctness in the limit of consolidating after every single batch.
"""


@dataclass
class _IncrementalPrototype:
    centroid: np.ndarray
    count: int
    member_indices: List[int]


def _dpmeans_assign_batch(
    vectors: np.ndarray,
    global_idxs: Sequence[int],
    prototypes: List[_IncrementalPrototype],
    spawn_threshold: float,
) -> None:
    """One DP-means pass over a batch, mutating `prototypes` in place.
    O(batch_size * len(prototypes)) -- each point compared only against the
    CURRENT prototypes (typically << n), never against other raw points and
    never against points outside this batch."""
    for local_i in range(vectors.shape[0]):
        x = vectors[local_i]
        gi = global_idxs[local_i]
        if prototypes:
            centroids = np.stack([p.centroid for p in prototypes])
            d = np.linalg.norm(centroids - x, axis=1)
            j = int(np.argmin(d))
            dmin = float(d[j])
        else:
            j, dmin = -1, np.inf
        if j >= 0 and dmin <= spawn_threshold:
            p = prototypes[j]
            p.count += 1
            p.centroid = p.centroid + (x - p.centroid) / p.count  # Welford running mean
            p.member_indices.append(gi)
        else:
            prototypes.append(
                _IncrementalPrototype(centroid=x.copy(), count=1, member_indices=[gi])
            )


def _consolidate_prototypes(
    prototypes: List[_IncrementalPrototype], merge_threshold: float
) -> List[_IncrementalPrototype]:
    """BIRCH-style periodic merge over the CURRENT prototype set only --
    O(k^2) per call, independent of n. Greedily merges the closest pair
    below `merge_threshold`, repeating until no pair qualifies. Merged
    centroid is the count-weighted mean of the two originals, so a
    prototype that has already absorbed many points appropriately
    outweighs one that has only just spawned."""
    if len(prototypes) < 2:
        return prototypes
    changed = True
    while changed and len(prototypes) > 1:
        changed = False
        centroids = np.stack([p.centroid for p in prototypes])
        dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
        if dmat[i, j] <= merge_threshold:
            a, b = prototypes[i], prototypes[j]
            merged_count = a.count + b.count
            merged_centroid = (a.centroid * a.count + b.centroid * b.count) / merged_count
            merged = _IncrementalPrototype(
                centroid=merged_centroid,
                count=merged_count,
                member_indices=a.member_indices + b.member_indices,
            )
            prototypes = [p for idx, p in enumerate(prototypes) if idx not in (i, j)] + [merged]
            changed = True
    return prototypes


def _stability_via_subsample(
    vectors: np.ndarray,
    prototype_centroids: np.ndarray,
    spawn_threshold: float,
    sample_size: int,
    n_bootstrap: int,
    rng_seed: int,
) -> "dict[int, ConfidenceBlock]":
    """Doc §15 step 3 stability, adapted for a corpus too large to bootstrap
    in full (the batch method's `_cohesion_fraction` resamples and
    re-clusters ALL n candidates per trial -- itself O(n^2) per trial, on
    top of the O(n^2) base clustering; completely infeasible at the scale
    this module exists to handle).

    Draws ONE random subsample of `sample_size` vectors from the full
    partition (not per trial -- the same subsample is reused across all
    `n_bootstrap` trials, only which subsample-indices get repeated varies),
    assigns it to the ALREADY-FIXED final `prototype_centroids` by nearest
    distance (not re-run DP-means -- the prototypes are the answer being
    stability-tested, so trials must ask "how consistently does resampling
    land points on the SAME prototype", not "would resampling invent
    different prototypes", which is a different and much more expensive
    question). Bootstrap-resamples that fixed subsample with replacement,
    and for each resample, measures cohesion the same way the batch method
    does: of a prototype's subsample members that reappear, what fraction
    end up nearest to that same prototype (not necessarily the same
    subsample-local cluster ID, since no new prototypes are spawned here --
    every resampled point is assigned to its nearest EXISTING prototype).

    Cost: O(sample_size * k) to build the subsample assignment once, then
    O(sample_size * n_bootstrap) for the resampling loop -- independent of n.
    Returns {prototype_index: ConfidenceBlock}; a prototype with zero
    subsample members gets score=None-equivalent (n=0), since a corpus this
    large drawing a bounded random subsample will not necessarily sample
    every rare class -- this is an honest gap in coverage, not a stability
    verdict, and should be read that way rather than as "unstable".
    """
    n = vectors.shape[0]
    rng = np.random.default_rng(rng_seed)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample_vectors = vectors[sample_idx]

    d_to_proto = np.linalg.norm(
        sample_vectors[:, None, :] - prototype_centroids[None, :, :], axis=-1
    )
    sample_assignment = np.argmin(d_to_proto, axis=1)  # which prototype each sample point is nearest

    out: "dict[int, ConfidenceBlock]" = {}
    for proto_idx in range(prototype_centroids.shape[0]):
        own_positions = np.where(sample_assignment == proto_idx)[0]
        if len(own_positions) == 0:
            out[proto_idx] = ConfidenceBlock(n=0, method="bootstrap", score=0.0)
            continue
        if n_bootstrap <= 0:
            # Fast mode: caller only wants class structure/count (e.g. the
            # alphabet-size-curve snapshots), not a stability estimate --
            # skip the resampling loop rather than computing statistics
            # over an empty sample, which is undefined, not zero.
            out[proto_idx] = ConfidenceBlock(
                n=len(own_positions), method="expert_provisional", score=0.0,
            )
            continue
        own_set = set(own_positions.tolist())
        cohesions = []
        for _ in range(n_bootstrap):
            resample_local_idx = rng.integers(0, sample_size, size=sample_size)
            present = [p for p, orig in enumerate(resample_local_idx) if orig in own_set]
            if not present:
                cohesions.append(0.0)
                continue
            resampled_assignment = sample_assignment[resample_local_idx[present]]
            cohesions.append(float(np.mean(resampled_assignment == proto_idx)))
        score = float(np.mean(cohesions))
        lo, hi = np.percentile(cohesions, [2.5, 97.5])
        out[proto_idx] = ConfidenceBlock(
            n=len(own_positions), method="bootstrap",
            score=score, interval=(float(lo), float(hi)),
        )
    return out


def dpmeans_incremental_cluster(
    candidates: Sequence[LetterCandidate],
    spawn_threshold: Union[float, Mapping[str, float]],
    l_max_bispectrum: int,
    merge_threshold: float | None = None,
    batch_size: int = 2000,
    consolidate_every_n_batches: int = 5,
    stability_sample_size: int = 5000,
    n_bootstrap: int = 50,
    rng_seed: int = 0,
    corpus_version: str = "unversioned",
    class_id_prefix: str = "letter",
) -> List[LetterClass]:
    """DP-means alternative to `cluster_candidates_bootstrap` for corpora
    where O(n^2) is not just slow but infeasible -- see the module-level
    note above this function for the method, its complexity, and its
    known order-dependence limitation.

    Same contract as `cluster_candidates_bootstrap`: partitions by
    `critical_point.field_type` first (§3.2, never pools types), returns
    `LetterClass` objects with the same fields populated, and
    `DiscoveryProvenance.persistence_threshold_used` records
    `spawn_threshold` (the DP-means analogue of the batch method's
    `distance_threshold` -- same role: the single hyperparameter
    controlling how finely this partition's alphabet splits).

    `merge_threshold` defaults to `spawn_threshold` if not given (DP-means'
    usual choice: the same distance that triggers a new prototype also
    triggers consolidating two that have drifted together). Set it lower
    than `spawn_threshold` for more conservative merging if consolidation
    is proving too aggressive on your corpus.

    `stability_sample_size`/`n_bootstrap` control the subsample-based
    stability estimate (`_stability_via_subsample`) -- NOT full-corpus
    bootstrap, which is itself O(n^2) per trial and exactly what this
    function exists to avoid. Lower `stability_sample_size` for a cheaper
    (wider-interval) estimate on an extremely large corpus.
    """
    if not candidates:
        return []

    classes: List[LetterClass] = []
    for field_type, idxs in sorted(_partition_by_field_type(candidates).items()):
        rng = np.random.default_rng(rng_seed)
        order = rng.permutation(len(idxs))  # order-dependence: see module note above
        ordered_idxs = [idxs[i] for i in order]

        prototypes: List[_IncrementalPrototype] = []
        if isinstance(spawn_threshold, Mapping):
            if field_type not in spawn_threshold:
                # `suggest_distance_threshold(group_by_field_type=True)`
                # deliberately omits types with fewer than 2 candidates --
                # there is no pair yet to suggest a quantile from (see its
                # docstring). That is not a reason to abort the whole
                # corpus over one rare type (`cage_critical_point` at n=1
                # is the case that surfaced this): with no data to support
                # ANY merge distance, the only defensible threshold is one
                # that merges nothing -- every candidate of this type
                # stands as its own provisional class until enough
                # accumulate for a real threshold, not zero classes and a
                # crashed run. `_consolidate_prototypes` below still runs
                # normally; at spawn_threshold=0.0 it is a no-op (nothing
                # is ever <= 0.0 distance apart in continuous descriptor
                # space unless literally identical).
                logger.warning(
                    f"no DP-means threshold for critical-point type {field_type!r} "
                    f"(fewer than 2 candidates were available to suggest one) -- "
                    f"treating each of its candidates as its own provisional class "
                    f"rather than merging on an unsupported guess"
                )
                type_spawn_threshold = 0.0
            else:
                type_spawn_threshold = float(spawn_threshold[field_type])
        else:
            type_spawn_threshold = float(spawn_threshold)
        merge_thr = merge_threshold if merge_threshold is not None else type_spawn_threshold
        for batch_start in range(0, len(ordered_idxs), batch_size):
            batch_idxs = ordered_idxs[batch_start : batch_start + batch_size]
            batch_vectors = np.stack(
                [candidates[i].descriptor.to_vector() for i in batch_idxs]
            )
            _dpmeans_assign_batch(
                batch_vectors, batch_idxs, prototypes, type_spawn_threshold
            )
            batch_no = batch_start // batch_size
            if (batch_no + 1) % consolidate_every_n_batches == 0:
                prototypes = _consolidate_prototypes(prototypes, merge_thr)
        prototypes = _consolidate_prototypes(prototypes, merge_thr)  # final pass

        all_vectors_this_type = np.stack(
            [candidates[i].descriptor.to_vector() for i in idxs]
        )
        proto_centroids = np.stack([p.centroid for p in prototypes])
        stability_by_idx = _stability_via_subsample(
            all_vectors_this_type, proto_centroids, type_spawn_threshold,
            stability_sample_size, n_bootstrap, rng_seed,
        )

        for proto_idx, p in enumerate(prototypes):
            conf = stability_by_idx[proto_idx]
            classes.append(
                LetterClass(
                    class_id=f"{class_id_prefix}_{field_type}_{proto_idx}",
                    discovery_provenance=DiscoveryProvenance(
                    persistence_threshold_used=type_spawn_threshold,
                        l_max_bispectrum=l_max_bispectrum,
                        critical_point_type=field_type,
                    ),
                    invariant_centroid=p.centroid,
                    cluster_stability=ConfidenceBlock(
                        n=conf.n, method=conf.method, score=conf.score,
                        interval=conf.interval, corpus_version=corpus_version, stale=False,
                    ),
                    semantic_status=SemanticStatus(orphan=True, orphan_confidence=None),
                )
            )
    return classes


def _dpmeans_assign_batch_without_members(
    vectors: np.ndarray,
    prototypes: List[_IncrementalPrototype],
    spawn_threshold: float,
) -> None:
    """Streaming DP-means update that deliberately does not retain points.

    Growth-profile checkpoints only need the evolving alphabet, not a
    candidate-to-prototype registry.  Keeping `member_indices` at every
    checkpoint would make a one-pass diagnostic silently consume O(N) extra
    RAM (and, in the former runner, retaining every prefix pool consumed
    several times O(N)).  `_IncrementalPrototype` is reused solely for its
    centroid/count fields; its member lists stay empty here.
    """
    for x in vectors:
        if prototypes:
            centroids = np.stack([p.centroid for p in prototypes])
            distances = np.linalg.norm(centroids - x, axis=1)
            best = int(np.argmin(distances))
            best_distance = float(distances[best])
        else:
            best, best_distance = -1, np.inf
        if best >= 0 and best_distance <= spawn_threshold:
            prototype = prototypes[best]
            prototype.count += 1
            prototype.centroid = prototype.centroid + (x - prototype.centroid) / prototype.count
        else:
            prototypes.append(
                _IncrementalPrototype(centroid=x.copy(), count=1, member_indices=[])
            )


def _snapshot_letter_classes(
    prototypes_by_type: Mapping[str, Sequence[_IncrementalPrototype]],
    thresholds: Union[float, Mapping[str, float]],
    l_max_bispectrum: int,
    corpus_version: str,
    class_id_prefix: str,
) -> List[LetterClass]:
    """Materialise a lightweight, immutable view of streaming prototypes."""
    classes: List[LetterClass] = []
    for field_type in sorted(prototypes_by_type):
        threshold = (
            float(thresholds[field_type]) if isinstance(thresholds, Mapping)
            else float(thresholds)
        )
        for proto_index, prototype in enumerate(prototypes_by_type[field_type]):
            classes.append(
                LetterClass(
                    class_id=f"{class_id_prefix}_{field_type}_{proto_index}",
                    discovery_provenance=DiscoveryProvenance(
                        persistence_threshold_used=threshold,
                        l_max_bispectrum=l_max_bispectrum,
                        critical_point_type=field_type,
                    ),
                    invariant_centroid=prototype.centroid.copy(),
                    # A snapshot is an alphabet-size / drift diagnostic, not
                    # a fresh bootstrap experiment.  Mark this explicitly
                    # rather than fabricating a stability interval.
                    cluster_stability=ConfidenceBlock(
                        n=0, method="expert_provisional", score=0.0,
                        corpus_version=corpus_version, stale=False,
                    ),
                    semantic_status=SemanticStatus(orphan=True, orphan_confidence=None),
                )
            )
    return classes


def dpmeans_incremental_snapshots(
    candidate_batches: Sequence[Sequence[LetterCandidate]],
    checkpoint_batch_counts: Sequence[int],
    spawn_threshold: Union[float, Mapping[str, float]],
    l_max_bispectrum: int,
    *,
    batch_size: int = 2000,
    consolidate_every_n_batches: int = 5,
    corpus_version: str = "unversioned",
    class_id_prefix: str = "snapshot",
) -> List[List[LetterClass]]:
    """One streaming DP-means pass with alphabet snapshots at checkpoints.

    ``candidate_batches`` normally contains one batch per molecule in a
    *pre-registered, seeded random molecule order*.  Processing it once
    gives the actual alphabet after each growing prefix; it is not four
    independent re-clusterings of four ever-larger in-memory pools.  The
    output is suitable for §19.7's class-count, novel-letter and drift
    curves.  It intentionally does not estimate bootstrap stability or
    retain member lists: those are properties of the final fit and would
    invalidate the bounded-memory purpose of this function.

    Checkpoints are 1-based batch counts and must be non-decreasing.  Empty
    molecule batches are valid and still advance the corpus checkpoint.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if consolidate_every_n_batches < 1:
        raise ValueError("consolidate_every_n_batches must be positive")
    checkpoints = list(checkpoint_batch_counts)
    if checkpoints != sorted(checkpoints) or any(c < 1 for c in checkpoints):
        raise ValueError("checkpoint_batch_counts must be sorted positive batch counts")
    if checkpoints and checkpoints[-1] > len(candidate_batches):
        raise ValueError("a snapshot checkpoint exceeds candidate_batches")

    prototypes_by_type: "dict[str, List[_IncrementalPrototype]]" = defaultdict(list)
    pending_by_type: "dict[str, List[np.ndarray]]" = defaultdict(list)
    flush_count_by_type: "dict[str, int]" = defaultdict(int)

    def threshold_for(field_type: str) -> float:
        if isinstance(spawn_threshold, Mapping):
            if field_type not in spawn_threshold:
                raise ValueError(
                    f"missing DP-means threshold for critical-point type {field_type!r}"
                )
            return float(spawn_threshold[field_type])
        return float(spawn_threshold)

    def flush(field_type: str, *, force_consolidate: bool = False) -> None:
        pending = pending_by_type[field_type]
        if pending:
            vectors = np.stack(pending)
            _dpmeans_assign_batch_without_members(
                vectors, prototypes_by_type[field_type], threshold_for(field_type)
            )
            pending.clear()
            flush_count_by_type[field_type] += 1
        if force_consolidate or (
            flush_count_by_type[field_type]
            and flush_count_by_type[field_type] % consolidate_every_n_batches == 0
        ):
            prototypes_by_type[field_type] = _consolidate_prototypes(
                prototypes_by_type[field_type], threshold_for(field_type)
            )

    snapshots: List[List[LetterClass]] = []
    next_checkpoint = 0
    for batch_number, candidates in enumerate(candidate_batches, start=1):
        for candidate in candidates:
            field_type = candidate.critical_point.field_type
            pending_by_type[field_type].append(candidate.descriptor.to_vector())
            if len(pending_by_type[field_type]) >= batch_size:
                flush(field_type)

        while next_checkpoint < len(checkpoints) and batch_number == checkpoints[next_checkpoint]:
            # No prefix point may remain in a pending buffer at the moment
            # its alphabet is inspected.
            for field_type in list(pending_by_type):
                flush(field_type, force_consolidate=True)
            snapshots.append(_snapshot_letter_classes(
                prototypes_by_type, spawn_threshold, l_max_bispectrum,
                corpus_version, f"{class_id_prefix}_{batch_number}",
            ))
            next_checkpoint += 1

    return snapshots


def _dpmeans_pass_with_rss(
    vectors: np.ndarray,
    spawn_threshold: float,
    batch_size: int,
    consolidate_every_n_batches: int,
    rng_seed: int,
) -> Tuple[List[_IncrementalPrototype], float]:
    """One full DP-means pass (with consolidation) over `vectors`, LOCAL
    index space (vectors[i] for global index i is just i). Returns both the
    final prototypes and RSS -- total squared distance of every point to
    its OWN final assigned prototype's centroid. Kept separate from
    `dpmeans_incremental_cluster` (which builds full LetterClass/schema
    objects and estimates stability) so a threshold sweep of T candidate
    values costs T lean passes, not T expensive ones -- used by
    `select_threshold_by_mdl` below.
    """
    n = vectors.shape[0]
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(n)
    prototypes: List[_IncrementalPrototype] = []
    for batch_start in range(0, n, batch_size):
        batch_local_idxs = order[batch_start : batch_start + batch_size].tolist()
        batch_vectors = vectors[batch_local_idxs]
        _dpmeans_assign_batch(batch_vectors, batch_local_idxs, prototypes, spawn_threshold)
        batch_no = batch_start // batch_size
        if (batch_no + 1) % consolidate_every_n_batches == 0:
            prototypes = _consolidate_prototypes(prototypes, spawn_threshold)
    prototypes = _consolidate_prototypes(prototypes, spawn_threshold)

    rss = 0.0
    for p in prototypes:
        if not p.member_indices:
            continue
        member_vectors = vectors[p.member_indices]
        rss += float(np.sum((member_vectors - p.centroid) ** 2))
    return prototypes, rss


def select_threshold_by_mdl(
    candidates: Sequence[LetterCandidate],
    threshold_grid: Sequence[float] | None = None,
    n_grid_points: int = 12,
    batch_size: int = 2000,
    consolidate_every_n_batches: int = 5,
    rng_seed: int = 0,
) -> "dict[str, dict]":
    """Principled alternative to hand-picking `spawn_threshold`, picked per
    critical-point-type partition (§3.2) by minimizing a BIC-style two-part
    description length -- NOT by any criterion that knows or cares whether
    the resulting alphabet looks more or less consistent with the theory.
    That distinction is what keeps this a legitimate search rather than
    p-hacking (see the 2026-07 "поиск лучших вариантов" discussion): model-
    complexity parameters like this one have an objective, theory-agnostic
    selection criterion; evidentiary/significance thresholds elsewhere in
    this codebase (e.g. `commutation.py`'s `result_confidence` cutoffs,
    `paradigm.py`'s `min_functionally_close_pairs`) do NOT, and must stay
    fixed by principled reasoning made before looking at results, not
    searched.

    BIC approximates the {bits to store the k prototypes} + {bits to store
    each point's residual from its own prototype} two-part MDL code the
    same way it does for any Gaussian mixture / k-means-like model:

        BIC(k) = n*d*log(RSS(k) / (n*d))  +  k*d*log(n)

    First term: cost of encoding the data given the model (assumes
    isotropic Gaussian residuals), decreasing as k grows and fits the data
    more finely. Second term: cost of encoding the model itself (k*d free
    centroid coordinates, BIC's standard log(n)-per-parameter penalty). The
    threshold minimizing BIC is the one where these trade off best -- not
    the one giving the most classes, the fewest, or the best-looking
    correlation with any named property. This is the exact same quantity
    `compression_gain` (doc §19.2/§19.7) already computes as a diagnostic
    after the fact; this function uses it as a selection criterion before.

    Returns ``{field_type: {"best_threshold": float, "sweep": [(threshold,
    k, rss, bic), ...]}}``. The full sweep is returned, not just the
    argmin, so a human can check the minimum is a real, well-shaped one --
    a BIC still decreasing at the edge of the grid means the grid needs to
    extend further, not that the edge value should be trusted as-is.
    """
    out: "dict[str, dict]" = {}
    for field_type, idxs in _partition_by_field_type(candidates).items():
        if len(idxs) < 4:
            continue
        vectors = np.stack([candidates[i].descriptor.to_vector() for i in idxs])
        n, d = vectors.shape

        if threshold_grid is None:
            from scipy.spatial.distance import pdist

            sample = vectors
            if n > 3000:
                sub_idx = np.random.default_rng(rng_seed).choice(n, size=3000, replace=False)
                sample = vectors[sub_idx]
            dists = pdist(sample)
            grid = np.quantile(dists, np.linspace(0.02, 0.6, n_grid_points))
        else:
            grid = np.asarray(threshold_grid, dtype=float)

        sweep = []
        for thr in grid:
            prototypes, rss = _dpmeans_pass_with_rss(
                vectors, float(thr), batch_size, consolidate_every_n_batches, rng_seed
            )
            k = len(prototypes)
            rss_floor = max(rss, 1e-12)
            bic = n * d * np.log(rss_floor / (n * d)) + k * d * np.log(max(n, 2))
            sweep.append((float(thr), k, float(rss), float(bic)))

        best = min(sweep, key=lambda row: row[3])
        out[field_type] = {"best_threshold": best[0], "sweep": sweep}
    return out
