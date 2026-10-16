"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
critical_points.py
===================
Doc §15, step 1 ("Выбор кандидатов на «буквы»"): find topological critical
points of a scalar field (grad(field) = 0) and classify them the way Bader's
QTAIM classifies critical points of the electron density, by the (rank,
signature) of the Hessian at the point:

    (3,-3)  local maximum        -> "nuclear_attractor"
    (3,-1)  saddle, 2 neg/1 pos  -> "bond_critical_point"
    (3,+1)  saddle, 1 neg/2 pos  -> "ring_critical_point"
    (3,+3)  local minimum        -> "cage_critical_point"
    rank<3 (degenerate)          -> "degenerate" (flagged, not trusted --
                                     doc §15: "особенности, не устойчивые при
                                     переходе [между базисами], помечаются
                                     как артефакт метода, а не физическая буква")

`rank` = number of Hessian eigenvalues with |eigenvalue| above a tolerance;
`signature` = sum of the signs of those eigenvalues. The Poincaré-Hopf index
of a non-degenerate critical point in R^3 is (-1)^k where k = number of
negative eigenvalues -- this is what the doc's schema calls
`poincare_hopf_index` on each critical point record.

This classification is meaningful as written for density-like fields
(field_kind == "density"); for ESP or ELF the same (rank, signature) machinery
applies but the QTAIM *names* don't -- `classify` still returns rank/signature/
index, but `field_type` falls back to a generic rank/signature label unless
`field_kind == "density"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import ScalarField
from .schema import ConfidenceBlock

_QTAIM_LABELS = {
    (3, -3): "nuclear_attractor",
    (3, -1): "bond_critical_point",
    (3, 1): "ring_critical_point",
    (3, 3): "cage_critical_point",
}


@dataclass
class CriticalPoint:
    position: np.ndarray
    value: float
    eigenvalues: np.ndarray
    rank: int
    signature: int
    poincare_hopf_index: Optional[int]
    field_type: str
    basis_set_convergence: Optional[ConfidenceBlock] = None
    # Retained for later quality/convergence analysis.  These diagnostics do
    # not affect class assignment, but make it possible to filter or audit a
    # field-root without rerunning an expensive field calculation.
    gradient_norm: Optional[float] = None
    discovery_stage: str = "unknown"


def classify(hessian: np.ndarray, value: float, position: np.ndarray,
             field_kind: str = "density", degenerate_tol: float = 1e-3) -> CriticalPoint:
    """Classify a single critical point from its Hessian. `degenerate_tol` is
    an *absolute* eigenvalue threshold below which an eigenvalue is treated as
    numerically zero (rank-deficient) rather than genuinely non-zero but
    small -- this depends on field units/scale, tune per field_kind."""
    eigvals = np.linalg.eigvalsh(hessian)
    signs = np.sign(np.where(np.abs(eigvals) > degenerate_tol, eigvals, 0.0))
    rank = int(np.count_nonzero(signs))
    signature = int(np.sum(signs))

    if rank == 3:
        n_negative = int(np.sum(signs < 0))
        ph_index: Optional[int] = (-1) ** n_negative
        if field_kind == "density" and (rank, signature) in _QTAIM_LABELS:
            field_type = _QTAIM_LABELS[(rank, signature)]
        else:
            field_type = f"rank3_signature{signature:+d}"
    else:
        ph_index = None  # not well-defined for a degenerate critical point
        field_type = "degenerate"

    return CriticalPoint(
        position=np.asarray(position, dtype=float),
        value=float(value),
        eigenvalues=eigvals,
        rank=rank,
        signature=signature,
        poincare_hopf_index=ph_index,
        field_type=field_type,
    )


def _damped_newton(field: ScalarField, r0: np.ndarray, max_iter: int, grad_tol: float,
                    max_step: float) -> Optional[np.ndarray]:
    """Newton-Raphson on grad(field) = 0 using the local Hessian, with step
    clipping (not a full line search -- adequate for a scaffold, but a
    production system should add one, especially for ESP fields which are
    much less well-behaved than rho(r) near nuclei)."""
    r = np.asarray(r0, dtype=float).copy()
    for _ in range(max_iter):
        g = field.gradient(r)
        if np.linalg.norm(g) < grad_tol:
            return r
        H = field.hessian(r)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step = step * (max_step / step_norm)
        r = r - step
    g = field.gradient(r)
    return r if np.linalg.norm(g) < grad_tol * 10 else None


def _try_seed(field, seed, lo, hi, max_iter, grad_tol, max_step, dedupe_radius,
              degenerate_tol, found, kept_positions, discovery_stage="unknown"):
    root = _damped_newton(field, seed, max_iter, grad_tol, max_step)
    if root is None:
        return
    if np.any(root < lo - 1e-6) or np.any(root > hi + 1e-6):
        return  # converged outside the region of interest
    if any(np.linalg.norm(root - kept) < dedupe_radius for kept in kept_positions):
        return
    H = field.hessian(root)
    cp = classify(H, field.value(root), root, field.field_kind, degenerate_tol)
    cp.gradient_norm = float(np.linalg.norm(field.gradient(root)))
    cp.discovery_stage = str(discovery_stage)
    found.append(cp)
    kept_positions.append(root)


def _field_grid_maximum_seeds(
    field: ScalarField,
    *,
    limit: int,
) -> List[np.ndarray]:
    """Return local-maximum seeds read from a field grid, if it exposes one.

    The cube's own scalar samples, origin and grid vectors are the only
    inputs.  In particular, no atom block is consulted.  This handles sharp
    density maxima that a coarse Cartesian multi-start grid can miss, while
    keeping the actual critical-point decision in `_try_seed` (gradient zero
    plus Hessian classification).  Analytic/non-grid fields simply return no
    extra seeds and retain the ordinary multi-start search.
    """
    if limit < 1:
        return []
    grid = getattr(field, "grid", None)
    origin = getattr(field, "origin", None)
    vectors = getattr(field, "vectors", None)
    if grid is None or origin is None or vectors is None:
        return []
    values = np.asarray(grid)
    if values.ndim != 3 or min(values.shape) < 3 or not np.all(np.isfinite(values)):
        return []

    # Strict 26-neighbour maxima avoid a vast plateau of equal near-zero
    # values becoming artificial seeds.  The boolean work array is one byte
    # per interior voxel and is released before Newton refinement.
    maxima = np.ones(tuple(size - 2 for size in values.shape), dtype=bool)
    centre = values[1:-1, 1:-1, 1:-1]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                neighbour = values[
                    1 + di:values.shape[0] - 1 + di,
                    1 + dj:values.shape[1] - 1 + dj,
                    1 + dk:values.shape[2] - 1 + dk,
                ]
                maxima &= centre > neighbour

    indices = np.argwhere(maxima)
    if len(indices) == 0:
        return []
    indices += 1  # interior-grid coordinates to full-grid coordinates
    scores = values[tuple(indices.T)]
    if len(indices) > limit:
        selected = np.argpartition(scores, -limit)[-limit:]
        indices = indices[selected]
        scores = scores[selected]
    # Deterministic strongest-first ordering lets the dedupe radius suppress
    # weaker grid maxima around the same continuous feature.
    indices = indices[np.argsort(scores)[::-1]]
    return [np.asarray(origin, dtype=float) + index @ np.asarray(vectors, dtype=float) for index in indices]


def find_critical_points(
    field: ScalarField,
    bbox: Tuple[np.ndarray, np.ndarray],
    n_seeds_per_axis: int = 6,
    max_iter: int = 60,
    grad_tol: float = 1e-5,
    max_step: float = 0.5,
    dedupe_radius: float = 0.05,
    degenerate_tol: float = 1e-3,
    bond_search_radius: Optional[float] = 3.0,
    ring_search: bool = True,
    rng_seed: int = 0,
    nuclear_seeds: Optional[Sequence[np.ndarray]] = None,
    field_grid_maximum_seed_limit: int = 4096,
) -> List[CriticalPoint]:
    """Multi-start damped Newton-Raphson critical-point search over an axis-
    aligned box `bbox = (lo, hi)`, in three stages:

    Stage 0 -- optional externally supplied reference seeds.
    This is an *audit/search-coverage* aid, not part of the field-first
    alphabet extractor. `nuclear_seeds=None` means no Stage 0; this function
    never reads `field.atom_positions` implicitly. A seed merely starts
    Newton; every retained point must converge and be classified from the
    scalar field's gradient and Hessian. Near a sharp density maximum, a
    smaller `max_step` is used because the gradient is large.

    Stage 0b -- field-grid local maxima.  For a cube-backed density this
    scans the scalar samples themselves for strict local maxima, then
    Newton-refines and classifies them.  It prevents narrow field maxima from
    being missed merely because the regular starting grid is coarse; the
    cube atom block is not read.

    Stage 1 -- regular grid of seeds. Catches non-nuclear maxima and any
    saddles whose basins the grid happens to hit.

    Stage 2 -- targeted saddle search, the standard practical fix used by real
    QTAIM search tools: once attractors are known, search for bond critical
    points near the midpoint of every attractor pair within
    `bond_search_radius` (plus a few small perpendicular jitters, since the
    true BCP is only exactly at the geometric midpoint for symmetric cases),
    and (if `ring_search`) for ring critical points near the centroid of every
    mutually-close attractor triple. This stage is what makes bond critical
    points -- the doc's central worked example -- actually discoverable; skip
    it (bond_search_radius=None) only if you already know there are no
    (3,-3) attractors, e.g. for an ESP field.

    Not a certified/complete search either way: a production system wants a
    subdivision + interval-arithmetic sign-change proof, or a full Morse-Smale
    complex, for completeness guarantees (see schema.py's
    TopologicalSkeleton.morse_smale_graph_edges, left for that upgrade).
    """
    lo, hi = np.asarray(bbox[0], dtype=float), np.asarray(bbox[1], dtype=float)

    found: List[CriticalPoint] = []
    kept_positions: List[np.ndarray] = []

    # --- Stage 0: explicitly requested reference seeds, never implicit. ---
    if nuclear_seeds is not None and len(nuclear_seeds) > 0:
        # Near nuclei |grad| is large and Newton steps explode; clamp hard.
        nuclear_max_step = min(max_step, 0.15)
        nuclear_grad_tol = max(grad_tol, 1e-3)
        for seed in nuclear_seeds:
            seed_arr = np.asarray(seed, dtype=float)
            if seed_arr.shape != (3,):
                continue
            if np.any(seed_arr < lo - 1e-6) or np.any(seed_arr > hi + 1e-6):
                continue
            _try_seed(
                field, seed_arr, lo, hi, max_iter, nuclear_grad_tol, nuclear_max_step,
                dedupe_radius, degenerate_tol, found, kept_positions, "external_reference_seed",
            )
            # Deliberately do not add a point when Newton fails.  A declared
            # atom is a reference annotation, never a licence to manufacture
            # a field feature or assign it a topological type.

    # --- Stage 0b: maxima proposed by the scalar grid itself. ---
    for seed in _field_grid_maximum_seeds(
        field, limit=field_grid_maximum_seed_limit
    ):
        _try_seed(
            field, seed, lo, hi, max_iter, grad_tol, max_step,
            dedupe_radius, degenerate_tol, found, kept_positions, "field_grid_local_maximum",
        )

    # --- Stage 1: regular grid ---
    axes = [np.linspace(lo[i], hi[i], n_seeds_per_axis) for i in range(3)]
    seeds = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    for seed in seeds:
        _try_seed(field, seed, lo, hi, max_iter, grad_tol, max_step, dedupe_radius,
                  degenerate_tol, found, kept_positions, "regular_field_grid")

    if bond_search_radius is not None:
        attractors = [cp for cp in found if cp.field_type == "nuclear_attractor"]
        rng = np.random.default_rng(rng_seed)

        pairs = [
            (a, b)
            for i, a in enumerate(attractors)
            for b in attractors[i + 1:]
            if np.linalg.norm(a.position - b.position) <= bond_search_radius
        ]
        for a, b in pairs:
            mid = 0.5 * (a.position + b.position)
            bond_vec = b.position - a.position
            bond_len = np.linalg.norm(bond_vec)
            jitters = [np.zeros(3)] + [
                0.15 * bond_len * rng.normal(size=3) for _ in range(4)
            ]
            for j in jitters:
                _try_seed(field, mid + j, lo, hi, max_iter, grad_tol, max_step,
                          dedupe_radius, degenerate_tol, found, kept_positions,
                          "field_attractor_pair")

        if ring_search and len(attractors) >= 3:
            for i, a in enumerate(attractors):
                for j, b in enumerate(attractors[i + 1:], start=i + 1):
                    if np.linalg.norm(a.position - b.position) > bond_search_radius:
                        continue
                    for c in attractors[j + 1:]:
                        if (np.linalg.norm(b.position - c.position) > bond_search_radius
                                or np.linalg.norm(a.position - c.position) > bond_search_radius):
                            continue
                        centroid = (a.position + b.position + c.position) / 3.0
                        _try_seed(field, centroid, lo, hi, max_iter, grad_tol, max_step,
                                  dedupe_radius, degenerate_tol, found, kept_positions,
                                  "field_attractor_triple")

    return found


def basis_convergence_discrepancy(
    cp_low: CriticalPoint, cp_high: CriticalPoint, position_scale: float = 1.0
) -> float:
    """Continuous magnitude of disagreement between the same critical point
    found at two levels of theory: 0.0 = perfect agreement, growing
    unboundedly as position/type/index diverge. `position_scale` is a
    field-appropriate length (e.g. a typical bond length in the field's
    units) so the positional term is dimensionless and comparable across
    fields with different natural scales, rather than depending on an
    arbitrary absolute cutoff.

    A change of *kind* (different field_type or a flipped Poincare-Hopf
    index) is qualitatively worse than a small positional shift -- a
    critical point that changed kind across levels of theory should never
    read as "mostly converged" just because its position stayed close, so
    each categorical mismatch adds a fixed penalty on top of the positional
    term rather than being blended smoothly into one continuous distance.
    """
    position_term = float(np.linalg.norm(cp_low.position - cp_high.position)) / position_scale
    type_penalty = 0.0 if cp_low.field_type == cp_high.field_type else 1.0
    index_penalty = 0.0 if cp_low.poincare_hopf_index == cp_high.poincare_hopf_index else 1.0
    return position_term + type_penalty + index_penalty


def check_basis_convergence(
    critical_points_by_level: Sequence[CriticalPoint], position_scale: float = 1.0
) -> ConfidenceBlock:
    """Doc §15 extraction precaution, schema note on `basis_set_convergence`:
    "было binary flag -- теперь величина расхождения + n уровней базиса,
    т.к. 'сошлось/не сошлось' на двух базисах -- это n=2, слабая оценка"
    ("was a binary flag -- now [should be] a magnitude of discrepancy + n
    levels of basis, since 'converged/not converged' on two basis sets is
    n=2, a weak estimate"). This replaces the previous hard agree/disagree
    boolean with `score = 1 / (1 + mean pairwise discrepancy)` -- a
    continuous stability measure computed via `basis_convergence_discrepancy`
    over every pair of the given levels -- and generalizes from exactly two
    levels to any n >= 2 the caller supplies, so `ConfidenceBlock.n` reports
    the actual number of levels compared instead of being permanently
    frozen at 2. `interval` records the (min, max) pairwise discrepancy, so
    a caller can see how much the mean is hiding, not just its value.

    Caller is still responsible for matching up which critical point at
    each level corresponds to which (e.g. nearest neighbor by position) --
    that correspondence problem is unchanged from before.
    """
    n = len(critical_points_by_level)
    if n < 2:
        raise ValueError("need at least 2 levels of theory to check convergence")

    discrepancies = [
        basis_convergence_discrepancy(critical_points_by_level[i], critical_points_by_level[j], position_scale)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    mean_discrepancy = float(np.mean(discrepancies))
    score = 1.0 / (1.0 + mean_discrepancy)
    interval = (float(min(discrepancies)), float(max(discrepancies)))

    return ConfidenceBlock(
        n=n,
        method="expert_provisional",
        score=score,
        interval=interval,
        corpus_version="n/a",
        stale=False,
    )


def find_critical_points_blind(
    field: ScalarField,
    bbox: Tuple[np.ndarray, np.ndarray],
    **kwargs,
) -> List[CriticalPoint]:
    """Find critical points without accepting atom/nucleus positions.

    This is the only discovery path used by the alphabet pipeline. It
    searches the scalar field by cube-grid local maxima (when available), a
    regular grid, and then field-derived attractor-pair/triple seeds. Every
    retained point is classified from local field topology. Atomic coordinates
    may be compared afterwards, but cannot alter candidate existence, type,
    descriptor, or class.
    """
    kwargs.pop("nuclear_seeds", None)
    return find_critical_points(field, bbox, nuclear_seeds=[], **kwargs)


def compare_seeded_vs_blind(
    field: ScalarField,
    bbox: Tuple[np.ndarray, np.ndarray],
    match_radius: float = 0.5,
    **kwargs,
) -> dict:
    """The actual §3.1/§3.2-honest experiment: run critical-point search
    TWICE on the same field -- once with nuclear-position seeding used only
    as starting points (`find_critical_points`) and
    once fully blind (`find_critical_points_blind`, no reference to nuclear
    coordinates anywhere) -- and report how the two nuclear_attractor sets
    actually compare, instead of assuming the seeded run's answer is
    correct because it usually finds "the right number".

    Three things this distinguishes, all collapsed into one number by only
    looking at counts:
      - recovered:      atom positions with a blind attractor within
                         `match_radius` -- topology genuinely supports an
                         attractor there, independent of being told to look.
      - missed:          atom positions with NO nearby blind attractor --
                         a field-search coverage limitation or a genuine
                         mismatch, worth flagging PER
                         position, not just counting, since a systematic
                         pattern here (e.g. always a specific element, or
                         always atoms in a delocalized ring) is itself a
                         finding, not just a search-parameter shortfall.
      - novel:           blind attractors with NO atom within
                         `match_radius` -- topology finds a genuine
                         maximum where no nucleus was declared. This is the
                         single most interesting possible outcome
                         mentioned in the 2026-07 discussion: a minimal
                         topological unit that does not coincide with an
                         atom at all (sub-atomic lobe, or a region spanning
                         parts of several atoms' basins) would show up
                         here, not in `recovered`.

    Returns a dict with `recovered`, `missed` (each a list of atom
    positions), `novel` (a list of CriticalPoint), and `n_atoms`/
    `n_blind_attractors` for context. A field with no `atom_positions`
    (anything but a real cube-file-backed density) raises -- this
    comparison is meaningless without independently-known ground truth to
    compare the blind search against.
    """
    atom_positions = getattr(field, "atom_positions", None)
    if atom_positions is None or len(atom_positions) == 0:
        raise ValueError(
            "field has no atom_positions -- compare_seeded_vs_blind needs "
            "independently-known nuclear coordinates as ground truth to "
            "compare the blind search against, not just a field to search"
        )
    atom_positions = np.asarray(atom_positions, dtype=float)

    # This seeded pass is an external-reference audit only.  Its output is
    # returned for comparison and is never mixed into the field-first corpus.
    seeded = find_critical_points(field, bbox, nuclear_seeds=atom_positions, **kwargs)
    blind = find_critical_points_blind(field, bbox, **kwargs)
    blind_attractors = [cp for cp in blind if cp.field_type == "nuclear_attractor"]
    blind_positions = (
        np.stack([cp.position for cp in blind_attractors])
        if blind_attractors
        else np.zeros((0, 3))
    )

    recovered, missed = [], []
    for a in atom_positions:
        if len(blind_positions):
            d = np.linalg.norm(blind_positions - a, axis=1)
            (recovered if d.min() <= match_radius else missed).append(a.copy())
        else:
            missed.append(a.copy())

    novel = []
    for cp in blind_attractors:
        d = np.linalg.norm(atom_positions - cp.position, axis=1)
        if d.min() > match_radius:
            novel.append(cp)

    return {
        "n_atoms": len(atom_positions),
        "n_blind_attractors": len(blind_attractors),
        "n_seeded_attractors": sum(1 for cp in seeded if cp.field_type == "nuclear_attractor"),
        "recovered": recovered,
        "missed": missed,
        "novel": novel,
    }
