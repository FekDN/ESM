"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
unsupervised_validation.py
===========================
Doc §16.2 ("Валидация без супервизии") -- the one entry in the doc's
epistemology section (§16) not yet built anywhere in this scaffold. §16.1 is
a problem statement, not code; §16.3's orphan search is `semantics.py`'s job
once class-level correlation exists; §16.4 is a later screening protocol,
out of scope the same way the rest of steps 7-11 are (see `__init__.py`).
§16.2 needs no new data -- it is assembled entirely out of pieces this
scaffold already has: `fields.py`'s backends, `pipeline.py`'s extraction,
and `fundamentality.py`'s §12 compressibility machinery.

The doc's criterion, precisely: besides checking whether the invariant
correlates with already-named properties, check a purely structural
criterion that presupposes nothing about what the structure is -- does the
invariant descriptor space compress (§12) BETTER than a control model built
on "scrambled or random patches of the field of the same scale"? This tests
for the presence of real structure at all, without prejudging what it will
turn out to be (§16.2: "проверяет наличие реальной структуры вообще, не
предрешая, какой именно она окажется").

This module is exactly the piece `fundamentality.py`'s module docstring
names and defers: "compare corpus compressibility against a
scrambled/randomized control FIELD of the same scale... a separate
cross-check the doc keeps apart from §12's own criterion." The
reuse-breadth permutation null already in `fundamentality.py` answers a
different question -- is a candidate's atom set shared across *real*
molecules more than a random relabeling of those same real molecules would
predict? This module asks a strictly prior question: is the real corpus
compressible at all, relative to structureless data of the same amplitude
scale? A corpus that failed this check would make `fundamentality.py`'s
whole reuse analysis moot -- there'd be nothing structured to reuse.

Pipeline:
    1. `fields.ScrambledFieldControl` -- turns any reference `ScalarField`
       into a null field "of the same scale" (see that class's docstring).
       Two selectable, doc-mandated constructions: `mode="scrambled"`
       (permute a sampled grid's values) or `mode="random"` (i.i.d.
       resample from that same empirical distribution).
    2. `build_control_corpus` -- runs the SAME `pipeline.extract_letter_candidates`
       (critical-point search + invariant descriptor) on a
       `ScrambledFieldControl` built from each real molecule's own
       field/bbox, so the control is put through the identical machinery
       the real corpus went through, not just superficially similar data
       fed to a different procedure. This is what makes the check a test of
       whether the WHOLE METHOD finds real structure, not a comparison of
       two unrelated numbers.
    3. `compressibility_statistic` -- reduces a candidate pool to the single
       scalar §16.2 compares: mean §12 reconstruction fidelity under one
       shared-dictionary fit at a FIXED model capacity
       (`fundamentality.FundamentalityConfig`), so real vs. control is a
       comparison of achieved compression at matched capacity, not of
       capacity itself.
    4. `validate_structure_against_control` -- the §16.2 entry point:
       repeats steps 2-3 `n_controls` times (independent randomizations) to
       build a null distribution of the control's compressibility
       statistic, and reports the real corpus's statistic against that null
       band as a `schema.ConfidenceBlock` -- the same "beat a matched null,
       not an arbitrary constant" idiom `commutation.py`'s random-direction
       null and `fundamentality.reuse_breadth_confidences`'s permutation
       null already use elsewhere in this codebase. Real structure is
       supported when `score` clears `interval[1]`, exactly the convention
       `fundamentality.compute_fundamentality` already applies to its own
       null.

Left out of scope, same caveat as everywhere else in this scaffold: this
checks compressibility of the *candidate pool as a whole*, not any one
`corpus.LetterClass`'s cluster-level fidelity, and it says nothing about
*which* real structure was found -- only that the extraction pipeline is
not just finding artifacts of its own machinery. That's §16.3/§16.4's job,
downstream of this check having passed.
"""

from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

import numpy as np

from .corpus import LetterCandidate
from .fields import ScalarField, ScrambledFieldControl
from .fundamentality import FundamentalityConfig, fit_shared_dictionary, reconstruction_fidelity
from .pipeline import extract_letter_candidates
from .schema import ConfidenceBlock


def build_control_corpus(
    molecules: Sequence[Tuple[str, ScalarField, Tuple[np.ndarray, np.ndarray]]],
    control_mode: Literal["scrambled", "random"] = "scrambled",
    grid_resolution: int = 10,
    rng_seed=0,
    shell_radii: Sequence[float] = (0.5, 0.9, 1.3),
    r0: float = 1.1,
    sigma: float = 0.25,
    l_max: int = 4,
    n_points_per_shell: int = 150,
    cp_kwargs: dict | None = None,
) -> List[LetterCandidate]:
    """Run `pipeline.extract_letter_candidates` -- unchanged -- on a
    `ScrambledFieldControl` built from each real (molecule_id, field, bbox)
    triple, so the returned candidate pool went through exactly the same
    critical-point search and invariant-descriptor construction the real
    corpus did, just on data with the real spatial structure destroyed.

    `rng_seed` may be a plain int or a `numpy.random.SeedSequence`; either
    way it is spawned once per molecule via `SeedSequence.spawn`, giving
    every molecule's control field an independent, reproducible seed
    without any ad hoc seed arithmetic -- `validate_structure_against_control`
    relies on this to hand each of its `n_controls` repetitions a distinct
    child `SeedSequence` and get, in turn, a distinct seed per molecule
    within that repetition.

    Control candidates are tagged with molecule id `f"control::{molecule_id}"`
    (not the real id) so a caller who accidentally pools real and control
    candidates together will not have them silently merge under
    `fundamentality.py`'s per-molecule reuse-breadth accounting.
    """
    cp_kwargs = cp_kwargs or {}
    seed_seq = rng_seed if isinstance(rng_seed, np.random.SeedSequence) else np.random.SeedSequence(rng_seed)
    molecule_seeds = seed_seq.spawn(len(molecules))

    control_candidates: List[LetterCandidate] = []
    for (molecule_id, field, bbox), mseed in zip(molecules, molecule_seeds):
        control_field = ScrambledFieldControl(
            reference=field,
            bbox=bbox,
            grid_resolution=grid_resolution,
            mode=control_mode,
            rng_seed=mseed,
        )
        control_candidates.extend(
            extract_letter_candidates(
                control_field,
                bbox,
                molecule_id=f"control::{molecule_id}",
                shell_radii=shell_radii,
                r0=r0,
                sigma=sigma,
                l_max=l_max,
                n_points_per_shell=n_points_per_shell,
                cp_kwargs=cp_kwargs,
            )
        )
    return control_candidates


def compressibility_statistic(
    candidates: Sequence[LetterCandidate], config: FundamentalityConfig
) -> float:
    """§12's own measure, reduced to the single scalar §16.2 compares: mean
    reconstruction fidelity of the whole pool under one shared-dictionary
    fit at `config`'s fixed capacity (`n_atoms`, `n_nonzero_coefs`).

    Returns 0.0 for a pool too small to fit a dictionary at all (fewer than
    2 candidates -- e.g. a control field whose critical-point search
    happened to find almost nothing). A pool that cannot even be fit is, by
    construction, not compressible; this is the correct value to feed into
    the null distribution below, not an error to propagate. If control
    pools come out this small often, see the caveat in
    `validate_structure_against_control`'s docstring.
    """
    if len(candidates) < 2:
        return 0.0
    fit = fit_shared_dictionary(candidates, config)
    vectors = np.stack([c.descriptor.to_vector() for c in candidates])
    fidelities = [
        reconstruction_fidelity(vectors[i], fit.codes[i], fit.dictionary)
        for i in range(len(candidates))
    ]
    return float(np.mean(fidelities))


def validate_structure_against_control(
    molecules: Sequence[Tuple[str, ScalarField, Tuple[np.ndarray, np.ndarray]]],
    real_candidates: Sequence[LetterCandidate],
    dictionary_config: FundamentalityConfig,
    control_mode: Literal["scrambled", "random"] = "scrambled",
    n_controls: int = 20,
    grid_resolution: int = 10,
    shell_radii: Sequence[float] = (0.5, 0.9, 1.3),
    r0: float = 1.1,
    sigma: float = 0.25,
    l_max: int = 4,
    n_points_per_shell: int = 150,
    cp_kwargs: dict | None = None,
    rng_seed: int = 0,
    corpus_version: str = "unversioned",
) -> ConfidenceBlock:
    """§16.2, the module's entry point: is the real corpus's invariant space
    more compressible (§12) than `n_controls` independent
    scrambled/random-field controls of the same scale?

    `molecules` must be the same (molecule_id, field, bbox) triples used to
    build `real_candidates` (e.g. exactly what was passed to
    `pipeline.decode_alphabet`) -- this is what lets each control be built
    at the correct bbox/scale per molecule, rather than one global guess.

    `dictionary_config` is used UNCHANGED for the real corpus and for every
    control repetition (same `n_atoms` / `n_nonzero_coefs`), so the
    comparison is of achieved compression at matched model capacity, not of
    capacity itself. Pick `n_atoms` relative to whichever of
    (len(real_candidates), a typical control pool size) is smaller -- the
    same failure mode `fundamentality.suggest_n_atoms` already warns about
    (a dictionary with about as many atoms as candidates trivially achieves
    near-perfect fidelity for everyone) applies to whichever side happens to
    have fewer candidates, and control pools are exactly the side likely to
    be small or unpredictable in size (a noisy, interpolated control field
    can yield far fewer -- or, from spurious numerical noise, more --
    discoverable critical points than the real field it was built from).

    Returns a `ConfidenceBlock` (`method="permutation"`, the same
    established convention as `fundamentality.reuse_breadth_confidences`
    and `commutation`'s random-direction null): `score` is the real
    corpus's `compressibility_statistic`; `interval` is the (2.5, 97.5)
    percentile band of that same statistic across the `n_controls`
    independent control corpora. Per the doc's framing, real structure is
    supported when `score` clears `interval[1]` -- the real corpus
    compresses better than essentially all randomizations of data at the
    same scale, not merely better on average.

    Caveat: control pool sizes are not guaranteed to match `real_candidates`
    and commonly don't -- in practice a scrambled/random grid is rougher
    than a real field (its interpolant has a kink at every grid cell
    boundary), so multi-start Newton often finds MORE spurious critical
    points in a control than in the real field it was built from, not
    fewer. A much larger, incoherent control pool is informative and fine
    (it is exactly what a low `compressibility_statistic` should reflect).
    The failure mode worth checking for directly is the opposite: a control
    pool so small (0 or 1 candidates) that `compressibility_statistic`
    returns a hard 0.0, which biases the null band low and can make the
    real corpus look more structured by comparison than it is. Inspect pool
    sizes via `build_control_corpus` directly if the returned interval looks
    suspiciously tight against 0.0.

    This is the most expensive check in the whole pipeline: each of
    `n_controls` repetitions reruns full multi-start-Newton critical-point
    search plus invariant-descriptor construction on every molecule, then a
    fresh dictionary-learning fit. Reduce `n_controls`, `grid_resolution`,
    or `cp_kwargs['n_seeds_per_axis']` if this is too slow for a large
    corpus -- the same runtime/precision trade-off `n_bootstrap` and
    `n_null_permutations` already represent elsewhere in this codebase.

    OBSERVABILITY NOTE (fixed): this loop used to run silently for its
    entire duration -- on a real, fixed-Hessian corpus (see fields.py) each
    molecule now legitimately yields many more real critical points than
    before, and a scrambled/random control field routinely finds even MORE
    (its interpolant has a kink at every grid cell boundary, so multi-start
    Newton has more spurious extrema to converge to -- see the caveat
    above). Combined with a control pool sized as a fraction of a large
    corpus, a single repetition can now legitimately take much longer than
    it used to, with nothing printed in between -- indistinguishable from a
    hang from the caller's side. Each repetition now logs its own timing
    and candidate count as it finishes, so a slow-but-progressing run is
    visibly different from a stuck one.
    """
    import logging
    import time

    real_score = compressibility_statistic(real_candidates, dictionary_config)

    master_seed_seq = np.random.SeedSequence(rng_seed)
    control_seeds = master_seed_seq.spawn(n_controls)

    logging.info(
        f"§16.2: старт {n_controls} контрольных повторений "
        f"({len(molecules)} молекул каждое, mode={control_mode})."
    )
    null_scores = np.zeros(n_controls)
    for b in range(n_controls):
        t0 = time.monotonic()
        control_candidates = build_control_corpus(
            molecules,
            control_mode=control_mode,
            grid_resolution=grid_resolution,
            rng_seed=control_seeds[b],
            shell_radii=shell_radii,
            r0=r0,
            sigma=sigma,
            l_max=l_max,
            n_points_per_shell=n_points_per_shell,
            cp_kwargs=cp_kwargs,
        )
        null_scores[b] = compressibility_statistic(control_candidates, dictionary_config)
        elapsed = time.monotonic() - t0
        logging.info(
            f"§16.2: повторение {b + 1}/{n_controls} завершено за {elapsed:.1f}с "
            f"({len(control_candidates)} контрольных кандидатов, score={null_scores[b]:.4f})."
        )

    lo, hi = np.percentile(null_scores, [2.5, 97.5])
    return ConfidenceBlock(

        n=n_controls,
        method="permutation",
        score=real_score,
        interval=(float(lo), float(hi)),
        corpus_version=corpus_version,
        stale=False,
    )
