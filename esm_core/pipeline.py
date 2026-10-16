"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
pipeline.py
===========
Top-level entry points for "letter decoding" -- wires together:

    fields.ScalarField            (what to read the field from)
      -> critical_points.find_critical_points     (§15 step 1)
      -> invariants.build_invariant_descriptor     (§15 step 2)
      -> corpus.cluster_candidates_bootstrap        (§15 step 3)
      -> schema.LetterClass                         (the decoded alphabet)

Two entry points:
    extract_letter_candidates -- run steps 1-2 on a single molecule/field.
    decode_alphabet           -- run steps 1-2 across an entire corpus of
                                  molecules, then step 3: cluster into letter classes.

Everything downstream of `LetterClass` (semantic correlation §16, commutation
testing §11/step 4, dynamics §8-10/steps 7-11) is out of this scaffold's scope -- see
schema.py's module docstring for exactly what is and isn't implemented.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .corpus import LetterCandidate, cluster_candidates_bootstrap
from .critical_points import find_critical_points_blind
from .fields import ScalarField
from .invariants import (
    ContextualInvariantDescriptor,
    build_invariant_descriptor,
    build_relational_topology_vectors,
)
from .schema import LetterClass


def extract_letter_candidates(
    field: ScalarField,
    bbox: Tuple[np.ndarray, np.ndarray],
    molecule_id: str,
    shell_radii: Sequence[float] = (0.5, 0.9, 1.3),
    r0: float = 1.1,
    sigma: float = 0.25,
    l_max: int = 4,
    n_points_per_shell: int = 150,
    cp_kwargs: dict | None = None,
) -> List[LetterCandidate]:
    """Doc §15 steps 1-2, run end to end on one molecule: find critical points,
    then build a rotation-invariant descriptor around each one.

    Candidate discovery is deliberately field-first: atom/nucleus coordinates
    are not passed as seeds and cannot create, type, describe, or classify a
    letter.  They may be attached later as independent chemical annotations.

    ``cp_kwargs["descriptor_regime"]`` selects a falsifiable representation
    hypothesis: ``local_field`` (default), ``relational_topology`` (only the
    arrangement of independently field-discovered landmarks), or
    ``balanced_combined`` (both blocks with per-realisation RMS balancing).
    It is removed before critical-point search and is recorded by the main
    runner; it is not a hidden search prior or grammar rule.

    `shell_radii` / `r0` / `sigma` are physically meaningful and should be set
    relative to the field's natural length scale (e.g. typical bond length)
    -- the defaults are tuned for the toy `GaussianMixtureDensity` demo, NOT
    for a real electron density in Bohr or Angstrom; a production caller must
    pass values appropriate to their field's units and level of theory.
    """
    cp_kwargs = dict(cp_kwargs or {})
    descriptor_regime = str(cp_kwargs.pop("descriptor_regime", "local_field"))
    if descriptor_regime not in {
        "local_field", "relational_topology", "balanced_combined",
    }:
        raise ValueError(
            "descriptor_regime must be local_field, relational_topology, or balanced_combined"
        )
    # Be strict rather than silently accepting an atom-conditioned request.
    # The blind routine would discard this key, but an error makes an invalid
    # analysis configuration visible in the report/log instead of merely
    # giving the appearance that it influenced a field-only extraction.
    prohibited = {"nuclear_seeds", "nuclear_seed_fallback"} & set(cp_kwargs)
    if prohibited:
        raise ValueError(
            "extract_letter_candidates is field-first; atom/nucleus seed options "
            f"are prohibited: {sorted(prohibited)}"
        )
    critical_points = find_critical_points_blind(field, bbox, **cp_kwargs)
    relational_vectors, relational_components = build_relational_topology_vectors(
        critical_points, shell_radii
    )

    candidates = []
    for critical_point_index, cp in enumerate(critical_points):
        if cp.field_type == "degenerate":
            continue  # §15 precaution: not trusted as a physical feature
        local_descriptor = build_invariant_descriptor(
            field, cp.position, shell_radii, r0, sigma, l_max, n_points_per_shell
        )
        descriptor = (
            local_descriptor if descriptor_regime == "local_field" else
            ContextualInvariantDescriptor(
                local_descriptor=local_descriptor,
                relational_vector=relational_vectors[critical_point_index],
                relational_components=relational_components,
                regime=descriptor_regime,
            )
        )
        candidates.append(
            LetterCandidate(
                source_molecule_id=molecule_id,
                critical_point=cp,
                descriptor=descriptor,
                environment_condition=field.environment,
            )
        )
    return candidates


def decode_alphabet(
    molecules: Sequence[Tuple[str, ScalarField, Tuple[np.ndarray, np.ndarray]]],
    distance_threshold: float,
    shell_radii: Sequence[float] = (0.5, 0.9, 1.3),
    r0: float = 1.1,
    sigma: float = 0.25,
    l_max: int = 4,
    n_points_per_shell: int = 150,
    n_bootstrap: int = 200,
    cp_kwargs: dict | None = None,
    corpus_version: str = "unversioned",
) -> Tuple[List[LetterClass], List[LetterCandidate]]:
    """Doc §15 steps 1-3 across a corpus: run steps 1-2 per molecule via
    `extract_letter_candidates` on every (molecule_id, field, bbox) triple,
    pool all candidates, cluster into letter classes with bootstrap
    `cluster_stability`. Returns (classes, all_candidates) -- the candidates
    are returned too because they're what you need to populate a class's
    realization_registry (schema.py) or to inspect which molecules
    contributed to a given class.
    """
    all_candidates: List[LetterCandidate] = []
    for molecule_id, field, bbox in molecules:
        all_candidates.extend(
            extract_letter_candidates(
                field, bbox, molecule_id, shell_radii, r0, sigma, l_max,
                n_points_per_shell, cp_kwargs,
            )
        )

    classes = cluster_candidates_bootstrap(
        all_candidates,
        distance_threshold=distance_threshold,
        l_max_bispectrum=l_max,
        n_bootstrap=n_bootstrap,
        corpus_version=corpus_version,
    )
    return classes, all_candidates
