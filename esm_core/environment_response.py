"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
environment_response.py
=======================
Doc §4.3 ("Индексикальность контекста: систематическая, а не случайная
модуляция значения").

Evaluates whether a class's realization across different environments
(e.g. changing dielectric constant) shifts smoothly (indexicality - "character"
remains the same while "content" shifts) or jumps abruptly (discontinuous polysemy -
a qualitative change in meaning/mechanism).

Per the doc warning: "КРИТИЧНО: при малом числе точек сетки (§4.3) классификация
ненадёжна" -- this is strictly enforced by falling back to "insufficient_data"
if the grid is too sparse to distinguish a curve from a jump.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from .schema import (
    EnvironmentResponseSample,
    EnvironmentResponseProfile,
    ResponseSmoothness,
    ConfidenceBlock,
)

def analyze_response_smoothness(
    samples: Sequence[EnvironmentResponseSample],
    jump_threshold_multiplier: float = 2.5,
    min_monotonicity: float = 0.6,
    corpus_version: str = "unversioned"
) -> ResponseSmoothness:
    """
    Analyzes an ordered sequence of environmental samples to populate the
    ResponseSmoothness block. Samples must be pre-sorted by the driving
    environmental parameter (e.g. sequentially increasing permittivity).
    """
    n_samples = len(samples)
    vectors = [s.invariant_vector for s in samples if s.invariant_vector is not None]
    
    if len(vectors) < 3:
        # Not enough points to establish a trend vs a jump
        return ResponseSmoothness(
            derivative_profile=[],
            monotonicity_score=0.0,
            classification="insufficient_data",
            classification_confidence=ConfidenceBlock(
                n=len(vectors), method="analytic_CI", score=0.0,
                corpus_version=corpus_version
            )
        )

    # 1. Derivative profile (step distances)
    vecs = np.stack(vectors)
    diffs = vecs[1:] - vecs[:-1]
    step_distances = np.linalg.norm(diffs, axis=1)
    
    # 2. Monotonicity score: straight line distance / sum of path distances
    total_path_len = np.sum(step_distances)
    net_displacement = np.linalg.norm(vecs[-1] - vecs[0])
    
    if total_path_len < 1e-12:
        monotonicity = 1.0
    else:
        monotonicity = float(net_displacement / total_path_len)
        
    # 3. Discontinuity detection (outlier in step distances)
    median_step = np.median(step_distances)
    max_step = np.max(step_distances)
    
    is_discontinuous = False
    if median_step > 1e-12 and max_step > jump_threshold_multiplier * median_step:
        is_discontinuous = True
        
    if monotonicity < min_monotonicity:
        is_discontinuous = True
        
    classification = "discontinuous_polysemy" if is_discontinuous else "indexical"
    
    confidence = ConfidenceBlock(
        n=len(vectors),
        method="expert_provisional",
        score=1.0 - (1.0 / len(vectors)),  # Confidence grows asymptotically with grid density
        corpus_version=corpus_version
    )
    
    return ResponseSmoothness(
        derivative_profile=step_distances.tolist(),
        monotonicity_score=monotonicity,
        classification=classification,
        classification_confidence=confidence
    )


def update_environment_response_profile(
    profile: EnvironmentResponseProfile,
    jump_threshold_multiplier: float = 2.5,
    min_monotonicity: float = 0.6,
    corpus_version: str = "unversioned"
) -> EnvironmentResponseProfile:
    """
    Doc §4.3/§13.4, schema.py's `EnvironmentResponseProfile` docstring: the
    one function allowed to set `profile.response_smoothness`. Runs
    `analyze_response_smoothness` over `profile.samples` and replaces the
    nested block atomically (schema.py is explicit about why this must be
    one atomic replace rather than four separately-settable fields:
    `derivative_profile`, `monotonicity_score`, `classification` and
    `classification_confidence` are all derived TOGETHER from the same
    `samples` by the same call, and setting them one at a time would let a
    caller observe -- or leave behind -- a `response_smoothness` whose
    fields don't agree with each other or with the `samples` that produced
    them).

    Mutates `profile` in place (so any other reference to the same
    `EnvironmentResponseProfile` sees the update) and also returns it, so
    this composes into a pipeline the same way
    `semantics.update_semantic_status` does for `LetterClass`.
    """
    profile.response_smoothness = analyze_response_smoothness(
        profile.samples,
        jump_threshold_multiplier=jump_threshold_multiplier,
        min_monotonicity=min_monotonicity,
        corpus_version=corpus_version,
    )
    return profile
