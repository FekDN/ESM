"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
invariants.py
=============
Doc §15, step 2: build a rotation-invariant "fingerprint" of
the field around a critical point, comparable across different molecules
without aligning them or picking a local coordinate frame.

Pipeline for one critical point:
    1. sample the field on several concentric spherical shells around it,
       using a *soft* radial window (§15 precaution: "мягкие, а не жёсткие
       границы регионов") rather than a hard cutoff -- neighbouring regions'
       contributions fade smoothly instead of being clipped;
    2. fit each shell's samples to real spherical harmonics up to degree
       `l_max` (least-squares -- the sample directions are a Fibonacci-sphere
       lattice, not an exact quadrature grid, so this is an approximation
       that improves with more samples per shell; see `sample_shell`);
    3. build two rotation-invariant contractions of the per-shell SH
       coefficients:
         - power spectrum p_l(shell_i, shell_j) = sum_m c_{l,m}^(i) * conj(c_{l,m}^(j))
           (2nd order; provably invariant under simultaneous rotation because
           each l-subspace carries a unitary irrep of SO(3), so this inner
           product is preserved -- this holds shell-by-shell and cross-shell,
           since the same rotation acts identically on every shell);
         - bispectrum B_{l1,l2,l3} = sum_{m1,m2,m3} conj(c_{l3,m3}) *
           CG(l1,m1; l2,m2 | l3,m3) * c_{l1,m1} * c_{l2,m2}, for l3 in the
           allowed CG triangle |l1-l2| <= l3 <= l1+l2, l3 <= l_max (3rd order;
           the standard bispectrum invariant used in SO(3) atomic-environment
           descriptors, e.g. Bartok et al. 2013's local bispectrum -- also
           provably invariant, and retains phase information the power
           spectrum discards).

Rotation invariance of both is checked directly in
`tests/test_rotation_invariance.py` (numeric, not just asserted) -- for a
"maximally accurate" scaffold the invariant descriptor is exactly the piece
that must not be taken on faith.

`InvariantDescriptor.to_vector()` concatenates power spectrum + bispectrum
into the doc schema's fixed-length `bispectrum_vector`; `l_max` used is
recorded in `schema.DiscoveryProvenance.l_max_bispectrum`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import sph_harm_y

from .fields import ScalarField


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """n_points roughly-uniformly distributed unit vectors on S^2 (golden-
    angle spiral). Avoids the pole clustering of a naive (theta, phi) grid,
    which would otherwise bias the least-squares SH fit."""
    i = np.arange(n_points)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    radius_xy = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    theta_azimuth = golden_angle * i
    x = radius_xy * np.cos(theta_azimuth)
    y = radius_xy * np.sin(theta_azimuth)
    return np.stack([x, y, z], axis=1)


def soft_radial_window(r: np.ndarray, r0: float, sigma: float) -> np.ndarray:
    """Smooth (Fermi-Dirac-style) falloff weight, used so a region's boundary
    fades rather than being a hard mask (§15 precaution). w(r0)=0.5; w -> 1
    for r << r0, w -> 0 for r >> r0. `sigma` sets the width of the transition."""
    return 1.0 / (1.0 + np.exp((r - r0) / sigma))


@dataclass
class ShellSample:
    radius: float
    directions: np.ndarray  # (n, 3) unit vectors
    values: np.ndarray  # (n,) field values, already soft-windowed
    weight: float  # this shell's overall soft-window weight at its radius


def sample_shell(
    field: ScalarField, center: np.ndarray, radius: float, n_points: int,
    r0: float, sigma: float,
) -> ShellSample:
    """Sample `field` on a Fibonacci-sphere shell of the given radius around
    `center`, weighting the whole shell by the soft radial window at that
    radius (§15: the boundary is used only to set centroid/scale, not as a
    binary mask -- weighting shells this way is how that principle is
    implemented for a discretized multi-shell sampling)."""
    directions = fibonacci_sphere(n_points)
    values = np.array([field.value(center + radius * d) for d in directions])
    weight = float(soft_radial_window(np.array([radius]), r0, sigma)[0])
    return ShellSample(radius=radius, directions=directions, values=values * weight, weight=weight)


# --------------------------------------------------------------------------
# spherical harmonic fit
# --------------------------------------------------------------------------


def _directions_to_angles(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cartesian unit vectors -> (theta polar in [0,pi], phi azimuth in [0,2pi)),
    matching scipy's `sph_harm_y(n, m, theta, phi)` convention."""
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    return theta, phi


def _sh_design_matrix(theta: np.ndarray, phi: np.ndarray, l_max: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Design matrix Y[n_samples, n_coeffs] of Y_l^m at each sample direction,
    plus the (l, m) index for each column, ordered l=0..l_max, m=-l..l."""
    index: List[Tuple[int, int]] = []
    cols = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            index.append((l, m))
            cols.append(sph_harm_y(l, m, theta, phi))
    Y = np.stack(cols, axis=1)
    return Y, index


def fit_sh_coefficients(shell: ShellSample, l_max: int) -> dict:
    """Least-squares fit c_{l,m} such that sum_{l,m} c_{l,m} Y_l^m(theta,phi)
    ~= shell.values at the sampled directions. Returns {(l,m): complex c_lm}.

    Least squares (rather than exact quadrature) because a Fibonacci-sphere
    lattice is not a Gauss-Legendre / Lebedev quadrature grid; this is exact
    only in the limit of many samples relative to (l_max+1)^2 coefficients --
    keep n_points well above (l_max+1)^2 (a factor of ~4-6x is a reasonable
    default) or the fit is under-determined and coefficients (and hence the
    invariants built from them) become sample-grid-dependent.
    """
    theta, phi = _directions_to_angles(shell.directions)
    Y, index = _sh_design_matrix(theta, phi, l_max)
    coeffs, *_ = np.linalg.lstsq(Y, shell.values, rcond=None)
    return dict(zip(index, coeffs))


# --------------------------------------------------------------------------
# Clebsch-Gordan coefficients (cached; sympy gives exact rationals/surds)
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _cg(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
    from sympy.physics.quantum.cg import CG

    return float(CG(l1, m1, l2, m2, l3, m3).doit())


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


@dataclass
class InvariantDescriptor:
    l_max: int
    shell_radii: Tuple[float, ...]
    power_spectrum: dict  # {(l, i, j): complex}  i,j index into shell_radii
    bispectrum: dict  # {(l1, l2, l3, shell): complex}

    def component_layout(self) -> list[dict]:
        """Names in exactly the same order as :meth:`to_vector`.

        Keeping the layout beside the vector construction lets an archive
        distinguish a local-field-only descriptor from a later relational
        field-context descriptor without guessing from its length.
        """
        ps_keys = sorted(self.power_spectrum)
        components = []
        components.extend({"kind": "power_diagonal_real", "key": list(key)} for key in ps_keys if key[1] == key[2])
        components.extend({"kind": "power_cross_real", "key": list(key)} for key in ps_keys if key[1] != key[2])
        components.extend({"kind": "power_cross_imag", "key": list(key)} for key in ps_keys if key[1] != key[2])
        components.extend({"kind": "bispectrum_real", "key": list(key)} for key in sorted(self.bispectrum))
        components.extend({"kind": "bispectrum_imag", "key": list(key)} for key in sorted(self.bispectrum))
        return components

    def to_vector(self) -> np.ndarray:
        """Fixed-length real vector: doc schema's `invariant_centroid` /
        `bispectrum_vector[fixed_length]`. Ordering is deterministic (sorted
        keys) so vectors from different molecules are directly comparable
        component-by-component.

        Same-shell power-spectrum terms (i == j) are always real by
        construction (sum_m c_lm * conj(c_lm) = sum_m |c_lm|^2), so only
        the real part is meaningful there. Cross-shell terms (i != j) are
        complex in general -- and, per the module docstring, the *entire*
        complex inner product p_l(i,j) is rotation-invariant (both its real
        and imaginary parts survive simultaneous rotation exactly, since
        the same unitary Wigner-D acts on both shells' coefficients).
        Keeping only `.real` there was silently discarding a real,
        non-trivial invariant signal that the bispectrum below does not
        otherwise cover -- the bispectrum already keeps both real and
        imaginary parts for exactly this reason, so the power spectrum is
        treated the same way here for consistency and to avoid losing
        discriminating power in the fingerprint.
        """
        ps_keys = sorted(self.power_spectrum)
        ps_diag = [self.power_spectrum[k].real for k in ps_keys if k[1] == k[2]]
        ps_cross_re = [self.power_spectrum[k].real for k in ps_keys if k[1] != k[2]]
        ps_cross_im = [self.power_spectrum[k].imag for k in ps_keys if k[1] != k[2]]
        bs_re = [self.bispectrum[k].real for k in sorted(self.bispectrum)]
        bs_im = [self.bispectrum[k].imag for k in sorted(self.bispectrum)]
        return np.array(ps_diag + ps_cross_re + ps_cross_im + bs_re + bs_im, dtype=float)


@dataclass
class ContextualInvariantDescriptor:
    """A transparent multi-view descriptor for a field landmark.

    The local SO(3)-invariant field fingerprint and an independently built
    topology-of-neighbours fingerprint are retained separately.  ``regime``
    chooses which hypothesis is sent to clustering:

    * ``local_field``: local scalar-field shape only;
    * ``relational_topology``: position/arrangement among other *field*
      landmarks only; and
    * ``balanced_combined``: both blocks after per-realisation RMS
      normalisation, so raw field magnitude cannot drown out relational
      structure merely by having larger numerical units.

    This is not a claim that any regime is the true definition of a letter.
    It makes those competing assumptions explicit and reproducible.
    """

    local_descriptor: InvariantDescriptor
    relational_vector: np.ndarray
    relational_components: Tuple[str, ...]
    regime: str = "local_field"

    @property
    def l_max(self) -> int:
        return self.local_descriptor.l_max

    @property
    def shell_radii(self) -> Tuple[float, ...]:
        return self.local_descriptor.shell_radii

    @property
    def power_spectrum(self) -> dict:
        return self.local_descriptor.power_spectrum

    @property
    def bispectrum(self) -> dict:
        return self.local_descriptor.bispectrum

    @staticmethod
    def _rms_normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        scale = float(np.sqrt(np.mean(vector * vector))) if len(vector) else 0.0
        if not np.isfinite(scale) or scale <= 1e-12:
            return np.zeros_like(vector)
        return vector / scale

    def to_vector(self) -> np.ndarray:
        local = self.local_descriptor.to_vector()
        relational = np.asarray(self.relational_vector, dtype=float)
        if self.regime == "local_field":
            return local
        if self.regime == "relational_topology":
            return relational
        if self.regime == "balanced_combined":
            return np.concatenate((self._rms_normalize(local), self._rms_normalize(relational)))
        raise ValueError(f"unknown descriptor regime: {self.regime!r}")

    def component_layout(self) -> list[dict]:
        local = self.local_descriptor.component_layout()
        relational = [
            {"kind": "relational_topology", "name": name}
            for name in self.relational_components
        ]
        if self.regime == "local_field":
            return local
        if self.regime == "relational_topology":
            return relational
        if self.regime == "balanced_combined":
            return [
                {**component, "block": "local_field", "normalization": "per_realisation_rms"}
                for component in local
            ] + [
                {**component, "block": "relational_topology", "normalization": "per_realisation_rms"}
                for component in relational
            ]
        raise ValueError(f"unknown descriptor regime: {self.regime!r}")


def power_spectrum(coeffs_by_shell: Sequence[dict], l_max: int) -> dict:
    """p_l(i, j) = sum_m c_{l,m}^(i) * conj(c_{l,m}^(j)) for every shell pair
    (i, j) including i == j. Rotation-invariant: see module docstring."""
    n_shells = len(coeffs_by_shell)
    out = {}
    for l in range(l_max + 1):
        ms = range(-l, l + 1)
        for i in range(n_shells):
            for j in range(i, n_shells):
                s = sum(
                    coeffs_by_shell[i][(l, m)] * np.conj(coeffs_by_shell[j][(l, m)])
                    for m in ms
                )
                out[(l, i, j)] = s
    return out


def bispectrum(coeffs_by_shell: Sequence[dict], l_max: int) -> dict:
    """B_{l1,l2,l3} per shell (bispectrum computed within a single shell's
    coefficients -- cross-shell bispectra are a valid but combinatorially
    larger extension, left out of this scaffold to keep descriptor length
    manageable; the power spectrum above already covers cross-shell terms)."""
    out = {}
    for shell_idx, coeffs in enumerate(coeffs_by_shell):
        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                    total = 0.0 + 0.0j
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            m3 = m1 + m2
                            if abs(m3) > l3:
                                continue
                            cg = _cg(l1, m1, l2, m2, l3, m3)
                            if cg == 0.0:
                                continue
                            total += (
                                cg
                                * coeffs[(l1, m1)]
                                * coeffs[(l2, m2)]
                                * np.conj(coeffs[(l3, m3)])
                            )
                    out[(l1, l2, l3, shell_idx)] = total
    return out


def build_invariant_descriptor(
    field: ScalarField,
    center: np.ndarray,
    shell_radii: Sequence[float],
    r0: float,
    sigma: float,
    l_max: int = 4,
    n_points_per_shell: int = 150,
) -> InvariantDescriptor:
    """Full Phase-2 pipeline for one critical point: sample shells -> fit SH
    coefficients per shell -> power spectrum + bispectrum. `n_points_per_shell`
    should be several times (l_max+1)^2 -- see `fit_sh_coefficients`."""
    shells = [
        sample_shell(field, center, radius, n_points_per_shell, r0, sigma)
        for radius in shell_radii
    ]
    coeffs_by_shell = [fit_sh_coefficients(s, l_max) for s in shells]
    ps = power_spectrum(coeffs_by_shell, l_max)
    bs = bispectrum(coeffs_by_shell, l_max)
    return InvariantDescriptor(
        l_max=l_max, shell_radii=tuple(shell_radii), power_spectrum=ps, bispectrum=bs
    )


# --------------------------------------------------------------------------
# Relational field topology (a competing descriptor view, not grammar)
# --------------------------------------------------------------------------

_RELATIONAL_TOPOLOGY_FAMILIES = (-3, -1, 1, 3, "other")


def relational_topology_components(shell_radii: Sequence[float]) -> Tuple[str, ...]:
    """Stable names for the field-landmark context-vector components."""
    names: List[str] = []
    for shell_index, _radius in enumerate(shell_radii):
        for signature in _RELATIONAL_TOPOLOGY_FAMILIES:
            names.append(f"shell_{shell_index}:signature_{signature}:log_count")
            names.append(f"shell_{shell_index}:signature_{signature}:mean_radius_fraction")
        names.append(f"shell_{shell_index}:all_neighbours:pair_angle_P1")
        names.append(f"shell_{shell_index}:all_neighbours:pair_angle_P2")
    return tuple(names)


def _topology_family(critical_point: object) -> int | str:
    signature = getattr(critical_point, "signature", None)
    rank = getattr(critical_point, "rank", None)
    return signature if rank == 3 and signature in {-3, -1, 1, 3} else "other"


def build_relational_topology_vectors(
    critical_points: Sequence[object],
    shell_radii: Sequence[float],
) -> Tuple[List[np.ndarray], Tuple[str, ...]]:
    """Describe a landmark by the *field topology around it*.

    This deliberately contains no atom coordinates, bonds, molecule graph or
    raw density value.  For every non-degenerate critical point it records,
    in radial shells, which field-topology signatures occur nearby, their
    relative distances and the invariant pair-angle moments of their spatial
    arrangement.  Thus a void/minimum with a distinctive arrangement of
    surrounding maxima/saddles can be tested as a candidate even if its own
    local density parameters are unremarkable.

    The result is a descriptor *view*, not a discovered grammar and not an
    assertion that a point is a letter.  It is intended to be compared with
    the local-field view under the same corpus controls.
    """
    radii = tuple(float(radius) for radius in shell_radii)
    if not radii or any(radius <= 0.0 for radius in radii) or tuple(sorted(radii)) != radii:
        raise ValueError("shell_radii must be a sorted, non-empty sequence of positive values")
    components = relational_topology_components(radii)
    valid_indices = [
        index for index, point in enumerate(critical_points)
        if getattr(point, "field_type", "degenerate") != "degenerate"
    ]
    vectors = [np.zeros(len(components), dtype=float) for _ in critical_points]
    if not valid_indices:
        return vectors, components

    positions = np.stack([
        np.asarray(getattr(critical_points[index], "position"), dtype=float)
        for index in valid_indices
    ])
    from scipy.spatial import cKDTree
    neighbour_tree = cKDTree(positions)
    families = [_topology_family(critical_points[index]) for index in valid_indices]

    for local_center, original_center in enumerate(valid_indices):
        nearby = np.asarray(
            neighbour_tree.query_ball_point(positions[local_center], radii[-1]), dtype=int
        )
        delta = positions[nearby] - positions[local_center]
        distances = np.linalg.norm(delta, axis=1)
        offset = 0
        inner_radius = 0.0
        for outer_radius in radii:
            mask = (distances > inner_radius) & (distances <= outer_radius)
            selected = np.flatnonzero(mask)
            for family in _RELATIONAL_TOPOLOGY_FAMILIES:
                family_selected = [
                    neighbour for neighbour in selected
                    if families[nearby[neighbour]] == family
                ]
                count = len(family_selected)
                vectors[original_center][offset] = np.log1p(count)
                vectors[original_center][offset + 1] = (
                    float(np.mean(distances[family_selected] / outer_radius)) if count else 0.0
                )
                offset += 2

            if len(selected) >= 2:
                directions = delta[selected] / distances[selected, None]
                cosine = directions @ directions.T
                upper = cosine[np.triu_indices(len(selected), k=1)]
                vectors[original_center][offset] = float(np.mean(upper))
                vectors[original_center][offset + 1] = float(
                    np.mean(0.5 * (3.0 * upper * upper - 1.0))
                )
            offset += 2
            inner_radius = outer_radius

    return vectors, components
