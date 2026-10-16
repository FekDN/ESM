"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
analytic_field.py
==================
A `ScalarField` whose `value`/`gradient`/`hessian` are exact contractions
over the density matrix and AO derivatives -- not finite differences on a
cube's trilinear interpolant.

Why this exists: `CubeFileField.hessian` (fields.py) reconstructs curvature
from a piecewise-LINEAR interpolant of gridded values, via a finite
difference sized to the grid spacing itself -- documented there as
recovering "the true underlying curvature ... to within the interpolant's
own smoothing error", which is an honestly-bounded approximation, not an
exact one. Classification (rank/signature, `critical_points.classify`) acts
directly on this Hessian, so that smoothing error is coupled to the cube's
`resolution` parameter -- a representational choice (doc §20), not a
property of the field. This class removes the coupling entirely for any
molecule whose PySCF `mol` and converged density matrix are still
available: rho, its gradient and its Hessian are evaluated in closed form
from the basis set, at the exact point asked for, independent of any grid.

Formulas (D the converged density matrix, phi the AO basis, real orbitals
so D is symmetric):
    rho       = sum_mn D_mn phi_m phi_n                      = ao^T D ao
    grad rho  = 2 sum_mn D_mn (d phi_m) phi_n                = 2 (d ao)^T D ao
    hess rho  = 2 sum_mn D_mn [ (d2 phi_m) phi_n + (da phi_m)(db phi_n) ]
    (each pairwise term appears twice in the mu<->nu sum; D's symmetry is
    exactly what lets both be folded into the factor of 2 above, rather
    than needing the mu!=nu cross terms written out separately.)

Grid-dependent quantities this class does NOT provide on its own --
`.grid`/`.origin`/`.vectors`/`.dims` are deliberately absent, not stubbed.
`critical_points.py`'s
Stage 0b/1 (grid-maxima seeding, regular-grid coverage) already need a
sampled grid to do their job at all, and read those three attributes via
`getattr(field, ..., None)` with a graceful skip when absent -- exactly the
extension point this class is meant to use: pair an `AnalyticDensityField`
for value/gradient/hessian with a `CubeFileField` of the *same* density for
seeding, via `AnalyticDensityField.paired_with_grid`, rather than silently
answering "grid" with something that isn't one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .fields import ScalarField, CubeFileField


class AnalyticDensityField(ScalarField):
    field_kind = "density"

    def __init__(
        self,
        mol,  # pyscf.gto.Mole, already built
        density_matrix: np.ndarray,
        *,
        level_of_theory: str = "unspecified",
        environment: str = "vacuum",
        grid_source: Optional[CubeFileField] = None,
    ) -> None:
        self._mol = mol
        self._dm = np.asarray(density_matrix, dtype=float)
        if self._dm.ndim != 2 or self._dm.shape[0] != self._dm.shape[1]:
            raise ValueError("density_matrix must be a square (n_ao, n_ao) array")
        self.level_of_theory = level_of_theory
        self.environment = environment
        self._grid_source = grid_source
        if grid_source is not None:
            # Duck-typed by critical_points.py's Stage 0b/1 -- present only
            # when a caller explicitly opted in via `paired_with_grid`,
            # never fabricated from the analytic field alone (there is no
            # natural grid to offer without one).
            self.grid = grid_source.grid
            self.origin = grid_source.origin
            self.vectors = grid_source.vectors
            # The analytic derivatives have no natural discretization. Once
            # paired with a sampled field, expose its full grid geometry so
            # callers can derive bounds consistently from the same samples.
            self.dims = grid_source.dims
            self.atom_positions = grid_source.atom_positions
            self.atomic_numbers = grid_source.atomic_numbers

    def paired_with_grid(self, grid_source: CubeFileField) -> "AnalyticDensityField":
        """Return an equivalent field that also exposes `.grid`/`.origin`/
        `.vectors`/`.dims` (read from `grid_source`, e.g. a `CubeFileField`
        of the SAME density) so Stage 0b/1 seeding still has a grid to scan, while
        every `value`/`gradient`/`hessian` call stays exact and analytic --
        it never reads `grid_source`'s own (interpolated) derivatives."""
        return AnalyticDensityField(
            self._mol, self._dm, level_of_theory=self.level_of_theory,
            environment=self.environment, grid_source=grid_source,
        )

    def _ao_deriv2(self, r: np.ndarray) -> np.ndarray:
        """(10, n_ao): value, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz --
        PySCF's actual `GTOval_sph_deriv2` ordering (upper-triangular,
        row-major; verified empirically against finite differences on raw
        AO values -- NOT the [xx,yy,zz,xy,xz,yz] grouping a first guess at
        this API would reach for, and the earlier draft of this file used
        exactly that wrong guess, undetected by the gradient check alone
        since the gradient never touches indices 4-9)."""
        coords = np.asarray(r, dtype=float).reshape(1, 3)
        return self._mol.eval_gto("GTOval_sph_deriv2", coords)[:, 0, :]

    def value(self, r: np.ndarray) -> float:
        ao = self._ao_deriv2(r)[0]
        return float(ao @ self._dm @ ao)

    def gradient(self, r: np.ndarray, h: float = 1e-4) -> np.ndarray:
        deriv = self._ao_deriv2(r)
        ao, d_ao = deriv[0], deriv[1:4]
        return np.array([2.0 * float(d_ao[a] @ self._dm @ ao) for a in range(3)])

    def hessian(self, r: np.ndarray, h: float = 2e-3) -> np.ndarray:
        deriv = self._ao_deriv2(r)
        ao, d_ao, d2_ao = deriv[0], deriv[1:4], deriv[4:10]
        # d2_ao rows (see _ao_deriv2): xx, xy, xz, yy, yz, zz
        pair_to_row = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
        H = np.zeros((3, 3))
        for a in range(3):
            for b in range(a, 3):
                row = pair_to_row[(a, b)]
                term = 2.0 * float(d2_ao[row] @ self._dm @ ao) + 2.0 * float(d_ao[a] @ self._dm @ d_ao[b])
                H[a, b] = H[b, a] = term
        return H
