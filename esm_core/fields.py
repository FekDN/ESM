"""
Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
fields.py
=========
Scalar-field backends for the letter-decoding pipeline (theory doc §2, §15).

The theory treats atoms as non-fundamental (§2) and works instead with a
continuous scalar field: electron density rho(r), electrostatic potential
ESP(r), or ELF(r). Every downstream stage (critical-point finder, invariant
descriptors) only needs a `ScalarField`: something that returns a value at a
point in R^3, plus a gradient and Hessian at that point.

Backends:
    ScalarField          -- abstract interface. Subclass and implement `value`;
                             gradient/Hessian have a numeric fallback so any
                             backend works immediately with the rest of the
                             pipeline (§15, precaution: numeric Hessians are
                             the least trustworthy part near-degenerate points
                             -- prefer overriding with analytic derivatives
                             when the backend can supply them).
    CubeFileField         -- reads a real Gaussian-cube file (the standard
                             output of Multiwfn / Gaussian / ORCA / VASP-cube
                             for rho(r) or ESP(r) on a grid). This is the
                             intended production entry point.
    GaussianMixtureDensity-- analytic sum-of-primitive-Gaussians toy density,
                             used only for self-tests and demos so the
                             pipeline is runnable with zero external QM
                             dependency. NOT a physically fitted density --
                             do not use it as a stand-in for real rho(r).
    ScrambledFieldControl -- §16.2 null-model backend: turns any reference
                             field into a structure-free field "of the same
                             scale" (same bbox, same sampled amplitude
                             distribution) by permuting or i.i.d.
                             resampling a regular grid of the reference
                             field's own values. Built entirely from data the
                             reference field already provides -- no new
                             measurements -- so it is the field-level piece
                             `unsupervised_validation.py` needs and nothing
                             more; see that module for how it's used.

`level_of_theory` and `field_kind` are plain metadata carried alongside every
field so that later stages can honor the §15 precaution "разделение уровня
теории для геометрии и для топологии поля": a critical point is only trusted
once it is shown to be stable across >=2 independent `level_of_theory` tags
(see critical_points.check_basis_convergence). `environment` (§4.3's
indexical parameter -- "vacuum", "water(eps=80)", "enzyme_pocket", etc.) is
the third piece of per-field metadata; `pipeline.extract_letter_candidates`
reads it to tag every `LetterCandidate.environment_condition`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as _dc_field
from typing import Sequence

import numpy as np


class ScalarField(ABC):
    """Abstract scalar field over R^3, with metadata about how it was produced."""

    level_of_theory: str = "unspecified"
    field_kind: str = "unspecified"  # "density" | "esp" | "elf" | ...
    environment: str = "unspecified"  # indexical context, §4.3 (e.g. "vacuum",
    # "water(eps=80)", "enzyme_pocket") -- read by pipeline.extract_letter_candidates
    # to tag every LetterCandidate's environment_condition; was previously
    # unset on every concrete backend below, which raised AttributeError as
    # soon as a molecule actually produced any candidates (empty results
    # never touched the attribute, so the gap was silent until then).

    @abstractmethod
    def value(self, r: np.ndarray) -> float:
        """Field value at Cartesian point r (shape (3,))."""
        raise NotImplementedError

    def gradient(self, r: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """Numeric central-difference gradient. Override with an analytic
        expression when the backend supports it -- this fallback is O(6)
        field evaluations per call and is the accuracy bottleneck for
        cheap/noisy fields."""
        r = np.asarray(r, dtype=float)
        g = np.zeros(3)
        for i in range(3):
            rp, rm = r.copy(), r.copy()
            rp[i] += h
            rm[i] -= h
            g[i] = (self.value(rp) - self.value(rm)) / (2.0 * h)
        return g

    def hessian(self, r: np.ndarray, h: float = 2e-3) -> np.ndarray:
        """Numeric central-difference Hessian (symmetrized). This is the
        noisiest quantity in the pipeline -- critical-point classification
        (critical_points.classify) is directly sensitive to it. Prefer an
        analytic override wherever available."""
        r = np.asarray(r, dtype=float)
        H = np.zeros((3, 3))
        for i in range(3):
            ei = np.zeros(3)
            ei[i] = h
            H[i, i] = (self.value(r + ei) - 2.0 * self.value(r) + self.value(r - ei)) / h**2
        for i in range(3):
            for j in range(i + 1, 3):
                ei = np.zeros(3)
                ej = np.zeros(3)
                ei[i] = h
                ej[j] = h
                f_pp = self.value(r + ei + ej)
                f_pm = self.value(r + ei - ej)
                f_mp = self.value(r - ei + ej)
                f_mm = self.value(r - ei - ej)
                val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h**2)
                H[i, j] = H[j, i] = val
        return H


@dataclass
class _GaussianPrimitive:
    center: np.ndarray
    amplitude: float
    exponent: float


class GaussianMixtureDensity(ScalarField):
    """Toy analytic scalar field: rho(r) = sum_k A_k * exp(-alpha_k * |r-c_k|^2).

    This is emphatically NOT a physically fitted electron density -- it exists
    only so the rest of the pipeline (critical points, invariants, clustering)
    has something concrete and *fast* to run against in demos and unit tests,
    with exact analytic gradient/Hessian to validate the numeric fallbacks in
    `ScalarField` against. Replace with `CubeFileField` for real molecules.
    """

    field_kind = "density"

    def __init__(
        self,
        centers: Sequence[Sequence[float]],
        amplitudes: Sequence[float],
        exponents: Sequence[float],
        level_of_theory: str = "toy/gaussian-mixture",
        environment: str = "vacuum",
    ) -> None:
        assert len(centers) == len(amplitudes) == len(exponents)
        self._prims = [
            _GaussianPrimitive(np.asarray(c, dtype=float), float(a), float(al))
            for c, a, al in zip(centers, amplitudes, exponents)
        ]
        self.level_of_theory = level_of_theory
        self.environment = environment

    def value(self, r: np.ndarray) -> float:
        r = np.asarray(r, dtype=float)
        total = 0.0
        for p in self._prims:
            d = r - p.center
            total += p.amplitude * np.exp(-p.exponent * float(d @ d))
        return total

    def gradient(self, r: np.ndarray, h: float = 1e-4) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        g = np.zeros(3)
        for p in self._prims:
            d = r - p.center
            val = p.amplitude * np.exp(-p.exponent * float(d @ d))
            g += val * (-2.0 * p.exponent) * d
        return g

    def hessian(self, r: np.ndarray, h: float = 2e-3) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        H = np.zeros((3, 3))
        for p in self._prims:
            d = r - p.center
            val = p.amplitude * np.exp(-p.exponent * float(d @ d))
            a = p.exponent
            H += val * (-2.0 * a * np.eye(3) + 4.0 * a**2 * np.outer(d, d))
        return H


class CubeFileField(ScalarField):
    """Reads a Gaussian-cube (.cube) volumetric scalar field.

    Cube format (Å in most QM codes write bohr internally -- we keep whatever
    unit the file declares and do not silently convert):

        line 1-2   : comments
        line 3     : N_atoms  origin_x origin_y origin_z  [N_val]
        line 4-6   : N_i  v_i_x v_i_y v_i_z          (i = 1,2,3; step vectors,
                                                       possibly non-orthogonal)
        N_atoms x  : atomic_number  charge  x y z
        remainder  : N1*N2*N3 values, fastest-varying axis last (axis 3)

    `value(r)` uses trilinear interpolation in the (possibly non-orthogonal)
    fractional grid coordinates obtained by solving r = origin + f @ vectors
    for f -- this is exact trilinear interpolation regardless of whether the
    grid axes are orthogonal, since the map from fractional to Cartesian
    coordinates is affine.
    """

    def __init__(self, path: str, field_kind: str = "density", level_of_theory: str = "from_cube",
                 environment: str = "unspecified"):
        # Stream the numeric payload straight into its final NumPy array.
        # The old implementation held `readlines()`, a Python list of every
        # voxel and the final array concurrently -- several copies of a cube
        # per DFT worker, which is untenable for a large field corpus.
        with open(path, "r", encoding="utf-8") as fh:
            fh.readline()  # cube comments are provenance only
            fh.readline()
            header3 = fh.readline().split()
            if len(header3) < 4:
                raise ValueError(f"invalid cube header in {path}")
            n_atoms = int(header3[0])
            origin = np.array([float(x) for x in header3[1:4]])

            dims = []
            vectors = []
            for _ in range(3):
                parts = fh.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"invalid cube grid header in {path}")
                dims.append(int(parts[0]))
                vectors.append([float(x) for x in parts[1:4]])
            self.dims = tuple(dims)  # (N1, N2, N3)
            self.vectors = np.array(vectors)  # rows = step vectors for axes 1,2,3
            self.origin = origin

            # Nuclear positions from the cube atom block. They are retained
            # for field construction provenance and optional post-hoc
            # comparison, not for alphabet candidate discovery (field-first).
            atom_numbers: list = []
            atom_positions: list = []
            for _ in range(abs(n_atoms)):
                parts = fh.readline().split()
                if len(parts) < 5:
                    raise ValueError(f"invalid cube atom block in {path}")
                atom_numbers.append(int(float(parts[0])))
                atom_positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
            self.atomic_numbers = np.asarray(atom_numbers, dtype=int)
            self.atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 3)

            n1, n2, n3 = self.dims
            expected = n1 * n2 * n3
            token_stream = (float(token) for line in fh for token in line.split())
            values = np.fromiter(
                token_stream,
                dtype=np.float64,
                count=expected,
            )
            # `count` protects peak RAM, but an extra trailing value must
            # still be rejected instead of silently changing the field.
            trailing = next(token_stream, None)
        if len(values) != expected or trailing is not None:
            found = len(values) + (1 if trailing is not None else 0)
            raise ValueError(
                f"cube file grid size mismatch: header declares {expected} "
                f"values ({n1}x{n2}x{n3}), found at least {found}"
            )
        self.grid = values.reshape(n1, n2, n3)

        self._inv_vectors = np.linalg.inv(self.vectors)
        self.field_kind = field_kind
        self.level_of_theory = level_of_theory
        self.environment = environment
        # Shortest step-vector norm = the grid's characteristic spacing,
        # used by the gradient/hessian overrides below (see their
        # docstrings) instead of ScalarField's toy-field-tuned defaults.
        self._char_length = float(np.min(np.linalg.norm(self.vectors, axis=1)))

    def _to_fractional(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=float) - self.origin) @ self._inv_vectors

    def value(self, r: np.ndarray) -> float:
        f = self._to_fractional(r)
        n1, n2, n3 = self.dims
        # clamp into valid interpolation range
        f = np.clip(f, 0.0, [n1 - 1.000001, n2 - 1.000001, n3 - 1.000001])
        i0 = np.floor(f).astype(int)
        t = f - i0
        i1 = i0 + 1
        acc = 0.0
        for da in (0, 1):
            for db in (0, 1):
                for dc in (0, 1):
                    w = (
                        (t[0] if da else 1 - t[0])
                        * (t[1] if db else 1 - t[1])
                        * (t[2] if dc else 1 - t[2])
                    )
                    idx = (
                        i0[0] + da if da else i0[0],
                        i0[1] + db if db else i0[1],
                        i0[2] + dc if dc else i0[2],
                    )
                    acc += w * self.grid[idx]
        return float(acc)

    def gradient(self, r: np.ndarray, h: float | None = None) -> np.ndarray:
        """Numeric gradient, but with a step size tied to the grid's own
        spacing rather than `ScalarField`'s tiny toy-field default.

        CORRECTNESS NOTE (fixed): trilinear interpolation is exactly LINEAR
        along each axis within one grid cell -- so with the inherited
        default (`h=1e-4`, tuned for a smooth analytic toy field), a
        gradient probe almost never crosses a cell boundary and mostly
        measures the interpolant's local slope faithfully (gradient is fine
        at first order), but `hessian` below is where this actually breaks;
        `gradient` is overridden here too only so both methods share one
        grid-spacing-derived default rather than mixing two different
        length scales silently.
        """
        h = h if h is not None else 0.5 * self._char_length
        return super().gradient(r, h=h)

    def hessian(self, r: np.ndarray, h: float | None = None) -> np.ndarray:
        """Numeric Hessian, with a step size tied to the grid's own spacing.

        CORRECTNESS NOTE (fixed): this was the real cause of "why does
        critical-point search on a real .cube density find almost no
        nuclear attractors" -- trilinear interpolation is exactly linear
        along each axis *within one cell*, so its analytic second
        derivative along any single axis is identically zero there; the
        inherited default step (`h=2e-3`, tuned for a smooth analytic toy
        field in `GaussianMixtureDensity`) stays inside one cell for any
        realistic cube grid spacing, so `ScalarField.hessian`'s central
        difference measured that structural zero, not the density's real
        curvature. Confirmed directly: on a synthetic trilinear grid with
        0.2-unit spacing, `h=2e-3` returned a diagonal Hessian of exactly
        [0,0,0], while `h` comparable to the grid spacing recovered the
        true underlying curvature to within the interpolant's own smoothing
        error. `_char_length` (the shortest step-vector norm from the cube
        header) is computed once at load time; the default `h` here spans
        roughly one grid cell, which is the minimum needed for a central
        difference to see the interpolant change slope at all.
        """
        h = h if h is not None else self._char_length
        return super().hessian(r, h=h)


class ScrambledFieldControl(ScalarField):
    """§16.2 null-model field ("Валидация без супервизии"): same spatial
    domain and amplitude scale as a `reference` field, spatial structure
    destroyed. Used by `unsupervised_validation.py` to ask whether the
    invariant-descriptor pipeline finds real structure at all, by re-running
    it on data that is, by construction, structure-free -- see that
    module's docstring for how this class is used; this class only builds
    the field itself.

    Built entirely out of `reference`'s own sampled values -- no new
    measurements, only extra randomness -- so it needs no new field data
    beyond what the real corpus already used (§16.2 is explicit that this
    check requires none). Construction: sample `reference` once on a
    regular, axis-aligned grid of `grid_resolution` points per axis over
    `bbox` (the same box the real critical-point search covered), then
    either:

        mode="scrambled" -- apply ONE random permutation to the grid's
            values. The exact empirical value multiset is unchanged (same
            histogram, same min/max -- "того же масштаба", the same scale in
            the most literal sense) while every spatial correlation the
            reference field had is destroyed. This is the standard
            shuffle-test null for spatial/temporal structure.
        mode="random" -- i.i.d. resample (with replacement) from that same
            empirical distribution at every grid point, instead of a single
            bijective permutation. Matches the distribution but not the
            exact multiset -- a genuinely unstructured random field, rather
            than a shuffled copy of a structured one. The doc lists
            "скремблированных ИЛИ случайных" as two distinct valid
            constructions, not two strengths of one idea, so both are
            implemented and selectable rather than the class picking one.

    `value(r)` is trilinear interpolation on this regular grid -- simpler
    than `CubeFileField`'s general oblique-axis case, since `bbox` is always
    an axis-aligned rectangular box with evenly-spaced steps by
    construction. Like any interpolated grid, the *value* is continuous but
    the *gradient* has kinks at cell boundaries (the interpolant is only
    piecewise multilinear) -- the same caveat `ScalarField`'s numeric
    Hessian fallback already carries generally, just sharper here since the
    grid itself is noise. `gradient`/`hessian` are not overridden: the base
    class's numeric fallback is used, which is the right level of accuracy
    for a null model that has no analytic structure to exploit.
    """

    def __init__(
        self,
        reference: ScalarField,
        bbox,
        grid_resolution: int = 10,
        mode: str = "scrambled",
        rng_seed=0,
    ) -> None:
        if mode not in ("scrambled", "random"):
            raise ValueError(f"mode must be 'scrambled' or 'random', got {mode!r}")
        if grid_resolution < 2:
            raise ValueError("grid_resolution must be >= 2 to define a grid spacing")

        lo, hi = np.asarray(bbox[0], dtype=float), np.asarray(bbox[1], dtype=float)
        axes = [np.linspace(lo[i], hi[i], grid_resolution) for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid_points = np.stack(mesh, axis=-1).reshape(-1, 3)
        raw_values = np.array([reference.value(p) for p in grid_points])

        rng = np.random.default_rng(rng_seed)
        if mode == "scrambled":
            shuffled = rng.permutation(raw_values)
        else:  # "random"
            shuffled = rng.choice(raw_values, size=raw_values.size, replace=True)

        self.grid = shuffled.reshape((grid_resolution, grid_resolution, grid_resolution))
        self.origin = lo
        self.step = (hi - lo) / (grid_resolution - 1)
        self.grid_resolution = grid_resolution
        self.mode = mode
        self.field_kind = reference.field_kind
        self.level_of_theory = f"{mode}_control({reference.level_of_theory})"
        self.environment = getattr(reference, "environment", "unspecified")
        # Same fix as CubeFileField, same reason: this is also trilinear
        # interpolation on a regular grid, so ScalarField's toy-tuned
        # numeric-derivative step sizes would sit entirely inside one cell
        # and see a structurally zero Hessian, not real curvature.
        self._char_length = float(np.min(self.step))

    def gradient(self, r: np.ndarray, h: float | None = None) -> np.ndarray:
        h = h if h is not None else 0.5 * self._char_length
        return super().gradient(r, h=h)

    def hessian(self, r: np.ndarray, h: float | None = None) -> np.ndarray:
        h = h if h is not None else self._char_length
        return super().hessian(r, h=h)

    def value(self, r: np.ndarray) -> float:
        f = (np.asarray(r, dtype=float) - self.origin) / self.step
        n = self.grid_resolution
        f = np.clip(f, 0.0, n - 1.000001)
        i0 = np.floor(f).astype(int)
        t = f - i0
        acc = 0.0
        for da in (0, 1):
            for db in (0, 1):
                for dc in (0, 1):
                    w = (
                        (t[0] if da else 1 - t[0])
                        * (t[1] if db else 1 - t[1])
                        * (t[2] if dc else 1 - t[2])
                    )
                    idx = (
                        i0[0] + da if da else i0[0],
                        i0[1] + db if db else i0[1],
                        i0[2] + dc if dc else i0[2],
                    )
                    acc += w * self.grid[idx]
        return float(acc)
