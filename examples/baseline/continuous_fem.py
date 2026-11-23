"""Mesh-independent manufactured elliptic family with a conforming FEM truth model.

The continuum source functions are fixed before choosing a mesh.  The default discrete
solutions are their energy projections, not grid-calibrated sine loads.
The analytic_snapshots option samples the exact solution at the same nodes
instead of solving the HF diffusion system; all assembly is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementTriP3,
    LinearForm,
    MeshTri,
    asm,
)


def sine_pairs(d: int) -> tuple[tuple[int, int], ...]:
    """First ``d`` positive modes, in an ordering independent of mesh and d."""
    if d < 1:
        raise ValueError("d must be positive")
    pairs = ((kx, ky) for kx in range(1, d + 1) for ky in range(1, d + 1))
    return tuple(sorted(pairs, key=lambda p: (p[0] ** 2 + p[1] ** 2, p[0], p[1]))[:d])


def sine_gradient(kx: int, ky: int, x: np.ndarray, y: np.ndarray):
    """Gradient of 2 sin(pi kx x) sin(pi ky y)."""
    gx = 2.0 * math.pi * kx * np.cos(math.pi * kx * x) * np.sin(math.pi * ky * y)
    gy = 2.0 * math.pi * ky * np.sin(math.pi * kx * x) * np.cos(math.pi * ky * y)
    return gx, gy


def mode_source(kx: int, ky: int, contrast: float, x: np.ndarray, y: np.ndarray):
    """Analytic -div(a grad phi) for a mesh-independent sine mode."""
    phi = 2.0 * np.sin(math.pi * kx * x) * np.sin(math.pi * ky * y)
    gx, gy = sine_gradient(kx, ky, x, y)
    sx, sy = np.sin(2.0 * math.pi * x), np.sin(2.0 * math.pi * y)
    a = 1.0 + contrast * sx * sy
    ax = 2.0 * math.pi * contrast * np.cos(2.0 * math.pi * x) * sy
    ay = 2.0 * math.pi * contrast * sx * np.cos(2.0 * math.pi * y)
    return a * math.pi**2 * (kx**2 + ky**2) * phi - ax * gx - ay * gy


def continuum_energy_gram(
    pairs: tuple[tuple[int, int], ...],
    contrast: float,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Energy Gram matrix of the unscaled sine functions on (0, 1)^2."""
    max_wave = max(max(pair) for pair in pairs)
    q = quadrature_order or (2 * max_wave + 16)
    nodes, weights = leggauss(q)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    weight = np.outer(weights, weights) * (
        1.0 + contrast * np.sin(2.0 * math.pi * x) * np.sin(2.0 * math.pi * y)
    )
    gradients = [sine_gradient(kx, ky, x, y) for kx, ky in pairs]
    gx = np.stack([g[0] for g in gradients])
    gy = np.stack([g[1] for g in gradients])
    gram = np.einsum("ipq,jpq,pq->ij", gx, gx, weight, optimize=True)
    gram += np.einsum("ipq,jpq,pq->ij", gy, gy, weight, optimize=True)
    return 0.5 * (gram + gram.T)


@dataclass
class FEMProblem:
    """Problem interface used by the existing greedy and ROM implementations."""

    n_grid: int
    d: int
    contrast: float
    spectrum_ratio: float
    A: sp.csc_matrix
    X: sp.csc_matrix
    B: np.ndarray
    S: np.ndarray
    solution_gram: np.ndarray
    singular_values: np.ndarray
    solve_X: object
    continuum_gram: np.ndarray
    source_transform: np.ndarray
    modes: tuple[tuple[int, int], ...]
    projection_error: float
    dofs: int
    quadrature_order: int
    element_order: int

    def source_value(self, j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate f_j=-rho^j div(a grad psi_j), with zero-based j."""
        if not 0 <= j < self.d:
            raise IndexError(j)
        result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
        for i, (kx, ky) in enumerate(self.modes[: j + 1]):
            result += self.source_transform[i, j] * mode_source(
                kx, ky, self.contrast, x, y
            )
        return self.spectrum_ratio**j * result

    @classmethod
    def build(
        cls,
        n_grid: int,
        d: int,
        contrast: float = 0.25,
        spectrum_ratio: float = 0.78,
        quadrature_order: int | None = None,
        element_order: int = 3,
        analytic_snapshots: bool = False,
    ) -> FEMProblem:
        if n_grid < 1 or not (0 <= contrast < 1) or not (0 < spectrum_ratio < 1):
            raise ValueError("Require n_grid >= 1, 0 <= contrast < 1, and 0 < rho < 1")
        if element_order not in (1, 2, 3):
            raise ValueError(
                "Only conforming triangular P1, P2, and P3 elements are supported"
            )
        pairs = sine_pairs(d)
        max_wave = max(max(pair) for pair in pairs)
        intorder = quadrature_order or min(19, max(12, 2 * max_wave + 4))
        g = continuum_energy_gram(pairs, contrast)
        upper = la.cholesky(g, lower=False)
        transform = la.solve_triangular(upper, np.eye(d), lower=False)
        np.testing.assert_allclose(transform.T @ g @ transform, np.eye(d), atol=2e-12)

        coords = np.linspace(0.0, 1.0, n_grid + 2)
        element = {1: ElementTriP1, 2: ElementTriP2, 3: ElementTriP3}[element_order]()
        basis = Basis(MeshTri.init_tensor(coords, coords), element, intorder=intorder)
        interior = basis.complement_dofs(basis.get_dofs())

        @BilinearForm
        def energy(u, v, w):
            x, y = w.x
            coeff = 1.0 + contrast * np.sin(2.0 * math.pi * x) * np.sin(
                2.0 * math.pi * y
            )
            return coeff * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

        @BilinearForm
        def laplace(u, v, w):
            return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]

        A = asm(energy, basis).tocsr()[interior, :][:, interior].tocsc()
        X = asm(laplace, basis).tocsr()[interior, :][:, interior].tocsc()
        b0 = np.empty((len(interior), d))

        def make_weak_source(kx: int, ky: int):
            @LinearForm
            def weak_source(v, w):
                x, y = w.x
                gx, gy = sine_gradient(kx, ky, x, y)
                coeff = 1.0 + contrast * np.sin(2.0 * math.pi * x) * np.sin(
                    2.0 * math.pi * y
                )
                return coeff * (gx * v.grad[0] + gy * v.grad[1])

            return weak_source

        for j, (kx, ky) in enumerate(pairs):
            b0[:, j] = asm(make_weak_source(kx, ky), basis)[interior]

        solve_X = spla.factorized(X)
        psi_loads = b0 @ transform
        if analytic_snapshots:
            # Only replace the HF diffusion solve: sample the exact functions
            # at the same finite-element degrees of freedom.
            nodes = basis.doflocs[:, interior]
            phi = np.column_stack([
                2.0*np.sin(math.pi*k*nodes[0])*np.sin(math.pi*l*nodes[1])
                for k,l in pairs])
            psi_snapshots = phi @ transform
        else:
            solve_A = spla.factorized(A)
            psi_snapshots = np.column_stack([solve_A(psi_loads[:, j]) for j in range(d)])
        normalized_gram = psi_snapshots.T @ (A @ psi_snapshots)
        normalized_gram = 0.5 * (normalized_gram + normalized_gram.T)
        spectrum = spectrum_ratio ** np.arange(d, dtype=float)
        B = psi_loads * spectrum[None, :]
        S = psi_snapshots * spectrum[None, :]
        solution_gram = normalized_gram * np.outer(spectrum, spectrum)
        singular_values = np.sqrt(np.maximum(la.eigvalsh(solution_gram)[::-1], 0.0))
        cross = psi_loads.T @ psi_snapshots
        projection_defect = np.eye(d) - cross - cross.T + normalized_gram
        projection_error = math.sqrt(max(float(la.eigvalsh(projection_defect)[-1]), 0.0))
        return cls(
            n_grid,
            d,
            contrast,
            spectrum_ratio,
            A,
            X,
            B,
            S,
            solution_gram,
            singular_values,
            solve_X,
            g,
            transform,
            pairs,
            projection_error,
            len(interior),
            intorder,
            element_order,
        )
