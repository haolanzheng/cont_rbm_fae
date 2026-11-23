"""Exact one-dimensional layered diffusion with paired coefficients.

On (0, 1), solve -(a u')'=0, u(0)=0, a(1)u'(1)=1.  Each of d
equal cells has coefficients a0+delta*(z_i-shift) and
a0-delta*(z_i-shift) on its two halves.  Parameters lie in [-1,1]^d.

Vectors store the 2d piecewise slopes divided by sqrt(2d); Euclidean
inner products are therefore exactly the reference H1 energy products.
The full solution, Galerkin solution, residual dual norm, and their
parameter derivatives are computed without a spatial discretization.
"""
from __future__ import annotations

import math

import numpy as np


class LayeredDiffusion:
    """Exact coefficient-PDE model; the source normalization is ||f||_X'^2=1."""

    def __init__(self, d: int, delta: float = 0.4, shift: float = 0.75,
                 a0: float = 1.0, chunk_size: int = 256):
        self.d = int(d)
        self.delta = float(delta)
        self.shift = float(shift)
        self.a0 = float(a0)
        self.chunk_size = int(chunk_size)
        if self.d < 1 or self.d != d:
            raise ValueError("d must be a positive integer")
        if not -1 <= self.shift <= 1:
            raise ValueError("The center snapshot must lie in the parameter cube")
        if not 0 < self.delta * (1 + abs(self.shift)) < self.a0:
            raise ValueError("Require 0 < delta*(1+abs(shift)) < a0 for coercivity")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.state_dimension = 2 * self.d
        self.source = np.full(self.state_dimension, 1 / math.sqrt(self.state_dimension))
        self.source_norm_squared = 1.0
        self.normalizer = 1.0
        self.coefficient_min = self.a0 - self.delta * (1 + abs(self.shift))
        self.coefficient_max = self.a0 + self.delta * (1 + abs(self.shift))

    def _parameters(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != self.d:
            raise ValueError(f"Parameters must have final dimension {self.d}")
        if not np.all(np.isfinite(z)):
            raise ValueError("Parameters must be finite")
        return z

    def center_parameter(self):
        """The actual constant-coefficient snapshot used to initialize the RB."""
        return np.full(self.d, self.shift)

    def coefficients(self, z):
        z = self._parameters(z)
        difference = self.delta * (z - self.shift)
        a = np.empty(z.shape[:-1] + (self.state_dimension,), dtype=np.float64)
        a[..., 0::2] = self.a0 + difference
        a[..., 1::2] = self.a0 - difference
        if np.any(a <= 0):
            raise ValueError("Diffusion coefficients must be positive")
        return a

    def solution(self, z):
        """Full solution in reference-energy coordinates, shape (..., 2d)."""
        return self.source / self.coefficients(z)

    def empty_basis(self):
        return np.empty((self.state_dimension, 0), dtype=np.float64)

    def append_basis(self, Q, z, tolerance: float = 1e-12):
        """Add one exact snapshot with two passes of reference orthogonalization.

        Return an unchanged copy when the snapshot is numerically dependent;
        callers can detect that case from the number of columns.
        """
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != self.state_dimension:
            raise ValueError("The basis must have shape (2d, n)")
        z = self._parameters(z)
        if z.ndim != 1:
            raise ValueError("Basis enrichment takes exactly one parameter")
        snapshot = self.solution(z)
        scale = np.linalg.norm(snapshot)
        residual = snapshot.copy()
        for _ in range(2):
            residual -= Q @ (Q.T @ residual)
        size = np.linalg.norm(residual)
        if size <= tolerance * scale:
            return Q.copy()
        return np.column_stack((Q, residual / size))

    def initial_basis(self):
        return self.append_basis(self.empty_basis(), self.center_parameter())

    def prepare(self, Q):
        """Cache reduced coefficient matrices for one fixed basis.

        Prefer prepare(Q).objective(...) inside a multi-step search so that
        the affine reduced operators are assembled once per greedy round.
        """
        return LayeredReducedModel(self, Q)

    def objective(self, z, Q, gradient: bool = False):
        return self.prepare(Q).objective(z, gradient=gradient)

    def errors(self, z, Q, norm: str = "reference", relative: bool = False):
        return self.prepare(Q).errors(z, norm=norm, relative=relative)

    def metadata(self):
        return {
            "model": "paired layered diffusion", "d": self.d,
            "state_dimension": self.state_dimension, "a0": self.a0,
            "delta": self.delta, "shift": self.shift,
            "coefficient_min": self.coefficient_min,
            "coefficient_max": self.coefficient_max,
            "objective": "squared reference-H1 dual residual",
            "normalizer": self.normalizer,
            "reference_norm": "L2 norm of piecewise slopes",
            "initial_snapshot": "z_i=shift for every i",
        }


class LayeredReducedModel:
    """Cached Galerkin matrices and analytic derivatives for one fixed Q."""

    def __init__(self, model: LayeredDiffusion, Q):
        self.model = model
        self.Q = np.array(Q, dtype=np.float64, copy=True)
        if self.Q.ndim != 2 or self.Q.shape[0] != model.state_dimension:
            raise ValueError("The basis must have shape (2d, n)")
        if not np.all(np.isfinite(self.Q)):
            raise ValueError("The basis must be finite")
        self.rank = self.Q.shape[1]
        if self.rank > model.state_dimension:
            raise ValueError("The basis has too many columns")
        if self.rank:
            defect = np.linalg.norm(self.Q.T @ self.Q - np.eye(self.rank), ord=2)
            if defect > 1e-8:
                raise ValueError(f"Basis is not reference-orthonormal: defect={defect}")
        self.rhs = self.Q.T @ model.source
        base_a = model.coefficients(np.zeros(model.d))
        self.reduced_base = self.Q.T @ (base_a[:, None] * self.Q)
        positive, negative = self.Q[0::2], self.Q[1::2]
        self.reduced_derivatives = model.delta * (
            positive[:, :, None] * positive[:, None, :]
            - negative[:, :, None] * negative[:, None, :]
        )
        self.chunk_size = min(model.chunk_size,
                              max(1, 2_000_000 // max(1, self.rank**2)))

    def _flat(self, z):
        z = self.model._parameters(z)
        return z.reshape((-1, self.model.d)), z.shape[:-1]

    def _state(self, z):
        """One chunk: coefficient diagonal, reduced matrix, and RB solution."""
        a = self.model.coefficients(z)
        if self.rank == 0:
            return a, None, np.zeros_like(a)
        reduced = self.reduced_base[None, :, :] + np.einsum(
            "bi,ijk->bjk", z, self.reduced_derivatives, optimize=True
        )
        rhs = np.broadcast_to(self.rhs, (len(z), self.rank))
        coefficients = np.linalg.solve(reduced, rhs[..., None])[..., 0]
        return a, reduced, coefficients @ self.Q.T

    def objective(self, z, gradient: bool = False):
        """Squared residual (and its exact parameter gradient when requested).

        For r=f-AQc and Ar=Q^T A Q, dc=-Ar^{-1} Q^T(dA)Qc.
        Setting w=Ar^{-1}Q^T A r gives d||r||^2=-2(r-Qw)^T(dA)Qc.
        Since dA/dz_i is supported on one pair of layers, every gradient
        component is evaluated directly without differentiated solvers.
        """
        flat, shape = self._flat(z)
        values = np.empty(len(flat), dtype=np.float64)
        derivatives = np.empty_like(flat) if gradient else None
        for start in range(0, len(flat), self.chunk_size):
            stop = min(start + self.chunk_size, len(flat))
            a, reduced, state = self._state(flat[start:stop])
            residual = self.model.source - a * state
            values[start:stop] = np.einsum("bi,bi->b", residual, residual)
            if gradient:
                if self.rank:
                    adjoint_rhs = (a * residual) @ self.Q
                    adjoint = np.linalg.solve(reduced, adjoint_rhs[..., None])[..., 0]
                    effective = residual - adjoint @ self.Q.T
                    derivatives[start:stop] = -2 * self.model.delta * (
                        effective[:, 0::2] * state[:, 0::2]
                        - effective[:, 1::2] * state[:, 1::2]
                    )
                else:
                    derivatives[start:stop] = 0
        values = values.reshape(shape)
        if not shape:
            values = float(values)
        if gradient:
            return values, derivatives.reshape(shape + (self.model.d,))
        return values

    def reduced_solution(self, z):
        flat, shape = self._flat(z)
        result = np.empty((len(flat), self.model.state_dimension), dtype=np.float64)
        for start in range(0, len(flat), self.chunk_size):
            stop = min(start + self.chunk_size, len(flat))
            result[start:stop] = self._state(flat[start:stop])[2]
        return result.reshape(shape + (self.model.state_dimension,))

    def errors(self, z, norm: str = "reference", relative: bool = False):
        """Actual Galerkin error in the reference or parameter-dependent norm.

        norm='reference': ||u-u_n||_X, X defined by unit conductivity.
        norm='energy': sqrt(integral a(x;z)*(u'-u_n')^2).
        The methods return errors, not squared errors.
        """
        if norm not in ("reference", "energy"):
            raise ValueError("norm must be 'reference' or 'energy'")
        flat, shape = self._flat(z)
        result = np.empty(len(flat), dtype=np.float64)
        for start in range(0, len(flat), self.chunk_size):
            stop = min(start + self.chunk_size, len(flat))
            a, _, state = self._state(flat[start:stop])
            truth = self.model.source / a
            error = truth - state
            weights = a if norm == "energy" else 1.0
            squared = np.sum(weights * error * error, axis=1)
            if relative:
                squared /= np.sum(weights * truth * truth, axis=1)
            result[start:stop] = np.sqrt(np.maximum(squared, 0.0))
        result = result.reshape(shape)
        return float(result) if not shape else result

