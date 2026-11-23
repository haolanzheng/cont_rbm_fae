"""Poisson with a smooth, unit-L2-norm load on a parameter cube.

On (0, 1), -u''=sqrt(2)*(sin(pi*x)+sum_j z_j*sin((j+1)*pi*x))
/sqrt(1+|z|^2), with homogeneous Dirichlet conditions.  No prescribed
geometric spectrum is used.  Energy-orthonormal solution coordinates have
weights 1/(j*pi), j=1,...,d+1.  The center snapshot is included initially.
"""
import numpy as np


class NormalizedLoad:
    def __init__(self, d):
        self.d = int(d)
        self.weights = 1 / (np.pi * np.arange(1, self.d + 2))

    def solution(self, z):
        z = np.asarray(z, dtype=float)
        w = np.concatenate((np.ones(z.shape[:-1] + (1,)), z), axis=-1)
        return w * self.weights / np.sqrt(np.sum(w*w, axis=-1, keepdims=True))

    def initial_basis(self):
        q = np.zeros((self.d+1, 1))
        q[0, 0] = 1
        return q

    def matrix(self, q):
        a = np.diag(self.weights) - q @ (q.T * self.weights)
        return a.T @ a

    def objective(self, h, z, gradient=False):
        z = np.asarray(z, dtype=float)
        w = np.concatenate((np.ones(z.shape[:-1] + (1,)), z), axis=-1)
        hw = w @ h
        denominator = np.sum(w*w, axis=-1)
        value = np.sum(w*hw, axis=-1) / denominator
        if not gradient:
            return value
        return value, 2 * (hw[..., 1:] - value[..., None]*z) / denominator[..., None]

    def search(self, q, z, alpha=.6, steps=80):
        h = self.matrix(q)
        ell = max(np.linalg.norm(h, 'fro'), 1e-30)
        z = np.array(z, dtype=float, copy=True)
        best = z.copy()
        best_value = self.objective(h, z)
        for _ in range(steps):
            _, g = self.objective(h, z, True)
            g /= ell
            z = np.clip(z + alpha*g/np.sqrt(1+np.sum(g*g, axis=-1, keepdims=True)), -1, 1)
            value = self.objective(h, z)
            improve = value > best_value
            best = np.where(np.asarray(improve)[..., None], z, best)
            best_value = np.maximum(value, best_value)
        return best, best_value

    def append(self, q, z):
        w = self.solution(z)
        for _ in range(2):
            w -= q @ (q.T @ w)
        size = np.linalg.norm(w)
        if size < 1e-13:
            raise ValueError('Numerically dependent snapshot')
        return np.column_stack((q, w / size))

    def error_bounds(self, q):
        """Analytic spectral bounds, evaluated in floating-point arithmetic.

        Every space contains the center snapshot. Thus J=z^T H z/(1+|z|^2)
        for H=matrix(q)[1:,1:]. The upper bound is sqrt(lambda_max(H)).
        A unit top eigenvector v gives a feasible point z=v/|v|_infinity,
        and a lower bound sqrt(lambda_max(H)/(1+|v|_infinity^2)).
        """
        h = self.matrix(q)[1:, 1:]
        values, vectors = np.linalg.eigh(h)
        value = max(float(values[-1]), 0.)
        v = vectors[:, -1]
        z = v / np.max(np.abs(v))
        return np.sqrt(value / (1+np.max(np.abs(v))**2)), np.sqrt(value), z
