#!/usr/bin/env python3
"""Fixed-space and full-greedy tests for two nonsmooth parametrized PDEs.

The two models are

1. a 2-D membrane obstacle problem in Moreau--Yosida form, and
2. a 2-D elastic cantilever with frictionless unilateral contact.

Both use a piecewise-linear positive/negative-part law.  The reduced model is
Galerkin, the greedy indicator is a relative dual residual, and the continuous
search uses exact active-set sensitivities (one-sided derivatives at switches)
with a smooth tanh parametrization of the parameter box.  True errors are used
only for the final, independent deployment evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import factorized, spsolve
from scipy.stats import qmc


Array = NDArray[np.float64]


def sobol_points(n: int, d: int, seed: int) -> Array:
    """Deterministic scrambled Sobol points in [-1,1]^d."""
    m = int(math.ceil(math.log2(max(1, n))))
    pts = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)
    return 2.0 * pts[:n] - 1.0


def x_orthonormalize(U: Array, K: csc_matrix, rtol: float = 1e-11) -> Array:
    """K-orthonormalize columns of U using a stable Gram eigendecomposition."""
    if U.size == 0:
        return np.zeros((K.shape[0], 0))
    G = np.asarray(U.T @ (K @ U))
    G = 0.5 * (G + G.T)
    lam, Q = np.linalg.eigh(G)
    keep = lam > rtol * max(float(lam[-1]), 1.0)
    if not np.any(keep):
        return np.zeros((K.shape[0], 0))
    return U @ (Q[:, keep] / np.sqrt(lam[keep])[None, :])


def append_basis(V: Array, u: Array, K: csc_matrix, tol: float = 1e-10) -> tuple[Array, bool]:
    """Append one K-normalized snapshot, with two-pass orthogonalization."""
    v = u.copy()
    for _ in range(2):
        if V.shape[1]:
            v -= V @ (V.T @ (K @ v))
    nrm2 = float(v @ (K @ v))
    ref2 = max(float(u @ (K @ u)), 1e-30)
    if nrm2 <= tol * tol * ref2:
        return V, False
    v /= math.sqrt(nrm2)
    return np.column_stack((V, v)), True


@dataclass
class SolveInfo:
    iterations: int
    active_count: int
    converged: bool


class PiecewisePDE:
    """F(u,mu)=K u-f(mu)+gamma B' h(Bu+b(mu))=0."""

    name: str
    d: int = 3

    def __init__(self) -> None:
        self.K: csc_matrix
        self.B: csr_matrix
        self.gamma: float
        self.kind: str
        self.bounds: Array
        self._solveK = None
        self.last_active: Array | None = None

    def finalize(self) -> None:
        self.K = csc_matrix(self.K)
        self.B = csr_matrix(self.B)
        self._solveK = factorized(self.K)

    def physical(self, z: Array) -> tuple[Array, Array]:
        mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        half = 0.5 * (self.bounds[:, 1] - self.bounds[:, 0])
        return mid + half * z, half

    def data(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """Return f, b and derivatives wrt normalized z."""
        raise NotImplementedError

    def is_active(self, s: Array) -> Array:
        return s > 0.0 if self.kind == "max" else s < 0.0

    def h(self, s: Array) -> Array:
        return np.maximum(s, 0.0) if self.kind == "max" else np.minimum(s, 0.0)

    def residual(self, u: Array, z: Array) -> Array:
        f, b, _, _ = self.data(z)
        return np.asarray(self.K @ u - f + self.gamma * (self.B.T @ self.h(self.B @ u + b)))

    def energy(self, u: Array, z: Array) -> float:
        f, b, _, _ = self.data(z)
        hs = self.h(self.B @ u + b)
        return 0.5 * float(u @ (self.K @ u)) - float(f @ u) + 0.5 * self.gamma * float(hs @ hs)

    def solve_full(self, z: Array, u0: Array | None = None, tol: float = 1e-10) -> tuple[Array, SolveInfo]:
        f, b, _, _ = self.data(z)
        n = self.K.shape[0]
        if u0 is None:
            u = np.asarray(self._solveK(f))
        else:
            u = u0.copy()
        converged = False
        for it in range(1, 101):
            s = self.B @ u + b
            active = self.is_active(s)
            res = np.asarray(self.K @ u - f + self.gamma * (self.B.T @ self.h(s)))
            if np.linalg.norm(res) <= tol * max(np.linalg.norm(f), 1.0):
                converged = True
                break
            Ba = self.B[active]
            A = self.K + self.gamma * (Ba.T @ Ba)
            direction = np.asarray(spsolve(A, -res))
            phi0 = self.energy(u, z)
            slope = float(res @ direction)
            step = 1.0
            while step > 1.0 / 4096.0:
                trial = u + step * direction
                if self.energy(trial, z) <= phi0 + 1e-4 * step * slope:
                    u = trial
                    break
                step *= 0.5
            else:
                # The energy is strongly convex; a tiny damped step is safe and
                # avoids accepting a different active-region minimizer wholesale.
                u = u + (1.0 / 4096.0) * direction
        self.last_active = self.is_active(self.B @ u + b)
        return u, SolveInfo(it, int(np.sum(self.last_active)), converged)

    def solve_reduced(self, V: Array, z: Array, c0: Array | None = None) -> tuple[Array, Array, SolveInfo]:
        f, b, _, _ = self.data(z)
        rdim = V.shape[1]
        if rdim == 0:
            u = np.zeros(self.K.shape[0])
            return np.zeros(0), u, SolveInfo(0, 0, True)
        KV = self.K @ V
        BV = self.B @ V
        Kred = np.asarray(V.T @ KV)
        if c0 is None:
            c = np.linalg.solve(Kred, V.T @ f)
        else:
            c = c0.copy()
        converged = False
        rhs0 = V.T @ f
        scale = max(float(np.linalg.norm(rhs0)), 1.0)
        for it in range(1, 101):
            s = BV @ c + b
            active = self.is_active(s)
            hs = self.h(s)
            grad = Kred @ c - rhs0 + self.gamma * (BV.T @ hs)
            if np.linalg.norm(grad) <= 1e-11 * scale:
                converged = True
                break
            BVa = BV[active]
            A = Kred + self.gamma * (BVa.T @ BVa)
            direction = np.linalg.solve(A, -grad)
            phi0 = 0.5 * float(c @ (Kred @ c)) - float(rhs0 @ c) + 0.5 * self.gamma * float(hs @ hs)
            slope = float(grad @ direction)
            step = 1.0
            while step > 1.0 / 4096.0:
                cand = c + step * direction
                hcand = self.h(BV @ cand + b)
                phicand = 0.5 * float(cand @ (Kred @ cand)) - float(rhs0 @ cand) + 0.5 * self.gamma * float(hcand @ hcand)
                if phicand <= phi0 + 1e-4 * step * slope:
                    c = cand
                    break
                step *= 0.5
            else:
                c = c + (1.0 / 4096.0) * direction
        u = V @ c
        return c, u, SolveInfo(it, int(np.sum(active)), converged)

    def indicator_and_grad(self, V: Array, z: Array) -> tuple[float, Array, dict[str, Any]]:
        """log relative dual residual and its active-set derivative."""
        f, b, df, db = self.data(z)
        c, u, info = self.solve_reduced(V, z)
        s = np.asarray(self.B @ u + b)
        active = self.is_active(s)
        hs = self.h(s)
        res = np.asarray(self.K @ u - f + self.gamma * (self.B.T @ hs))
        w = np.asarray(self._solveK(res))
        num = max(float(res @ w), 1e-300)
        wf = np.asarray(self._solveK(f))
        den = max(float(f @ wf), 1e-300)
        J = 0.5 * (math.log(num) - math.log(den))

        BV = self.B @ V
        BaV = BV[active]
        Ared = np.asarray(V.T @ (self.K @ V)) + self.gamma * (BaV.T @ BaV)
        grad = np.zeros(self.d)
        min_switch_distance = float(np.min(np.abs(s))) if s.size else math.inf
        for j in range(self.d):
            direct = -df[:, j]
            if np.any(active):
                direct = direct + self.gamma * np.asarray(self.B[active].T @ db[active, j]).ravel()
            dc = np.linalg.solve(Ared, -(V.T @ direct))
            du = V @ dc
            dr = np.asarray(self.K @ du + direct)
            if np.any(active):
                dr += self.gamma * np.asarray(self.B[active].T @ (BaV @ dc)).ravel()
            dnum = 2.0 * float(dr @ w)
            dden = 2.0 * float(df[:, j] @ wf)
            grad[j] = 0.5 * (dnum / num - dden / den)
        return J, grad, {
            "active_count": int(np.sum(active)),
            "min_switch_distance": min_switch_distance,
            "rom_iterations": info.iterations,
        }

    def indicator(self, V: Array, z: Array) -> float:
        return self.indicator_and_grad(V, z)[0]

    def relative_energy_error(self, V: Array, z: Array) -> float:
        truth, _ = self.solve_full(z)
        _, rom, _ = self.solve_reduced(V, z)
        e = truth - rom
        return math.sqrt(max(float(e @ (self.K @ e)), 0.0) / max(float(truth @ (self.K @ truth)), 1e-300))


class ObstacleModel(PiecewisePDE):
    def __init__(self, nxy: int = 24) -> None:
        super().__init__()
        self.name = "membrane_obstacle"
        self.kind = "max"
        self.nxy = nxy
        h = 1.0 / (nxy + 1)
        T = diags([-np.ones(nxy - 1), 2.0 * np.ones(nxy), -np.ones(nxy - 1)], [-1, 0, 1]) / h**2
        self.K = kron(eye(nxy), T) + kron(T, eye(nxy))
        self.B = eye(nxy * nxy, format="csr")
        self.gamma = 2.0e5
        x = np.linspace(h, 1.0 - h, nxy)
        X, Y = np.meshgrid(x, x, indexing="xy")
        self.x = X.ravel()
        self.y = Y.ravel()
        # source amplitude, horizontal source center, obstacle height
        self.bounds = np.array([[12.0, 105.0], [0.18, 0.82], [0.018, 0.115]])
        self.sigma = 0.115
        self.curvature = 0.12
        self.finalize()

    def data(self, z: Array) -> tuple[Array, Array, Array, Array]:
        p, half = self.physical(z)
        amp, xc, height = p
        shape = np.exp(-0.5 * (((self.x - xc) / self.sigma) ** 2 + ((self.y - 0.52) / self.sigma) ** 2))
        f = amp * shape
        obstacle = height + self.curvature * ((self.x - 0.5) ** 2 + (self.y - 0.5) ** 2)
        # s=u-obstacle; h=max(s,0)
        b = -obstacle
        df_phys = np.column_stack((shape, amp * shape * (self.x - xc) / self.sigma**2, np.zeros_like(f)))
        db_phys = np.column_stack((np.zeros_like(f), np.zeros_like(f), -np.ones_like(f)))
        return f, b, df_phys * half[None, :], db_phys * half[None, :]


def q1_elasticity(nx: int, ny: int, length: float, height: float, E: float, nu: float) -> tuple[csc_matrix, Array, Array, Array]:
    """Plane-strain Q1 stiffness; returns reduced K, coordinates, dof map, free dofs."""
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    coords = np.array([(x, y) for y in ys for x in xs], dtype=float)
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = fac * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]])
    hx, hy = length / nx, height / ny
    gp = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    ke = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dN_dxi = 0.25 * np.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])
            dNdx = dN_dxi * 2.0 / hx
            dNdy = dN_deta * 2.0 / hy
            Bm = np.zeros((3, 8))
            Bm[0, 0::2] = dNdx
            Bm[1, 1::2] = dNdy
            Bm[2, 0::2] = dNdy
            Bm[2, 1::2] = dNdx
            ke += Bm.T @ D @ Bm * hx * hy / 4.0
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for j in range(ny):
        for i in range(nx):
            nodes = [j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
            dofs = np.array([[2 * n, 2 * n + 1] for n in nodes]).ravel()
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows.extend(rr.ravel().tolist())
            cols.extend(cc.ravel().tolist())
            vals.extend(ke.ravel().tolist())
    ndof = 2 * len(coords)
    Kfull = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    fixed_nodes = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    fixed = np.array([[2 * n, 2 * n + 1] for n in fixed_nodes]).ravel()
    free = np.setdiff1d(np.arange(ndof), fixed)
    dof_map = -np.ones(ndof, dtype=int)
    dof_map[free] = np.arange(free.size)
    return Kfull[free][:, free].tocsc(), coords, dof_map, free


class ContactModel(PiecewisePDE):
    def __init__(self, nx: int = 28, ny: int = 8) -> None:
        super().__init__()
        self.name = "elastic_contact"
        self.kind = "min"
        self.length = 2.0
        self.height = 0.4
        K, coords, dof_map, free = q1_elasticity(nx, ny, self.length, self.height, E=1.0, nu=0.30)
        self.K = K
        self.coords = coords
        self.dof_map = dof_map
        self.free = free
        bottom_nodes = np.flatnonzero(np.isclose(coords[:, 1], 0.0) & (coords[:, 0] > 0.0))
        contact_dofs = dof_map[2 * bottom_nodes + 1]
        rows = np.arange(contact_dofs.size)
        self.B = coo_matrix((np.ones(contact_dofs.size), (rows, contact_dofs)), shape=(contact_dofs.size, K.shape[0])).tocsr()
        self.contact_x = coords[bottom_nodes, 0]
        self.gamma = 2.0e4
        # total downward force, force center, initial floor gap
        self.bounds = np.array([[0.010, 0.115], [0.45, 1.82], [0.012, 0.110]])
        self.load_width = 0.18
        self.finalize()

    def data(self, z: Array) -> tuple[Array, Array, Array, Array]:
        p, half = self.physical(z)
        force, xc, gap0 = p
        xy = self.coords
        # Smooth body load, concentrated near the upper surface; normalize so
        # parameter 1 is approximately the total force.
        raw = np.exp(-0.5 * ((xy[:, 0] - xc) / self.load_width) ** 2) * np.exp(-0.5 * ((xy[:, 1] - 0.32) / 0.09) ** 2)
        raw[xy[:, 0] == 0.0] = 0.0
        S = float(np.sum(raw))
        weights = raw / S
        draw = raw * (xy[:, 0] - xc) / self.load_width**2
        dS = float(np.sum(draw))
        dweights = (draw * S - raw * dS) / S**2
        ffull = np.zeros(2 * len(xy))
        ffull[1::2] = -force * weights
        df_force = np.zeros_like(ffull)
        df_force[1::2] = -weights
        df_xc = np.zeros_like(ffull)
        df_xc[1::2] = -force * dweights
        f = ffull[self.free]
        df_phys = np.column_stack((df_force[self.free], df_xc[self.free], np.zeros_like(f)))
        # Foundation has a gentle fixed waviness; s=u_y+gap and contact is s<0.
        waviness = 0.12 * gap0 * np.cos(2.0 * np.pi * self.contact_x / self.length)
        gap = gap0 + waviness
        b = gap
        db_gap = 1.0 + 0.12 * np.cos(2.0 * np.pi * self.contact_x / self.length)
        db_phys = np.column_stack((np.zeros_like(b), np.zeros_like(b), db_gap))
        return f, b, df_phys * half[None, :], db_phys * half[None, :]


def adam_search(
    model: PiecewisePDE,
    V: Array,
    z0: Array,
    lr: float,
    steps: int = 100,
) -> dict[str, Any]:
    """Single-start Adam ascent in unconstrained tanh coordinates."""
    zsafe = np.clip(z0, -1.0 + 1e-8, 1.0 - 1e-8)
    x = np.arctanh(zsafe)
    m = np.zeros(model.d)
    v = np.zeros(model.d)
    best_J = -math.inf
    best_z = zsafe.copy()
    values: list[float] = []
    best_values: list[float] = []
    active_counts: list[int] = []
    switches = 0
    previous_active = None
    for t in range(1, steps + 1):
        z = np.tanh(x)
        J, gz, meta = model.indicator_and_grad(V, z)
        gx = gz * (1.0 - z * z)
        m = 0.9 * m + 0.1 * gx
        v = 0.999 * v + 0.001 * gx * gx
        mh = m / (1.0 - 0.9**t)
        vh = v / (1.0 - 0.999**t)
        x += lr * mh / (np.sqrt(vh) + 1e-8)
        if J > best_J:
            best_J = J
            best_z = z.copy()
        values.append(float(J))
        best_values.append(float(best_J))
        ac = int(meta["active_count"])
        active_counts.append(ac)
        if previous_active is not None and ac != previous_active:
            switches += 1
        previous_active = ac
    # Evaluate the post-update endpoint too.
    zend = np.tanh(x)
    Jend = model.indicator(V, zend)
    if Jend > best_J:
        best_J, best_z = Jend, zend.copy()
    return {
        "best_J": float(best_J),
        "best_z": best_z,
        "values": values,
        "best_values": best_values,
        "active_counts": active_counts,
        "active_count_changes": switches,
    }


def build_common_basis(model: PiecewisePDE, n_snap: int, rank: int, seed: int) -> tuple[Array, list[dict[str, Any]]]:
    pts = sobol_points(n_snap, model.d, seed)
    snaps = []
    info = []
    for z in pts:
        u, si = model.solve_full(z)
        snaps.append(u)
        info.append({"z": z.tolist(), "active_count": si.active_count, "iterations": si.iterations, "converged": si.converged})
    U = np.column_stack(snaps)
    # Method of snapshots: keep dominant K-POD vectors.
    G = np.asarray(U.T @ (model.K @ U))
    G = 0.5 * (G + G.T)
    lam, Q = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1]
    lam, Q = lam[order], Q[:, order]
    keep = lam[:rank] > 1e-12 * max(lam[0], 1.0)
    V = U @ (Q[:, :rank][:, keep] / np.sqrt(lam[:rank][keep])[None, :])
    return V, info


def evaluate_indicators(model: PiecewisePDE, V: Array, pts: Array) -> Array:
    return np.array([model.indicator(V, z) for z in pts])


def wilson_interval(k: int, n: int, zcrit: float = 1.96) -> list[float]:
    phat = k / n
    den = 1.0 + zcrit * zcrit / n
    center = (phat + zcrit * zcrit / (2.0 * n)) / den
    half = zcrit * math.sqrt(phat * (1.0 - phat) / n + zcrit * zcrit / (4.0 * n * n)) / den
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def finite_difference_gradient_audit(model: PiecewisePDE, V: Array, pts: Array, eps: float = 2e-6) -> dict[str, Any]:
    rels = []
    abses = []
    switch_dist = []
    for z in pts:
        J, g, meta = model.indicator_and_grad(V, z)
        gfd = np.zeros(model.d)
        for j in range(model.d):
            zp, zm = z.copy(), z.copy()
            zp[j] += eps
            zm[j] -= eps
            gfd[j] = (model.indicator(V, zp) - model.indicator(V, zm)) / (2.0 * eps)
        ae = float(np.linalg.norm(g - gfd))
        re = ae / max(float(np.linalg.norm(gfd)), 1e-12)
        rels.append(re)
        abses.append(ae)
        switch_dist.append(meta["min_switch_distance"])
    return {
        "median_relative_error": float(np.median(rels)),
        "max_relative_error": float(np.max(rels)),
        "median_absolute_error": float(np.median(abses)),
        "min_switch_distance": float(np.min(switch_dist)),
        "n_points": len(pts),
    }


def fixed_space_experiment(
    model: PiecewisePDE,
    quick: bool,
    seed: int,
) -> tuple[dict[str, Any], Array]:
    n_snap, rank = (10, 5) if quick else (16, 7)
    n_ref = 4096 if quick else 32768
    n_tune = 16 if quick else 40
    n_trials = 96 if quick else 320
    V, basis_info = build_common_basis(model, n_snap, rank, seed + 1)

    audit_pts = 0.75 * sobol_points(8, model.d, seed + 2)
    grad_audit = finite_difference_gradient_audit(model, V, audit_pts)

    ref_pts = sobol_points(n_ref, model.d, seed + 3)
    ref_J = evaluate_indicators(model, V, ref_pts)
    quantiles = [0.99, 0.995, 0.999]
    thresholds = {str(q): float(np.quantile(ref_J, q)) for q in quantiles}
    empirical_p = {str(q): float(np.mean(ref_J >= thresholds[str(q)])) for q in quantiles}

    # Learning rate is selected on a disjoint tuning set by mean best objective.
    tune_starts = sobol_points(n_tune, model.d, seed + 4)
    lrs = [0.01, 0.03, 0.1, 0.3]
    tune_scores: dict[str, float] = {}
    for lr in lrs:
        vals = [adam_search(model, V, z, lr, steps=100)["best_J"] for z in tune_starts]
        tune_scores[str(lr)] = float(np.mean(vals))
    lr = max(lrs, key=lambda x: tune_scores[str(x)])

    starts = sobol_points(n_trials, model.d, seed + 5)
    traces = [adam_search(model, V, z, lr, steps=100) for z in starts]
    bestJ = np.array([t["best_J"] for t in traces])
    qs = {str(q): float(np.mean(bestJ >= thresholds[str(q)])) for q in quantiles}
    pool_success = {str(q): float(1.0 - (1.0 - empirical_p[str(q)]) ** 100) for q in quantiles}
    checkpoints = [1, 5, 10, 25, 50, 100]
    q_by_step: dict[str, dict[str, float]] = {}
    for T in checkpoints:
        vals = np.array([t["best_values"][T - 1] for t in traces])
        q_by_step[str(T)] = {str(q): float(np.mean(vals >= thresholds[str(q)])) for q in quantiles}

    # Empirical independent pools; max distributions are useful beyond one tau.
    rng = np.random.default_rng(seed + 6)
    pool_max = np.empty(n_trials)
    for i in range(n_trials):
        pool_pts = rng.uniform(-1.0, 1.0, size=(100, model.d))
        pool_max[i] = float(np.max(evaluate_indicators(model, V, pool_pts)))

    q_ci = {}
    pool_empirical = {}
    pool_empirical_ci = {}
    for q in quantiles:
        key = str(q)
        ka = int(np.sum(bestJ >= thresholds[key]))
        kp = int(np.sum(pool_max >= thresholds[key]))
        q_ci[key] = wilson_interval(ka, n_trials)
        pool_empirical[key] = kp / n_trials
        pool_empirical_ci[key] = wilson_interval(kp, n_trials)

    active_start = []
    active_best = []
    for z0, tr in zip(starts, traces):
        active_start.append(model.indicator_and_grad(V, z0)[2]["active_count"])
        active_best.append(model.indicator_and_grad(V, tr["best_z"])[2]["active_count"])

    result = {
        "model": model.name,
        "dimension": model.d,
        "truth_dofs": int(model.K.shape[0]),
        "contact_or_obstacle_nodes": int(model.B.shape[0]),
        "gamma": model.gamma,
        "basis_rank": int(V.shape[1]),
        "basis_snapshots": basis_info,
        "gradient_audit": grad_audit,
        "reference_size": n_ref,
        "reference_J": {
            "min": float(np.min(ref_J)), "median": float(np.median(ref_J)),
            "max": float(np.max(ref_J)), "q99": float(np.quantile(ref_J, 0.99)),
        },
        "thresholds": thresholds,
        "p": empirical_p,
        "pool100_success_formula": pool_success,
        "lr_tuning_mean_best_J": tune_scores,
        "chosen_lr": lr,
        "trials": n_trials,
        "q_T100": qs,
        "q_T100_wilson95": q_ci,
        "pool100_success_empirical": pool_empirical,
        "pool100_success_empirical_wilson95": pool_empirical_ci,
        "q_by_step": q_by_step,
        "adam_best_J": {
            "median": float(np.median(bestJ)), "mean": float(np.mean(bestJ)),
            "q10": float(np.quantile(bestJ, 0.1)), "q90": float(np.quantile(bestJ, 0.9)),
            "max": float(np.max(bestJ)),
        },
        "pool100_best_J": {
            "median": float(np.median(pool_max)), "mean": float(np.mean(pool_max)),
            "q10": float(np.quantile(pool_max, 0.1)), "q90": float(np.quantile(pool_max, 0.9)),
            "max": float(np.max(pool_max)),
        },
        "active_set": {
            "start_median": float(np.median(active_start)),
            "best_median": float(np.median(active_best)),
            "trajectory_count_changes_mean": float(np.mean([t["active_count_changes"] for t in traces])),
        },
        "search_samples": {
            "starts": starts.tolist(),
            "adam_best_J": bestJ.tolist(),
            "adam_best_z": [t["best_z"].tolist() for t in traces],
            "pool100_best_J": pool_max.tolist(),
        },
    }
    return result, V


def run_greedy(
    model: PiecewisePDE,
    method: str,
    n_basis: int,
    seed: int,
    lr: float,
    starts_per_step: int = 1,
) -> tuple[Array, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    z0 = np.zeros(model.d)
    u0, si = model.solve_full(z0)
    V, _ = append_basis(np.zeros((model.K.shape[0], 0)), u0, model.K)
    hist = [{"n": 1, "z": z0.tolist(), "active_count": si.active_count}]
    while V.shape[1] < n_basis:
        if method == "pool100":
            pts = rng.uniform(-1.0, 1.0, size=(100, model.d))
            vals = evaluate_indicators(model, V, pts)
            k = int(np.argmax(vals))
            zsel, Jsel = pts[k], float(vals[k])
        elif method == "adam100":
            candidates = []
            for _ in range(starts_per_step):
                zs = rng.uniform(-1.0, 1.0, size=model.d)
                candidates.append(adam_search(model, V, zs, lr, 100))
            best = max(candidates, key=lambda a: a["best_J"])
            zsel, Jsel = best["best_z"], best["best_J"]
        else:
            raise ValueError(method)
        u, si = model.solve_full(zsel)
        Vnew, added = append_basis(V, u, model.K)
        hist.append({
            "n": int(V.shape[1] + int(added)), "z": zsel.tolist(), "indicator_log": Jsel,
            "active_count": si.active_count, "truth_iterations": si.iterations, "added": bool(added),
        })
        if not added:
            # Deterministic small perturbation if a redundant point is selected.
            zsel = np.clip(zsel + 0.05 * rng.normal(size=model.d), -0.999, 0.999)
            u, _ = model.solve_full(zsel)
            Vnew, added = append_basis(V, u, model.K)
            if not added:
                break
        V = Vnew
    return V, hist


def precompute_truths(model: PiecewisePDE, pts: Array) -> tuple[list[Array], Array]:
    truths = []
    norms = []
    for z in pts:
        u, _ = model.solve_full(z)
        truths.append(u)
        norms.append(max(float(u @ (model.K @ u)), 1e-300))
    return truths, np.asarray(norms)


def deployment_errors(model: PiecewisePDE, V: Array, pts: Array, truths: list[Array], truth_norms: Array) -> Array:
    errs = np.empty(len(pts))
    for i, (z, truth) in enumerate(zip(pts, truths)):
        _, rom, _ = model.solve_reduced(V, z)
        e = truth - rom
        errs[i] = math.sqrt(max(float(e @ (model.K @ e)), 0.0) / truth_norms[i])
    return errs


def full_greedy_experiment(model: PiecewisePDE, fixed: dict[str, Any], quick: bool, seed: int) -> dict[str, Any]:
    n_basis = 8 if quick else 12
    n_seeds = 3 if quick else 8
    n_test = 512 if quick else 2048
    lr = float(fixed["chosen_lr"])
    test = sobol_points(n_test, model.d, seed + 200)
    truths, truth_norms = precompute_truths(model, test)
    out: dict[str, Any] = {"n_basis": n_basis, "n_seeds": n_seeds, "test_size": n_test, "methods": {}}
    for method in ["pool100", "adam100"]:
        runs = []
        for k in range(n_seeds):
            V, hist = run_greedy(model, method, n_basis, seed + 300 + 19 * k, lr)
            errs = deployment_errors(model, V, test, truths, truth_norms)
            runs.append({
                "seed": seed + 300 + 19 * k,
                "rank": int(V.shape[1]),
                "max_error": float(np.max(errs)),
                "q99_error": float(np.quantile(errs, 0.99)),
                "mean_error": float(np.mean(errs)),
                "history": hist,
            })
        out["methods"][method] = {
            "runs": runs,
            "max_error_median": float(np.median([r["max_error"] for r in runs])),
            "max_error_min": float(np.min([r["max_error"] for r in runs])),
            "max_error_max": float(np.max([r["max_error"] for r in runs])),
            "q99_error_median": float(np.median([r["q99_error"] for r in runs])),
            "mean_error_median": float(np.median([r["mean_error"] for r in runs])),
        }
    return out


def should_run_greedy(fixed: dict[str, Any]) -> bool:
    # Require a meaningful fixed-space advantage at a rare threshold.
    for q in ["0.995", "0.999"]:
        if fixed["q_T100"][q] >= fixed["pool100_success_formula"][q] + 0.08:
            return True
    return False


def make_plots(results: dict[str, Any], outdir: Path) -> None:
    models = list(results["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), squeeze=False)
    for ax, name in zip(axes[0], models):
        f = results["models"][name]["fixed_space"]
        steps = np.array([1, 5, 10, 25, 50, 100])
        for q, color in [("0.99", "C0"), ("0.995", "C1"), ("0.999", "C3")]:
            ys = [f["q_by_step"][str(t)][q] for t in steps]
            ax.plot(steps, ys, marker="o", color=color, label=f"Adam q, tau={q}")
            ax.axhline(f["pool100_success_formula"][q], color=color, linestyle="--", alpha=0.65)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Adam steps T")
        ax.set_ylabel("success probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fixed_space_success.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(5.7 * len(models), 4.4), squeeze=False)
    for ax, name in zip(axes[0], models):
        f = results["models"][name]["fixed_space"]
        ax.boxplot([f["search_samples"]["pool100_best_J"], f["search_samples"]["adam_best_J"]], tick_labels=["pool M=100", "Adam T=100"])
        ax.set_title(name.replace("_", " "))
        ax.set_ylabel("best log relative residual")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "best_indicator_distribution.png", dpi=190)
    plt.close(fig)


def write_report(results: dict[str, Any], outdir: Path) -> None:
    lines = [
        "# Nonsmooth continuous-search benchmark",
        "",
        "Two genuinely active-set-switching PDE discretizations were tested. Selection always uses the relative dual residual; truth errors appear only in independent deployment tests. Adam uses one random start, 100 steps, and a smooth tanh box map.",
        "",
        "## Fixed-space p-q test",
        "",
        "Dashed levels in the figure are the exact iid pool-M=100 success probability computed from empirical p. q is estimated on independent starts. Learning rates were chosen on a disjoint tuning set.",
        "",
        "| model | RB rank | chosen lr | grad audit median rel. | tau quantile | p | pool100 success | Adam q(T=100) | median best J: pool / Adam |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in results["models"].items():
        f = block["fixed_space"]
        for q in ["0.99", "0.995", "0.999"]:
            lines.append(
                f"| {name} | {f['basis_rank']} | {f['chosen_lr']:.3g} | {f['gradient_audit']['median_relative_error']:.2e} | {q} | {f['p'][q]:.4f} | {f['pool100_success_formula'][q]:.3f} | {f['q_T100'][q]:.3f} | {f['pool100_best_J']['median']:.3f} / {f['adam_best_J']['median']:.3f} |"
            )
    lines += ["", "![Fixed-space success](fixed_space_success.png)", "", "![Best objective distributions](best_indicator_distribution.png)", "", "## Full greedy", ""]
    for name, block in results["models"].items():
        g = block.get("full_greedy")
        if g is None:
            lines.append(f"- **{name}:** not run: the prespecified fixed-space separation criterion was not met.")
            continue
        p = g["methods"]["pool100"]
        a = g["methods"]["adam100"]
        ratio = p["max_error_median"] / a["max_error_median"]
        lines.append(
            f"- **{name}:** at rank {g['n_basis']}, median max deployment error is {p['max_error_median']:.3e} (pool) versus {a['max_error_median']:.3e} (Adam); pool/Adam ratio = {ratio:.2f}."
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- These experiments test search geometry, not indicator reliability. Both methods use exactly the same residual indicator.",
        "- A positive result requires q to exceed the iid-pool success at the same threshold, not merely a higher single run.",
        "- The Sobol reference maximum is not claimed to be the exact global maximum. Thresholds are empirical reference quantiles.",
        "- Active-set derivatives are classical derivatives inside each region and one-sided selections at switching surfaces; the finite-difference audit excludes exact measure-zero kinks only implicitly.",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", choices=["both", "obstacle", "contact"], default="both")
    parser.add_argument("--outdir", default="nonsmooth_pde_bench/results")
    parser.add_argument("--force-greedy", action="store_true")
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected = []
    if args.model in ("both", "obstacle"):
        selected.append(ObstacleModel(18 if args.quick else 24))
    if args.model in ("both", "contact"):
        selected.append(ContactModel(20 if args.quick else 28, 6 if args.quick else 8))
    results: dict[str, Any] = {
        "protocol": {
            "seed": args.seed, "quick": args.quick, "optimizer": "single-start Adam",
            "steps": 100, "pool_size": 100, "box_map": "z=tanh(x)",
            "selection": "relative dual residual only", "deployment": "relative K-energy error",
        },
        "models": {},
    }
    t0 = time.time()
    for i, model in enumerate(selected):
        print(f"[{model.name}] fixed-space experiment", flush=True)
        fixed, _ = fixed_space_experiment(model, args.quick, args.seed + 1000 * i)
        block: dict[str, Any] = {"fixed_space": fixed}
        run_full = args.force_greedy or (should_run_greedy(fixed) and not args.skip_greedy)
        if run_full and not args.skip_greedy:
            print(f"[{model.name}] fixed-space separation found; running full greedy", flush=True)
            block["full_greedy"] = full_greedy_experiment(model, fixed, args.quick, args.seed + 1000 * i)
        else:
            print(f"[{model.name}] full greedy skipped by prespecified gate", flush=True)
        results["models"][model.name] = block
        (outdir / "results.partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["wall_seconds"] = time.time() - t0
    (outdir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_plots(results, outdir)
    write_report(results, outdir)
    print(json.dumps({
        name: {
            "lr": block["fixed_space"]["chosen_lr"],
            "pool999": block["fixed_space"]["pool100_success_formula"]["0.999"],
            "q999": block["fixed_space"]["q_T100"]["0.999"],
            "greedy": "full_greedy" in block,
        }
        for name, block in results["models"].items()
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
