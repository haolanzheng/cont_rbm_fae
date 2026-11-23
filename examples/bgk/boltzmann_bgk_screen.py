#!/usr/bin/env python3
"""Continuous-greedy screening test inspired by Tong et al. (2026).

This is deliberately a screening model, not a reproduction of the paper's
full Boltzmann collision solver.  It keeps the paper's spatial/velocity grid,
wall temperatures, parameter boxes, training-pool size, and residual-based
mass-constrained ROM, while replacing the full collision integral by a
nonlinear BGK relaxation with the thermal average of the same VHS kernel.

The script compares, on fixed reduced spaces, a 50-point uniform pool with a
multistart continuous maximization of the minimized ROM residual.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space
from scipy.optimize import least_squares, minimize
from scipy.special import digamma, gamma
from scipy.stats import qmc


Array = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    nx: int = 100
    nv: int = 32
    x_left: float = 0.0
    x_right: float = 1.0
    t_left: float = 1.0
    t_right: float = 2.0
    lv: float = 4.5 * (3.0 + math.sqrt(2.0)) / 2.0
    fom_tol: float = 1.0e-6
    fom_maxiter: int = 6000
    fom_relaxation: float = 0.72
    inner_ftol: float = 1.0e-11
    inner_maxiter: int = 120
    seed: int = 260902578


class BGKModel:
    """1D-x/2D-v steady nonlinear kinetic screening model."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dx = (cfg.x_right - cfg.x_left) / cfg.nx
        self.dv = 2.0 * cfg.lv / cfg.nv
        self.x = cfg.x_left + (np.arange(cfg.nx) + 0.5) * self.dx
        self.v = -cfg.lv + (np.arange(cfg.nv) + 0.5) * self.dv
        self.v1, self.v2 = np.meshgrid(self.v, self.v, indexing="ij")
        self.vsq = self.v1**2 + self.v2**2
        self.weight = self.dx * self.dv**2
        self.vel_weight = self.dv**2
        self.pos = self.v1 > 0.0
        self.neg = ~self.pos
        self.wall_left_shape = np.exp(-self.vsq / (2.0 * cfg.t_left))
        self.wall_right_shape = np.exp(-self.vsq / (2.0 * cfg.t_right))

        t0 = cfg.t_left + (cfg.t_right - cfg.t_left) * self.x
        f0 = np.empty((cfg.nx, cfg.nv, cfg.nv))
        for i, temp in enumerate(t0):
            f0[i] = np.exp(-self.vsq / (2.0 * temp)) / (2.0 * np.pi * temp)
        self.initial = f0
        self.target_mass = float(np.sum(f0) * self.weight)

    def _wall_boundaries(self, f: Array) -> tuple[Array, Array]:
        # Standard diffuse reflection: one scalar density per wall enforces
        # zero total mass flux over the 2D velocity grid.
        out_left = np.sum((-self.v1[self.neg]) * f[0][self.neg]) * self.vel_weight
        den_left = np.sum(self.v1[self.pos] * self.wall_left_shape[self.pos]) * self.vel_weight
        out_right = np.sum(self.v1[self.pos] * f[-1][self.pos]) * self.vel_weight
        den_right = np.sum((-self.v1[self.neg]) * self.wall_right_shape[self.neg]) * self.vel_weight
        return (out_left / den_left) * self.wall_left_shape, (out_right / den_right) * self.wall_right_shape

    def moments(self, f: Array) -> tuple[Array, Array, Array, Array]:
        rho = np.sum(f, axis=(1, 2)) * self.vel_weight
        safe_rho = np.maximum(rho, 1.0e-12)
        ux = np.sum(f * self.v1[None], axis=(1, 2)) * self.vel_weight / safe_rho
        uy = np.sum(f * self.v2[None], axis=(1, 2)) * self.vel_weight / safe_rho
        thermal = (self.v1[None] - ux[:, None, None]) ** 2 + (self.v2[None] - uy[:, None, None]) ** 2
        temp = 0.5 * np.sum(f * thermal, axis=(1, 2)) * self.vel_weight / safe_rho
        temp = np.maximum(temp, 5.0e-3)
        return safe_rho, ux, uy, temp

    def equilibrium(self, f: Array) -> tuple[Array, Array, Array]:
        rho, ux, uy, temp = self.moments(f)
        exponent = -((self.v1[None] - ux[:, None, None]) ** 2 + (self.v2[None] - uy[:, None, None]) ** 2) / (2.0 * temp[:, None, None])
        maxwell = rho[:, None, None] * np.exp(exponent) / (2.0 * np.pi * temp[:, None, None])
        # Correct truncation so that the discrete Maxwellian has the requested
        # density. Momentum/energy truncation errors are negligible at this Lv.
        discrete_rho = np.sum(maxwell, axis=(1, 2)) * self.vel_weight
        maxwell *= (rho / discrete_rho)[:, None, None]
        return maxwell, rho, temp

    @staticmethod
    def kernel_factor(a: float, temp: Array) -> Array:
        # For two 2D Maxwellian velocities at temperature T,
        # E|v-v_*|^a = (4T)^(a/2) Gamma(1+a/2).
        return np.power(4.0 * temp, 0.5 * a) * gamma(1.0 + 0.5 * a) / (2.0 * np.pi)

    def collision(self, f: Array, a: float, b: float) -> tuple[Array, Array, Array, Array]:
        maxwell, rho, temp = self.equilibrium(f)
        nu = b * rho * self.kernel_factor(a, temp)
        q = nu[:, None, None] * (maxwell - f)
        return q, maxwell, nu, temp

    def transport(self, f: Array) -> Array:
        left, right = self._wall_boundaries(f)
        out = np.empty_like(f)
        prev = left
        for i in range(self.cfg.nx):
            out[i, self.pos] = self.v1[self.pos] * (f[i, self.pos] - prev[self.pos]) / self.dx
            prev = f[i]
        nxt = right
        for i in range(self.cfg.nx - 1, -1, -1):
            out[i, self.neg] = self.v1[self.neg] * (nxt[self.neg] - f[i, self.neg]) / self.dx
            nxt = f[i]
        return out

    def residual(self, f: Array, a: float, b: float) -> Array:
        q, _, _, _ = self.collision(f, a, b)
        return self.transport(f) - q

    def residual_sq(self, f: Array, a: float, b: float) -> float:
        r = self.residual(f, a, b)
        return float(np.sum(r * r) * self.weight)

    def solve(self, a: float, b: float, initial: Array | None = None) -> tuple[Array, dict]:
        f = np.array(self.initial if initial is None else initial, copy=True)
        f *= self.target_mass / (np.sum(f) * self.weight)
        started = time.perf_counter()
        residual_norm = math.inf
        for it in range(1, self.cfg.fom_maxiter + 1):
            maxwell, rho, temp = self.equilibrium(f)
            nu = b * rho * self.kernel_factor(a, temp)
            left, right = self._wall_boundaries(f)
            trial = np.empty_like(f)
            for i in range(self.cfg.nx):
                incoming = left if i == 0 else trial[i - 1]
                speed = self.v1[self.pos] / self.dx
                trial[i, self.pos] = (speed * incoming[self.pos] + nu[i] * maxwell[i, self.pos]) / (speed + nu[i])
            for i in range(self.cfg.nx - 1, -1, -1):
                incoming = right if i == self.cfg.nx - 1 else trial[i + 1]
                speed = -self.v1[self.neg] / self.dx
                trial[i, self.neg] = (speed * incoming[self.neg] + nu[i] * maxwell[i, self.neg]) / (speed + nu[i])
            trial *= self.target_mass / (np.sum(trial) * self.weight)
            f = (1.0 - self.cfg.fom_relaxation) * f + self.cfg.fom_relaxation * trial
            if it == 1 or it % 10 == 0:
                residual_norm = math.sqrt(self.residual_sq(f, a, b))
                if residual_norm < self.cfg.fom_tol:
                    break
        info = {
            "iterations": it,
            "residual_norm": residual_norm,
            "seconds": time.perf_counter() - started,
            "converged": bool(residual_norm < self.cfg.fom_tol),
        }
        return f, info


class FixedROM:
    def __init__(
        self,
        model: BGKModel,
        snapshots: list[Array],
        inner_log_scale: bool = False,
        inner_solver: str = "lbfgs",
    ):
        self.model = model
        matrix = np.column_stack([s.ravel() for s in snapshots])
        self.u, _ = np.linalg.qr(matrix, mode="reduced")
        self.n = self.u.shape[1]
        self.inner_log_scale = inner_log_scale
        self.inner_solver = inner_solver
        self.shaped_u = self.u.reshape(model.cfg.nx, model.cfg.nv, model.cfg.nv, self.n)
        self.mass = np.sum(self.shaped_u, axis=(0, 1, 2)) * model.weight
        self.c0 = self.mass * (model.target_mass / float(self.mass @ self.mass))
        self.z = null_space(self.mass[None, :])
        self.y0 = np.zeros(self.n - 1)
        self.last_y = self.y0.copy()
        self.eval_count = 0
        self.cache: dict[tuple[float, float], dict] = {}
        self.base_field = np.tensordot(self.shaped_u, self.c0, axes=([-1], [0]))
        self.directions = np.tensordot(self.shaped_u, self.z, axes=([-1], [0]))
        self.base_transport = model.transport(self.base_field)
        self.direction_transport = np.stack(
            [model.transport(self.directions[..., j]) for j in range(self.n - 1)], axis=-1
        ) if self.n > 1 else np.empty((*self.base_field.shape, 0))

        # Parameter-independent directional moments used by the exact reduced
        # coefficient gradient.
        h = self.directions
        self.drho = np.sum(h, axis=(1, 2)) * model.vel_weight
        self.dmx = np.sum(h * model.v1[None, :, :, None], axis=(1, 2)) * model.vel_weight
        self.dmy = np.sum(h * model.v2[None, :, :, None], axis=(1, 2)) * model.vel_weight
        self.dme2 = np.sum(h * model.vsq[None, :, :, None], axis=(1, 2)) * model.vel_weight

    def field(self, y: Array) -> Array:
        return self.base_field + np.tensordot(self.directions, y, axes=([-1], [0]))

    def _residual_jacobian(self, y: Array, a: float, b: float) -> tuple[Array, Array]:
        self.eval_count += 1
        f = self.field(y)
        m = self.model
        rho = np.sum(f, axis=(1, 2)) * m.vel_weight
        mx = np.sum(f * m.v1[None], axis=(1, 2)) * m.vel_weight
        my = np.sum(f * m.v2[None], axis=(1, 2)) * m.vel_weight
        me2 = np.sum(f * m.vsq[None], axis=(1, 2)) * m.vel_weight
        if np.min(rho) <= 1.0e-8:
            residual = np.zeros(f.size)
            residual[0] = 1.0e6 + 1.0e4 * float(np.linalg.norm(np.minimum(rho, 0.0)))
            return residual, np.zeros((f.size, self.n - 1))
        ux = mx / rho
        uy = my / rho
        e2 = me2 / rho
        temp = 0.5 * e2 - 0.5 * (ux * ux + uy * uy)
        if np.min(temp) <= 1.0e-4:
            residual = np.zeros(f.size)
            residual[0] = 1.0e6 + 1.0e4 * float(np.linalg.norm(np.minimum(temp, 0.0)))
            return residual, np.zeros((f.size, self.n - 1))

        dux = (self.dmx - ux[:, None] * self.drho) / rho[:, None]
        duy = (self.dmy - uy[:, None] * self.drho) / rho[:, None]
        de2 = (self.dme2 - e2[:, None] * self.drho) / rho[:, None]
        dtemp = 0.5 * de2 - ux[:, None] * dux - uy[:, None] * duy

        centered_x = m.v1[None] - ux[:, None, None]
        centered_y = m.v2[None] - uy[:, None, None]
        centered_sq = centered_x**2 + centered_y**2
        g = np.exp(-centered_sq / (2.0 * temp[:, None, None]))
        zsum = np.sum(g, axis=(1, 2)) * m.vel_weight
        maxwell = rho[:, None, None] * g / zsum[:, None, None]
        dlogg = (
            centered_x[..., None] * dux[:, None, None, :]
            + centered_y[..., None] * duy[:, None, None, :]
        ) / temp[:, None, None, None]
        dlogg += centered_sq[..., None] * dtemp[:, None, None, :] / (2.0 * temp[:, None, None, None] ** 2)
        mean_dlogg = np.sum(maxwell[..., None] * dlogg, axis=(1, 2)) * m.vel_weight / rho[:, None]
        dmaxwell = maxwell[..., None] * (
            self.drho[:, None, None, :] / rho[:, None, None, None]
            + dlogg
            - mean_dlogg[:, None, None, :]
        )

        nu = b * rho * m.kernel_factor(a, temp)
        dnu = nu[:, None] * (
            self.drho / rho[:, None] + 0.5 * a * dtemp / temp[:, None]
        )
        q = nu[:, None, None] * (maxwell - f)
        dq = (
            dnu[:, None, None, :] * (maxwell - f)[..., None]
            + nu[:, None, None, None] * (dmaxwell - self.directions)
        )
        transport = self.base_transport + np.tensordot(self.direction_transport, y, axes=([-1], [0]))
        residual = transport - q
        dresidual = self.direction_transport - dq
        scale = math.sqrt(m.weight)
        residual_vector = residual.ravel() * scale
        if self.n == 1:
            jacobian = np.empty((residual_vector.size, 0))
        else:
            jacobian = dresidual.reshape(residual_vector.size, self.n - 1) * scale
        return residual_vector, jacobian

    def _objective_gradient(self, y: Array, a: float, b: float) -> tuple[float, Array]:
        residual, jacobian = self._residual_jacobian(y, a, b)
        value = float(residual @ residual)
        gradient = 2.0 * (jacobian.T @ residual)
        return value, gradient

    def _objective(self, y: Array, a: float, b: float) -> float:
        return self._objective_gradient(y, a, b)[0]

    def solve_inner(self, a: float, b: float, warm: bool = False) -> dict:
        cache_key = (round(float(a), 12), round(float(b), 12))
        if not warm and cache_key in self.cache:
            cached = self.cache[cache_key]
            return {
                **cached,
                "gradient": cached["gradient"].copy(),
                "y": cached["y"].copy(),
            }
        if self.n == 1:
            y = np.empty(0)
            value = self._objective(y, a, b)
            raw_coefficient_gradient = np.empty(0)
            result_success = True
            nit = 0
        else:
            start = self.last_y if warm else self.y0
            if self.inner_solver == "least_squares":
                cache: dict[str, Array | None] = {"y": None, "residual": None, "jacobian": None}

                def evaluate(yy):
                    if cache["y"] is None or not np.array_equal(yy, cache["y"]):
                        residual, jacobian = self._residual_jacobian(yy, a, b)
                        cache["y"] = yy.copy()
                        cache["residual"] = residual
                        cache["jacobian"] = jacobian
                    return cache["residual"], cache["jacobian"]

                result = least_squares(
                    lambda yy: evaluate(yy)[0],
                    start,
                    jac=lambda yy: evaluate(yy)[1],
                    bounds=(-12.0, 12.0),
                    ftol=1.0e-12,
                    xtol=1.0e-12,
                    gtol=1.0e-11,
                    max_nfev=self.model.cfg.inner_maxiter,
                    method="trf",
                )
                y = result.x
                residual, jacobian = self._residual_jacobian(y, a, b)
                value = float(residual @ residual)
                raw_coefficient_gradient = 2.0 * (jacobian.T @ residual)
                result_success = bool(result.success or np.linalg.norm(raw_coefficient_gradient, ord=np.inf) < 1.0e-9)
                nit = int(result.nfev)
            else:
                if self.inner_log_scale:
                    def optimization_objective(y, aa, bb):
                        raw_value, raw_gradient = self._objective_gradient(y, aa, bb)
                        safe_value = max(raw_value, 1.0e-30)
                        return math.log(safe_value), raw_gradient / safe_value
                else:
                    optimization_objective = self._objective_gradient
                result = minimize(
                    optimization_objective,
                    start,
                    args=(a, b),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(-12.0, 12.0)] * (self.n - 1),
                    options={"gtol": 1.0e-9, "ftol": self.model.cfg.inner_ftol, "maxiter": self.model.cfg.inner_maxiter},
                )
                y = result.x
                value, raw_coefficient_gradient = self._objective_gradient(y, a, b)
                result_success = bool(result.success or np.linalg.norm(result.jac, ord=np.inf) < 5.0e-5)
                nit = int(result.nit)
            if warm:
                self.last_y = y.copy()
        f = self.field(y)
        r = self.model.residual(f, a, b)
        q, maxwell, nu, temp = self.model.collision(f, a, b)
        dq_db = q / b
        dlognu_da = 0.5 * (np.log(4.0 * temp) + digamma(1.0 + 0.5 * a))
        dq_da = dlognu_da[:, None, None] * nu[:, None, None] * (maxwell - f)
        grad_a = -2.0 * float(np.sum(r * dq_da) * self.model.weight)
        grad_b = -2.0 * float(np.sum(r * dq_db) * self.model.weight)
        answer = {
            "value": value,
            "gradient": np.array([grad_a, grad_b]),
            "y": y,
            "success": result_success,
            "nit": nit,
            "min_rho": float(np.min(self.model.moments(f)[0])),
            "min_temp": float(np.min(self.model.moments(f)[3])),
            "coefficient_gradient_inf": float(np.max(np.abs(raw_coefficient_gradient))) if len(raw_coefficient_gradient) else 0.0,
        }
        if not warm:
            self.cache[cache_key] = {
                **answer,
                "gradient": answer["gradient"].copy(),
                "y": answer["y"].copy(),
            }
        return answer


def scale_to_box(z: Array, box: tuple[tuple[float, float], tuple[float, float]]) -> Array:
    lower = np.array([box[0][0], box[1][0]])
    upper = np.array([box[0][1], box[1][1]])
    return lower + z * (upper - lower)


def fixed_snapshot_parameters(box: tuple[tuple[float, float], tuple[float, float]], count: int, seed: int) -> Array:
    # A deterministic scrambled Sobol design gives nested, selection-neutral
    # fixed spaces for ranks 3, 6, 9.
    sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
    return scale_to_box(sobol.random_base2(m=4)[:count], box)


def continuous_search(rom: FixedROM, box, starts: int, budget: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lo = np.array([box[0][0], box[1][0]])
    hi = np.array([box[0][1], box[1][1]])
    scale = hi - lo
    start_points = np.vstack([
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
        rng.random((max(0, starts - 5), 2)),
    ])[:starts]
    calls = 0
    local_calls = 0
    local_limit = max(1, budget // starts)
    history = []
    best = None

    class BudgetReached(RuntimeError):
        pass

    def fg(z):
        nonlocal calls, local_calls, best
        if calls >= budget or local_calls >= local_limit:
            raise BudgetReached
        calls += 1
        local_calls += 1
        mu = lo + z * scale
        inner = rom.solve_inner(*mu, warm=False)
        value = max(inner["value"], 1.0e-300)
        candidate = {"mu": mu.copy(), "value": inner["value"], "success": inner["success"]}
        if best is None or candidate["value"] > best["value"]:
            best = candidate
        # Maximize log MinRes to reduce scale sensitivity.
        return -math.log(value), -(inner["gradient"] * scale) / value

    maxiter_each = max(2, budget // starts)
    for z0 in start_points:
        if calls >= budget:
            break
        local_calls = 0
        try:
            result = minimize(
                fg,
                z0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                options={"maxiter": maxiter_each, "maxfun": max(3, budget - calls), "ftol": 1.0e-12, "gtol": 1.0e-7},
            )
            history.append({"mu": (lo + result.x * scale).tolist(), "nit": int(result.nit), "success": bool(result.success)})
        except BudgetReached:
            history.append({"stopped_by_budget": True})
            continue
    assert best is not None
    best["calls"] = calls
    best["history"] = history
    return best


def projected_adam_search(
    rom: FixedROM,
    box,
    starts: int,
    budget: int,
    seed: int,
    learning_rate: float = 0.08,
    b_power: float = 0.0,
) -> dict:
    """Projected Adam ascent on log MinRes in normalized coordinates."""
    rng = np.random.default_rng(seed)
    lo = np.array([box[0][0], box[1][0]])
    hi = np.array([box[0][1], box[1][1]])
    scale = hi - lo
    initial = np.vstack([
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
        rng.random((max(0, starts - 5), 2)),
    ])[:starts]
    states = [{"z": z.copy(), "m": np.zeros(2), "v": np.zeros(2), "t": 0} for z in initial]
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    calls = 0
    best = None
    trace = []
    while calls < budget:
        state = states[calls % starts]
        z = state["z"]
        mu = lo + z * scale
        inner = rom.solve_inner(*mu, warm=False)
        calls += 1
        raw_value = max(inner["value"], 1.0e-300)
        value = raw_value / (mu[1] ** b_power)
        log_gradient_mu = inner["gradient"] / raw_value
        log_gradient_mu[1] -= b_power / mu[1]
        grad = log_gradient_mu * scale
        candidate = {
            "mu": mu.copy(),
            "value": value,
            "raw_value": inner["value"],
            "success": inner["success"],
        }
        if best is None or candidate["value"] > best["value"]:
            best = candidate
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * grad * grad
        mhat = state["m"] / (1.0 - beta1 ** state["t"])
        vhat = state["v"] / (1.0 - beta2 ** state["t"])
        state["z"] = np.clip(z + learning_rate * mhat / (np.sqrt(vhat) + eps), 0.0, 1.0)
        trace.append({"start": calls % starts, "mu": mu.tolist(), "value": inner["value"]})
    assert best is not None
    best["calls"] = calls
    best["learning_rate"] = learning_rate
    best["b_power"] = b_power
    best["history"] = trace
    return best


def pool_search(rom: FixedROM, box, size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    points = scale_to_box(rng.random((size, 2)), box)
    values = []
    for mu in points:
        values.append(rom.solve_inner(*mu, warm=False)["value"])
    idx = int(np.argmax(values))
    return {"mu": points[idx], "value": float(values[idx]), "calls": size, "points": points, "values": np.array(values)}


def tensor_search(rom: FixedROM, box, side: int = 7) -> dict:
    """Boundary-aware low-dimensional baseline (7x7 = 49 evaluations)."""
    avec = np.linspace(*box[0], side)
    bvec = np.linspace(*box[1], side)
    points = np.array([(a, b) for b in bvec for a in avec])
    values = np.array([rom.solve_inner(*mu, warm=False)["value"] for mu in points])
    idx = int(np.argmax(values))
    return {"mu": points[idx], "value": float(values[idx]), "calls": len(points), "points": points, "values": values}


def audit_grid(rom: FixedROM, box, grid_n: int) -> dict:
    avec = np.linspace(*box[0], grid_n)
    bvec = np.linspace(*box[1], grid_n)
    vals = np.empty((grid_n, grid_n))
    successes = 0
    for j, b in enumerate(bvec):
        rom.last_y = rom.y0.copy()
        for i, a in enumerate(avec):
            inner = rom.solve_inner(a, b, warm=True)
            vals[j, i] = inner["value"]
            successes += int(inner["success"])
    ij = np.unravel_index(np.argmax(vals), vals.shape)
    return {
        "a": avec,
        "b": bvec,
        "values": vals,
        "mu": np.array([avec[ij[1]], bvec[ij[0]]]),
        "value": float(vals[ij]),
        "inner_success_rate": successes / vals.size,
    }


def gradient_check(rom: FixedROM, mu: Array, box) -> dict:
    inner = rom.solve_inner(*mu, warm=False)
    analytic = inner["gradient"]
    finite = np.empty(2)
    for k in range(2):
        width = box[k][1] - box[k][0]
        h = 2.0e-5 * width
        plus = mu.copy()
        minus = mu.copy()
        plus[k] = min(box[k][1], plus[k] + h)
        minus[k] = max(box[k][0], minus[k] - h)
        vp = rom.solve_inner(*plus, warm=False)["value"]
        vm = rom.solve_inner(*minus, warm=False)["value"]
        finite[k] = (vp - vm) / (plus[k] - minus[k])
    rel = np.abs(analytic - finite) / np.maximum.reduce([np.abs(analytic), np.abs(finite), np.full(2, 1.0e-12)])
    return {"mu": mu.tolist(), "analytic": analytic.tolist(), "finite_difference": finite.tolist(), "relative_component_error": rel.tolist()}


def relative_l2(model: BGKModel, f: Array, truth: Array) -> float:
    numerator = math.sqrt(float(np.sum((f - truth) ** 2) * model.weight))
    denominator = math.sqrt(float(np.sum(truth**2) * model.weight))
    return numerator / denominator


def deployment_errors(model: BGKModel, snapshots: list[Array], selected_mu: Array, test_params: Array, test_truth: list[Array]) -> dict:
    selected_snapshot, selected_info = model.solve(*selected_mu)
    enriched = FixedROM(model, snapshots + [selected_snapshot])
    errors = []
    inner_successes = 0
    for mu, truth in zip(test_params, test_truth):
        inner = enriched.solve_inner(*mu, warm=False)
        errors.append(relative_l2(model, enriched.field(inner["y"]), truth))
        inner_successes += int(inner["success"])
    errors = np.array(errors)
    return {
        "selected_fom": selected_info,
        "max": float(np.max(errors)),
        "p95": float(np.quantile(errors, 0.95)),
        "median": float(np.median(errors)),
        "mean": float(np.mean(errors)),
        "inner_success_rate": inner_successes / len(errors),
        "errors": errors.tolist(),
    }


def plot_case(name: str, box, rank_results: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, len(rank_results), figsize=(5.1 * len(rank_results), 4.2), constrained_layout=True)
    if len(rank_results) == 1:
        axes = [axes]
    for ax, result in zip(axes, rank_results):
        audit = result["audit"]
        image = ax.contourf(audit["a"], audit["b"], np.log10(np.maximum(audit["values"], 1.0e-300)), levels=25, cmap="viridis")
        pool = result["pool"]
        tensor = result["tensor"]
        cont = result["continuous"]
        adam = result["adam"]
        ax.scatter(pool["points"][:, 0], pool["points"][:, 1], s=8, c="white", alpha=0.35, linewidths=0)
        ax.scatter(*pool["mu"], marker="s", s=70, c="#ffbf00", edgecolor="black", label="pool-50")
        ax.scatter(*tensor["mu"], marker="D", s=55, c="#c5ff5c", edgecolor="black", label="grid-49")
        ax.scatter(*cont["mu"], marker="*", s=130, c="#ff4b5c", edgecolor="black", label="continuous")
        ax.scatter(*adam["mu"], marker="P", s=85, c="#ff8cff", edgecolor="black", label="Adam")
        ax.scatter(*audit["mu"], marker="x", s=70, c="cyan", linewidths=2.2, label="grid audit")
        ax.set_title(f"rank {result['rank']}")
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(image, ax=ax, label=r"$\log_{10}\,\mathrm{MinRes}$")
    fig.suptitle(f"{name}: fixed-space residual landscape")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def serializable_result(case_result: dict) -> dict:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items() if k not in {"points", "values", "a", "b"}}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return convert(case_result)


def run(args) -> dict:
    cfg = Config()
    model = BGKModel(cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    boxes = {
        "transport": ((-1.0, 1.0), (1.0, 10.0)),
        "collision": ((-1.0, 1.0), (20.0, 50.0)),
    }
    all_results = {"config": asdict(cfg), "screening_model": "nonlinear thermal-average BGK; not the paper's full Boltzmann operator", "cases": {}}
    for case_index, (name, box) in enumerate(boxes.items()):
        snapshot_path = output / f"{name}_snapshots.npz"
        if args.reuse_snapshots and snapshot_path.exists():
            saved = np.load(snapshot_path)
            params = saved["parameters"]
            snapshots = list(saved["snapshots"])
            fom_info = [{"mu": mu.tolist(), "loaded": True} for mu in params]
            print(f"[{name}] reused {len(snapshots)} saved FOM snapshots", flush=True)
        else:
            params = fixed_snapshot_parameters(box, max(args.ranks), cfg.seed + 1000 * case_index)
            snapshots = []
            fom_info = []
            previous = None
            for mu in params:
                snapshot, info = model.solve(*mu, initial=previous)
                snapshots.append(snapshot)
                fom_info.append({"mu": mu.tolist(), **info})
                previous = snapshot
                print(f"[{name}] FOM {len(snapshots):02d}/{len(params)} mu={mu} residual={info['residual_norm']:.3e} it={info['iterations']} t={info['seconds']:.2f}s", flush=True)
            np.savez_compressed(snapshot_path, parameters=params, snapshots=np.stack(snapshots))
        test_rng = np.random.default_rng(cfg.seed + 30_000 + case_index)
        test_params = scale_to_box(test_rng.random((20, 2)), box)
        test_truth = []
        for mu in test_params:
            truth, _ = model.solve(*mu)
            test_truth.append(truth)
        rank_results = []
        for rank in args.ranks:
            rom = FixedROM(model, snapshots[:rank])
            pool = pool_search(rom, box, args.pool_size, cfg.seed + 10_000 + 100 * case_index + rank)
            tensor = tensor_search(rom, box, side=7)
            continuous = continuous_search(rom, box, args.starts, args.budget, cfg.seed + 20_000 + 100 * case_index + rank)
            adam = projected_adam_search(rom, box, args.starts, args.budget, cfg.seed + 25_000 + 100 * case_index + rank)
            audit = audit_grid(rom, box, args.audit_grid)
            grad_mu = np.array([(box[0][0] + box[0][1]) * 0.5 + 0.17, (box[1][0] + box[1][1]) * 0.5 - 0.11 * (box[1][1] - box[1][0])])
            grad = gradient_check(rom, grad_mu, box)
            enrichment = {
                "pool50": deployment_errors(model, snapshots[:rank], pool["mu"], test_params, test_truth),
                "grid49": deployment_errors(model, snapshots[:rank], tensor["mu"], test_params, test_truth),
                "continuous": deployment_errors(model, snapshots[:rank], continuous["mu"], test_params, test_truth),
                "adam": deployment_errors(model, snapshots[:rank], adam["mu"], test_params, test_truth),
            }
            rank_result = {
                "rank": rank,
                "pool": pool,
                "tensor": tensor,
                "continuous": continuous,
                "adam": adam,
                "audit": audit,
                "gradient_check": grad,
                "one_step_deployment": enrichment,
            }
            rank_results.append(rank_result)
            print(
                f"[{name}] rank={rank} audit={audit['value']:.3e}@{audit['mu']} "
                f"pool={pool['value']:.3e}@{pool['mu']} grid={tensor['value']:.3e}@{tensor['mu']} "
                f"cont={continuous['value']:.3e}@{continuous['mu']} "
                f"adam={adam['value']:.3e}@{adam['mu']} "
                f"maxerr(pool/grid/cont/adam)={enrichment['pool50']['max']:.3e}/"
                f"{enrichment['grid49']['max']:.3e}/{enrichment['continuous']['max']:.3e}/"
                f"{enrichment['adam']['max']:.3e}",
                flush=True,
            )
        plot_case(name, box, rank_results, output / f"{name}_landscape.png")
        all_results["cases"][name] = {
            "box": box,
            "fom": fom_info,
            "ranks": [serializable_result(x) for x in rank_results],
        }
    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    return all_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="boltzmann_continuous_screen/output")
    parser.add_argument("--ranks", nargs="+", type=int, default=[3, 6, 9])
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--audit-grid", type=int, default=31)
    parser.add_argument("--reuse-snapshots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
