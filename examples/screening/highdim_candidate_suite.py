"""High-dimensional continuous-greedy candidate screening suite.

Every continuous selection uses one fresh random start and exactly 100 Adam
updates.  Linear models admit an exact operator-norm audit.  Nonlinear models
are oracle geometry screens: they use differentiated full-model projection
error and are not claimed as deployable residual-based greedy methods.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from contgreedy.sampling.host import T_INNER, adam_ascent


def sphere_samples(rng, count, d, radius=None):
    radius = np.sqrt(d) if radius is None else radius
    z = rng.normal(size=(count, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return radius * z


def sphere_project(z, radius):
    n = np.linalg.norm(z)
    return z * (radius / n) if n else np.r_[radius, np.zeros(len(z) - 1)]


def tangent(z, g):
    return g - z * (z @ g) / (z @ z)


def residual(states, basis):
    if basis.size == 0:
        return states
    return states - (states @ basis) @ basis.T


def add_basis(basis, state):
    q = residual(state[None, :], basis)[0]
    if basis.size:
        q = q - basis @ (basis.T @ q)
    n = np.linalg.norm(q)
    return np.column_stack((basis, q / n)) if n > 1e-11 else basis


class LinearMapModel:
    def __init__(self, name, operator):
        self.name = name
        self.operator = np.asarray(operator, dtype=float)
        self.n, self.d = self.operator.shape
        self.radius = np.sqrt(self.d)

    def samples(self, rng, count):
        return sphere_samples(rng, count, self.d, self.radius)

    def project(self, z):
        return sphere_project(z, self.radius)

    def states(self, z):
        return np.atleast_2d(z) @ self.operator.T

    def objective_batch(self, z, basis):
        r = residual(self.states(z), basis)
        return np.einsum("bi,bi->b", r, r)

    def objective_grad(self, z, basis):
        state = self.operator @ z
        r = residual(state[None, :], basis)[0]
        value = float(r @ r)
        grad = 2.0 * self.operator.T @ r
        return value, tangent(z, grad)

    def exact_audit(self, basis):
        rop = residual(self.operator.T, basis).T
        singular = np.linalg.svd(rop, compute_uv=False)
        return {
            "exact_max": float(self.radius * singular[0]),
            "exact_rms": float(np.linalg.norm(singular)),
            "operator_rank": int(np.count_nonzero(singular > 1e-11)),
        }


def distributed_wave_model(nm=8, nf=16, nt=240):
    """Space-time distributed forcing for a damped 1-D wave equation."""
    d = nm * nf
    final = 2.0
    dt = final / nt
    modal_w = np.pi * np.arange(1, nm + 1)
    forcing_w = np.pi * np.arange(1, nf + 1)
    damping = 0.025
    q = np.zeros((d, nm)); vel = np.zeros_like(q)
    states = []
    mode_id = np.repeat(np.arange(nm), nf)
    freq_id = np.tile(np.arange(nf), nm)
    for it in range(nt + 1):
        t = it * dt
        states.append(q.copy())
        f = np.zeros_like(q)
        f[np.arange(d), mode_id] = np.sin(forcing_w[freq_id] * t)
        acc = f - 2 * damping * modal_w[None, :] * vel - modal_w[None, :]**2 * q
        vel = vel + dt * acc
        q = q + dt * vel
    op = np.stack(states, axis=1).reshape(d, -1).T * np.sqrt(dt)
    return LinearMapModel("distributed_wave_load", op)


def rte_inflow_model(nx=36, nmu=64, scatter=0.82):
    """Steady slab RTE with a function-valued angular inflow."""
    mu, weights = np.polynomial.legendre.leggauss(nmu)
    dx = 1.0 / nx
    n = nx * nmu
    a = lil_matrix((n, n), dtype=float)
    b = np.zeros((n, nmu))
    incoming = {}
    col = 0
    for m in range(nmu):
        if mu[m] > 0:
            incoming[(m, "left")] = col; col += 1
        else:
            incoming[(m, "right")] = col; col += 1
    for i in range(nx):
        for m in range(nmu):
            row = i * nmu + m
            speed = abs(mu[m]) / dx
            a[row, row] += speed + 1.0
            for mp in range(nmu):
                a[row, i * nmu + mp] -= 0.5 * scatter * weights[mp]
            if mu[m] > 0:
                if i > 0:
                    a[row, (i - 1) * nmu + m] -= speed
                else:
                    b[row, incoming[(m, "left")]] = speed
            else:
                if i < nx - 1:
                    a[row, (i + 1) * nmu + m] -= speed
                else:
                    b[row, incoming[(m, "right")]] = speed
    op = splu(csc_matrix(a)).solve(b)
    scale = np.tile(np.sqrt(weights * dx), nx)
    op *= scale[:, None]
    return LinearMapModel("rte_function_inflow", op)


def buckling_imperfection_model(d=128, load_fraction=0.985):
    """Pinned Euler--Bernoulli column driven by a random imperfection field."""
    k = np.arange(1, d + 1, dtype=float)
    # Unit-EI modal resolvent of w'''' + P w'' = f near first buckling load.
    denom = (np.pi * k) ** 4 - load_fraction * np.pi**2 * (np.pi * k) ** 2
    gains = 1.0 / denom
    # Orthonormal physical displacement samples.
    x = (np.arange(d) + 1) / (d + 1)
    modes = np.sqrt(2.0 / (d + 1)) * np.sin(np.pi * x[:, None] * k[None, :])
    op = modes * gains[None, :]
    op /= np.linalg.norm(op, ord="fro") / np.sqrt(d)
    return LinearMapModel("buckling_random_imperfection", op)


def pure_transport_model(d=128):
    """Unitary advection of an unrestricted random initial field."""
    # A periodic translation is orthogonal, hence all singular values coincide.
    shift = 17
    op = np.roll(np.eye(d), shift, axis=0)
    return LinearMapModel("random_initial_advection", op)


class CapillaryBundle:
    """High-dimensional permeability and regularized two-phase fronts.

    Eight parallel channels each contain sixteen uncertain permeability cells.
    Fixed total injection partitions according to the harmonic conductance of
    every complete channel.  The state is a regularized saturation front.
    """
    name = "two_phase_channel_bundle"

    def __init__(self, ny=8, nx=16, nt=36, log_contrast=3.5, width=0.045):
        self.ny, self.nx, self.nt = ny, nx, nt
        self.d = ny * nx
        self.radius = np.sqrt(self.d)
        self.log_contrast = log_contrast
        self.width = width
        self.x = (np.arange(nx) + 0.5) / nx
        self.times = np.linspace(0.08, 1.0, nt)
        self.n = nt * ny * nx

    def samples(self, rng, count):
        return sphere_samples(rng, count, self.d, self.radius)

    def project(self, z):
        return sphere_project(z, self.radius)

    def _velocity(self, z):
        z = np.atleast_2d(z).reshape(-1, self.ny, self.nx)
        logk = self.log_contrast * np.tanh(z)
        resistance = np.mean(np.exp(-logk), axis=2)
        conduct = 1.0 / resistance
        velocity = self.ny * conduct / conduct.sum(axis=1, keepdims=True)
        return velocity, z, logk, resistance, conduct

    def states(self, z):
        velocity, *_ = self._velocity(z)
        arg = (velocity[:, None, :, None] * self.times[None, :, None, None]
               - self.x[None, None, None, :]) / self.width
        sat = 1.0 / (1.0 + np.exp(-np.clip(arg, -40, 40)))
        return sat.reshape(len(velocity), -1) / np.sqrt(self.nt * self.ny * self.nx)

    def objective_batch(self, z, basis, batch=512):
        z = np.atleast_2d(z)
        out = np.empty(len(z))
        for lo in range(0, len(z), batch):
            hi = min(lo + batch, len(z))
            r = residual(self.states(z[lo:hi]), basis)
            out[lo:hi] = np.einsum("bi,bi->b", r, r)
        return out

    def objective_grad(self, theta, basis):
        velocity, z3, logk, resistance, conduct = self._velocity(theta)
        state = self.states(theta)[0]
        r = residual(state[None, :], basis)[0].reshape(
            self.nt, self.ny, self.nx) / np.sqrt(self.nt * self.ny * self.nx)
        arg = (velocity[:, None, :, None] * self.times[None, :, None, None]
               - self.x[None, None, None, :]) / self.width
        sat = 1.0 / (1.0 + np.exp(-np.clip(arg, -40, 40)))
        dsdv = sat * (1 - sat) * self.times[None, :, None, None] / self.width
        djdv = 2.0 * np.sum(r[None, ...] * dsdv, axis=(1, 3))[0]
        c = conduct[0]; total = c.sum()
        # dv_y / dc_l.
        jvc = self.ny * (np.eye(self.ny) * total - c[:, None]) / total**2
        djdc = djdv @ jvc
        # c_y = 1 / mean(exp(-a*tanh(z_yj))).
        sech2 = 1.0 - np.tanh(z3[0]) ** 2
        dc_dz = (np.exp(-logk[0]) * self.log_contrast * sech2 /
                 (self.nx * resistance[0, :, None] ** 2))
        grad = (djdc[:, None] * dc_dz).reshape(-1)
        value = float(np.dot(r.reshape(-1), r.reshape(-1)) *
                      self.nt * self.ny * self.nx)
        return value, tangent(theta, grad)


class AllenCahnField:
    name = "phase_field_random_initial"

    def __init__(self, nx=128, d=64, steps=80, final=0.35, eps=0.035):
        self.nx, self.d, self.steps = nx, d, steps
        self.radius = np.sqrt(d)
        self.dt = final / steps
        self.eps = eps
        x = 2 * np.pi * np.arange(nx) / nx
        basis = []
        for k in range(1, d // 2 + 1):
            decay = 0.22 / np.sqrt(1 + (k / 8) ** 4)
            basis.extend([decay * np.cos(k * x), decay * np.sin(k * x)])
        self.input_basis = np.asarray(basis[:d]).T
        wave = np.fft.fftfreq(nx, d=1.0 / nx)
        self.diff_half = np.exp(-0.5 * self.dt * eps**2 * wave**2)
        self.n = nx

    def samples(self, rng, count):
        return sphere_samples(rng, count, self.d, self.radius)

    def project(self, z):
        return sphere_project(z, self.radius)

    def _diffuse(self, u):
        return np.fft.ifft(np.fft.fft(u, axis=-1) * self.diff_half,
                           axis=-1).real

    def states(self, z):
        z = np.atleast_2d(z)
        u = z @ self.input_basis.T
        e = np.exp(-2 * self.dt)
        for _ in range(self.steps):
            u = self._diffuse(u)
            den = np.sqrt(e + (1 - e) * u * u)
            u = u / den
            u = self._diffuse(u)
        return u / np.sqrt(self.nx)

    def objective_batch(self, z, basis, batch=512):
        z = np.atleast_2d(z); out = np.empty(len(z))
        for lo in range(0, len(z), batch):
            hi = min(lo + batch, len(z))
            r = residual(self.states(z[lo:hi]), basis)
            out[lo:hi] = np.einsum("bi,bi->b", r, r)
        return out

    def objective_grad(self, theta, basis):
        u = theta @ self.input_basis.T
        tape = []
        e = np.exp(-2 * self.dt)
        for _ in range(self.steps):
            x = self._diffuse(u); tape.append(x)
            u = x / np.sqrt(e + (1 - e) * x * x)
            u = self._diffuse(u)
        state = u / np.sqrt(self.nx)
        r = residual(state[None, :], basis)[0]
        value = float(r @ r)
        g = 2 * r / np.sqrt(self.nx)
        for x in reversed(tape):
            g = self._diffuse(g)
            g *= e / (e + (1 - e) * x * x) ** 1.5
            g = self._diffuse(g)
        grad = self.input_basis.T @ g
        return value, tangent(theta, grad)


class BurgersField:
    """Differentiable shock analogue for high-dimensional inflow uncertainty."""
    name = "burgers_shock_field"

    def __init__(self, nx=128, d=64, steps=180, final=0.55):
        self.nx, self.d, self.steps = nx, d, steps
        self.radius = np.sqrt(d)
        self.dx = 2 * np.pi / nx
        self.dt = final / steps
        x = 2 * np.pi * np.arange(nx) / nx
        basis = []
        for k in range(1, d // 2 + 1):
            decay = 0.12 / np.sqrt(1 + (k / 10) ** 4)
            basis.extend([decay * np.cos(k*x), decay * np.sin(k*x)])
        self.input_basis = np.asarray(basis[:d]).T
        self.base = 0.75 * np.sin(x)
        self.n = nx

    def samples(self, rng, count):
        return sphere_samples(rng, count, self.d, self.radius)

    def project(self, z):
        return sphere_project(z, self.radius)

    def states(self, z):
        z = np.atleast_2d(z)
        u = self.base[None, :] + z @ self.input_basis.T
        c = self.dt / (2 * self.dx)
        for _ in range(self.steps):
            up = np.roll(u, -1, axis=1); um = np.roll(u, 1, axis=1)
            u = 0.5 * (up + um) - c * (up**2 - um**2)
        return u / np.sqrt(self.nx)

    def objective_batch(self, z, basis, batch=512):
        z = np.atleast_2d(z); out = np.empty(len(z))
        for lo in range(0, len(z), batch):
            hi = min(lo + batch, len(z))
            r = residual(self.states(z[lo:hi]), basis)
            out[lo:hi] = np.einsum("bi,bi->b", r, r)
        return out

    def objective_grad(self, theta, basis):
        u = self.base + theta @ self.input_basis.T
        tape = []; c = self.dt / (2 * self.dx)
        for _ in range(self.steps):
            tape.append(u)
            up = np.roll(u, -1); um = np.roll(u, 1)
            u = 0.5 * (up + um) - c * (up**2 - um**2)
        state = u / np.sqrt(self.nx)
        r = residual(state[None, :], basis)[0]
        value = float(r @ r)
        g = 2 * r / np.sqrt(self.nx)
        for u in reversed(tape):
            g = ((0.5 - 2*c*u) * np.roll(g, 1) +
                 (0.5 + 2*c*u) * np.roll(g, -1))
        grad = self.input_basis.T @ g
        return value, tangent(theta, grad)


@dataclass
class Trained:
    basis: np.ndarray
    selected: list
    seconds: float


def train(model, method, rounds, seed, pool=None, pool_states=None,
          learning_rate=0.1):
    rng = np.random.default_rng(seed)
    theta_init = model.samples(np.random.default_rng(700), 1)[0]
    basis = add_basis(np.empty((model.n, 0)), model.states(theta_init)[0])
    selected = []
    start = time.perf_counter()
    for _ in range(rounds):
        if method == "continuous":
            z0 = model.samples(rng, 1)[0]
            z, val = adam_ascent(lambda x: model.objective_grad(x, basis),
                                 z0, model.project, T_INNER, learning_rate)
        else:
            if pool_states is None:
                vals = model.objective_batch(pool, basis)
            else:
                rr = residual(pool_states, basis)
                vals = np.einsum("bi,bi->b", rr, rr)
            idx = int(np.argmax(vals)); z = pool[idx]; val = float(vals[idx])
        basis = add_basis(basis, model.states(z)[0])
        selected.append(val)
    return Trained(basis, selected, time.perf_counter() - start)


def screen_model(model, rounds, pool_large, seed, learning_rates, audit_count):
    rng = np.random.default_rng(seed)
    pool = model.samples(rng, pool_large)
    # Cache cheap-to-store nonlinear endpoints; this keeps the pool baseline
    # from re-solving the same full model in every greedy round.
    pool_states = model.states(pool) if model.n <= 512 else None
    initial_theta = model.samples(np.random.default_rng(700), 1)[0]
    initial = add_basis(np.empty((model.n, 0)), model.states(initial_theta)[0])
    if pool_states is None:
        pool_values = model.objective_batch(pool, initial)
    else:
        rr = residual(pool_states, initial)
        pool_values = np.einsum("bi,bi->b", rr, rr)
    z0 = model.samples(np.random.default_rng(seed), 1)[0]
    lr_scan = {}
    for lr in learning_rates:
        z, val = adam_ascent(lambda x: model.objective_grad(x, initial),
                             z0, model.project, T_INNER, lr)
        lr_scan[str(lr)] = float(val)
    lr = max(learning_rates, key=lambda x: lr_scan[str(x)])
    trained = {
        "pool100": train(model, "pool", rounds, seed, pool[:100],
                         None if pool_states is None else pool_states[:100]),
        f"pool{pool_large}": train(model, "pool", rounds, seed, pool,
                                   pool_states),
        "continuous": train(model, "continuous", rounds, seed,
                            learning_rate=lr),
    }
    audit_z = model.samples(np.random.default_rng(991), audit_count)
    audit_states = model.states(audit_z)
    shared_states = audit_states
    if not isinstance(model, LinearMapModel):
        adversarial = []
        arng = np.random.default_rng(1881)
        for result in trained.values():
            for z0 in model.samples(arng, 2):
                z, _ = adam_ascent(
                    lambda x, b=result.basis: model.objective_grad(x, b),
                    z0, model.project, T_INNER, lr)
                adversarial.append(z)
        if adversarial:
            shared_states = np.vstack((audit_states,
                                       model.states(np.asarray(adversarial))))
    audit = {}
    for name, result in trained.items():
        err = np.linalg.norm(residual(audit_states, result.basis), axis=1)
        norm = np.linalg.norm(audit_states, axis=1)
        rel = err / np.maximum(norm, 1e-14)
        shared_err = np.linalg.norm(residual(shared_states, result.basis), axis=1)
        shared_norm = np.linalg.norm(shared_states, axis=1)
        item = {
            "rank": int(result.basis.shape[1]),
            "absolute_mean": float(err.mean()),
            "absolute_q99": float(np.quantile(err, .99)),
            "absolute_max": float(err.max()),
            "relative_mean": float(rel.mean()),
            "relative_q99": float(np.quantile(rel, .99)),
            "relative_max": float(rel.max()),
            "shared_adversarial_absolute_max": float(shared_err.max()),
            "shared_adversarial_relative_max": float(
                np.max(shared_err / np.maximum(shared_norm, 1e-14))),
            "seconds": result.seconds,
            "selected_objective": result.selected,
        }
        if isinstance(model, LinearMapModel):
            item.update(model.exact_audit(result.basis))
        audit[name] = item
    return {
        "dimension": model.d,
        "state_dimension": model.n,
        "T_inner": T_INNER,
        "pool_initial": {
            "100": float(pool_values[:100].max()),
            str(pool_large): float(pool_values.max()),
        },
        "learning_rate_scan": lr_scan,
        "chosen_learning_rate": lr,
        "audit": audit,
    }


def gradient_check(model, seed=4):
    rng = np.random.default_rng(seed)
    z = model.samples(rng, 2)
    basis = add_basis(np.empty((model.n, 0)), model.states(z[0])[0])
    value, grad = model.objective_grad(z[1], basis)
    direction = rng.normal(size=model.d)
    direction = tangent(z[1], direction); direction /= np.linalg.norm(direction)
    eps = 2e-6
    fp = model.objective_batch(model.project(z[1] + eps*direction), basis)[0]
    fm = model.objective_batch(model.project(z[1] - eps*direction), basis)[0]
    return [float(value), float(grad @ direction), float((fp-fm)/(2*eps))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--pool-large", type=int, default=10000)
    ap.add_argument("--audit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--models", nargs="*", default=[
        "wave", "two_phase", "rte", "buckling", "burgers", "transport",
        "phase_field"])
    ap.add_argument("--output", type=Path,
                    default=Path("highdim_candidate_results.json"))
    args = ap.parse_args()
    factories = {
        "wave": distributed_wave_model,
        "two_phase": CapillaryBundle,
        "rte": rte_inflow_model,
        "buckling": buckling_imperfection_model,
        "burgers": BurgersField,
        "transport": pure_transport_model,
        "phase_field": AllenCahnField,
    }
    result = {"config": vars(args).copy(), "models": {}}
    result["config"]["output"] = str(result["config"]["output"])
    for key in args.models:
        print("building", key, flush=True)
        model = factories[key]()
        check = gradient_check(model)
        if abs(check[1]-check[2]) > 3e-4*max(1.0, abs(check[2])):
            raise RuntimeError(f"{key} gradient check failed: {check}")
        start = time.perf_counter()
        item = screen_model(model, args.rounds, args.pool_large, args.seed,
                            [0.003, 0.01, 0.03, 0.1, 0.3], args.audit)
        item["gradient_check"] = check
        item["wall_seconds"] = time.perf_counter() - start
        result["models"][key] = item
        args.output.write_text(json.dumps(result, indent=2))
        print(key, json.dumps(item["audit"]), flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
