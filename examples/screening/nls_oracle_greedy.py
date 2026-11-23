"""Oracle geometry test for continuous greedy on focusing NLS random fields.

This follows the high-dimensional random Fourier initial-condition construction
of Dematteis, Grafke, and Vanden-Eijnden (2018).  The default has 91 complex
Fourier coefficients, hence 182 real parameters, on a fixed Gaussian-action
sphere.  Evolution uses a converged Strang split-step discretization of

    u_xi = (i/2) u_tt + i |u|^2 u.

The greedy target is the exact final-state projection error.  Thus this is an
*oracle geometry screen*, deliberately separated from a deployable RB residual
indicator.  If even the oracle cannot beat a random pool, the problem should be
rejected before building a nonlinear ROM/error estimator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from contgreedy.sampling.host import T_INNER, adam_ascent


class FocusingNLS:
    def __init__(self, nx=256, modes=45, box=30.0, distance=0.2,
                 steps=100, spectral_width=np.pi, mean_power=1.25):
        self.nx = nx
        self.modes = modes
        self.ncomplex = 2 * modes + 1
        self.d = 2 * self.ncomplex
        self.box = box
        self.distance = distance
        self.steps = steps
        self.h = distance / steps
        self.radius = np.sqrt(self.d)
        self.x = np.arange(nx) * box / nx
        self.wave = 2.0 * np.pi * np.fft.fftfreq(nx, d=box / nx)
        self.linear_half = np.exp(-0.25j * self.wave**2 * self.h)
        self.mode_numbers = np.arange(-modes, modes + 1)
        omega = 2.0 * np.pi * self.mode_numbers / box
        spectrum = np.exp(-0.5 * (omega / spectral_width) ** 2)
        spectrum *= mean_power / spectrum.sum()
        self.spectrum = spectrum
        phase = np.exp(1j * self.x[:, None] * omega[None, :])
        amp = np.sqrt(spectrum / 2.0)[None, :]
        self.parameter_basis = np.empty((nx, self.d), dtype=complex)
        self.parameter_basis[:, 0::2] = amp * phase
        self.parameter_basis[:, 1::2] = 1j * amp * phase

    def project(self, theta):
        norm = np.linalg.norm(theta)
        if norm == 0:
            out = np.zeros_like(theta); out[0] = self.radius
            return out
        return theta * (self.radius / norm)

    def samples(self, rng, count):
        theta = rng.normal(size=(count, self.d))
        theta /= np.linalg.norm(theta, axis=1, keepdims=True)
        return self.radius * theta

    def initial(self, theta):
        return np.asarray(theta) @ self.parameter_basis.T

    def _linear(self, u, adjoint=False):
        phase = np.conj(self.linear_half) if adjoint else self.linear_half
        return np.fft.ifft(np.fft.fft(u, axis=-1) * phase, axis=-1)

    def evolve_batch(self, theta, batch=512):
        theta = np.atleast_2d(theta)
        out = np.empty((len(theta), self.nx), dtype=complex)
        for lo in range(0, len(theta), batch):
            hi = min(lo + batch, len(theta))
            u = self.initial(theta[lo:hi])
            for _ in range(self.steps):
                u = self._linear(u)
                u = u * np.exp(1j * self.h * np.abs(u) ** 2)
                u = self._linear(u)
            out[lo:hi] = u
        return out

    def evolve_with_tape(self, theta):
        u = self.initial(theta[None, :])[0]
        tape = []
        for _ in range(self.steps):
            x = self._linear(u)
            tape.append(x)
            u = x * np.exp(1j * self.h * np.abs(x) ** 2)
            u = self._linear(u)
        return u, tape

    @staticmethod
    def residual(u, v):
        if v.size == 0:
            return u.copy()
        return u - v @ (np.conj(v).T @ u)

    @staticmethod
    def add(v, u):
        q = FocusingNLS.residual(u, v)
        nq = np.linalg.norm(q)
        return np.column_stack((v, q / nq)) if nq > 1e-12 else v

    def objective_batch(self, theta, v, endpoints=None):
        u = self.evolve_batch(theta) if endpoints is None else endpoints
        if v.size:
            coeff = u @ np.conj(v)
            value = (np.sum(np.abs(u) ** 2, axis=1) -
                     np.sum(np.abs(coeff) ** 2, axis=1)) / self.nx
        else:
            value = np.sum(np.abs(u) ** 2, axis=1) / self.nx
        return np.maximum(value.real, 0.0)

    def objective_grad(self, theta, v):
        u, tape = self.evolve_with_tape(theta)
        r = self.residual(u, v)
        value = float(np.vdot(r, r).real / self.nx)
        # Real-complex convention: dJ = Re <g, du>.
        g = 2.0 * r / self.nx
        for x in reversed(tape):
            g = self._linear(g, adjoint=True)
            phase = np.exp(1j * self.h * np.abs(x) ** 2)
            c = np.conj(g) * phase
            alpha = 2.0 * self.h * np.real(1j * c * x)
            g = g * np.conj(phase) + alpha * x
            g = self._linear(g, adjoint=True)
        grad = np.real(np.conj(g) @ self.parameter_basis)
        return value, grad

    def sphere_objective_grad(self, theta, v):
        """Objective with the Riemannian gradient on the action sphere.

        Adam's diagonal preconditioner does not preserve radial/tangential
        splitting.  Removing the radial component before Adam is therefore
        essential; projecting only the updated parameter can even turn a
        purely radial gradient into a spurious downhill tangential step.
        """
        value, grad = self.objective_grad(theta, v)
        grad = grad - theta * (theta @ grad) / (theta @ theta)
        return value, grad

    def relative_error_batch(self, endpoints, v):
        total = np.sum(np.abs(endpoints) ** 2, axis=1)
        if v.size:
            coeff = endpoints @ np.conj(v)
            residual = total - np.sum(np.abs(coeff) ** 2, axis=1)
        else:
            residual = total
        return np.sqrt(np.maximum(residual, 0.0) / np.maximum(total, 1e-30))


def gradient_check():
    p = FocusingNLS(nx=128, modes=8, steps=30)
    rng = np.random.default_rng(3)
    base = p.samples(rng, 2)
    endpoint = p.evolve_batch(base[:1])[0]
    v = p.add(np.empty((p.nx, 0), dtype=complex), endpoint)
    theta = base[1]
    direction = rng.normal(size=p.d); direction /= np.linalg.norm(direction)
    value, grad = p.objective_grad(theta, v)
    eps = 2e-6
    fp = p.objective_batch(theta + eps * direction, v)[0]
    fm = p.objective_batch(theta - eps * direction, v)[0]
    return value, float(grad @ direction), float((fp - fm) / (2 * eps))


def initial_basis(problem, seed=700):
    rng = np.random.default_rng(seed)
    theta = problem.samples(rng, 1)
    u = problem.evolve_batch(theta)[0]
    return problem.add(np.empty((problem.nx, 0), dtype=complex), u)


def train_pool(problem, pool, endpoints, rounds, initial):
    v = initial.copy()
    selected = []
    for _ in range(rounds):
        values = problem.objective_batch(pool, v, endpoints)
        index = int(np.argmax(values))
        v = problem.add(v, endpoints[index])
        selected.append(float(values[index]))
    return v, selected


def train_continuous(problem, rounds, seed, learning_rate, initial):
    rng = np.random.default_rng(seed)
    v = initial.copy()
    selected = []
    parameters = []
    for _ in range(rounds):
        theta0 = problem.samples(rng, 1)[0]
        theta, value = adam_ascent(lambda z: problem.sphere_objective_grad(z, v),
                                   theta0, problem.project,
                                   steps=T_INNER, lr=learning_rate)
        endpoint = problem.evolve_batch(theta[None, :])[0]
        v = problem.add(v, endpoint)
        selected.append(value)
        parameters.append(theta.tolist())
    return v, selected, parameters


def audit(problem, trained, count=10000, adversarial=4, seed=991):
    rng = np.random.default_rng(seed)
    theta = problem.samples(rng, count)
    endpoints = problem.evolve_batch(theta)
    extra_theta = []
    extra_u = []
    for v, _, _, _ in trained.values():
        for z0 in problem.samples(rng, adversarial):
            z, _ = adam_ascent(lambda x: problem.sphere_objective_grad(x, v),
                               z0, problem.project, steps=200, lr=0.05)
            extra_theta.append(z)
            extra_u.append(problem.evolve_batch(z[None, :])[0])
    if extra_u:
        shared = np.vstack((endpoints, np.asarray(extra_u)))
    else:
        shared = endpoints
    out = {}
    for name, (v, seconds, selected, _) in trained.items():
        random_error = problem.relative_error_batch(endpoints, v)
        shared_error = problem.relative_error_batch(shared, v)
        random_absolute = np.sqrt(problem.objective_batch(
            theta, v, endpoints))
        shared_absolute = np.sqrt(np.maximum(
            np.sum(np.abs(shared) ** 2, axis=1) / problem.nx -
            (np.sum(np.abs(shared @ np.conj(v)) ** 2, axis=1) / problem.nx
             if v.size else 0.0), 0.0))
        out[name] = {
            "rank": int(v.shape[1]),
            "random_absolute_mean": float(random_absolute.mean()),
            "random_absolute_q99": float(np.quantile(random_absolute, 0.99)),
            "random_absolute_max": float(random_absolute.max()),
            "shared_adversarial_absolute_max": float(shared_absolute.max()),
            "random_mean": float(random_error.mean()),
            "random_q99": float(np.quantile(random_error, 0.99)),
            "random_max": float(random_error.max()),
            "shared_adversarial_max": float(shared_error.max()),
            "search_seconds": seconds,
            "selected_objective": selected,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--modes", type=int, default=45)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--pool-large", type=int, default=10000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--audit", type=int, default=10000)
    ap.add_argument("--audit-starts", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("nls_oracle_result.json"))
    args = ap.parse_args()
    check = gradient_check()
    if abs(check[1] - check[2]) > 2e-5 * max(1.0, abs(check[2])):
        raise RuntimeError(f"gradient check failed: {check}")
    p = FocusingNLS(nx=args.nx, modes=args.modes, steps=args.steps)
    initial = initial_basis(p)
    rng = np.random.default_rng(args.seed)
    pool_large = p.samples(rng, args.pool_large)
    start = time.perf_counter(); endpoints_large = p.evolve_batch(pool_large)
    pool_propagation = time.perf_counter() - start
    pool_small = pool_large[:100]; endpoints_small = endpoints_large[:100]
    trained = {}
    for name, pool, endpoints in (("pool100", pool_small, endpoints_small),
                                  (f"pool{args.pool_large}", pool_large, endpoints_large)):
        start = time.perf_counter()
        v, selected = train_pool(p, pool, endpoints, args.rounds, initial)
        trained[name] = (v, time.perf_counter() - start + pool_propagation,
                         selected, [])
        print(name, "trained", trained[name][1], flush=True)
    start = time.perf_counter()
    v, selected, pars = train_continuous(p, args.rounds, args.seed,
                                         args.learning_rate, initial)
    trained["continuous"] = (v, time.perf_counter() - start, selected, pars)
    print("continuous trained", trained["continuous"][1], flush=True)
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "T_inner": T_INNER,
        "gradient_check": check,
        "note": "Oracle final-state projection error; not a deployable RB indicator.",
        "audit": audit(p, trained, count=args.audit,
                       adversarial=args.audit_starts),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
