"""Continuous-vs-pool POD-greedy for the random-elasticity wave equation.

The model follows Dematteis--Grafke--Vanden-Eijnden (2018), Sec. 4.1:
30 masses/springs, positive random elasticity
    D_j(theta_j) = theta_j/2 + sqrt(theta_j**2/4 + 1),
zero initial data, and a time-dependent traction at the right endpoint.

Greedy search uses a reduced trajectory and its full-space dynamic residual;
the FOM is solved only for a selected snapshot and during the common audit.
Every continuous selection is one random start followed by exactly 100 Adam
updates on a fixed Gaussian-action sphere.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from contgreedy.sampling.host import T_INNER, adam_ascent


class RandomElasticRod:
    def __init__(self, n=30, final_time=1.0, dt=0.005, beta=1.5,
                 action_radius=None):
        self.n = n
        self.d = n
        self.final_time = final_time
        self.dt = dt
        self.nt = int(round(final_time / dt))
        self.times = np.linspace(0.0, final_time, self.nt + 1)
        self.beta = beta
        self.radius = float(np.sqrt(n) if action_radius is None else action_radius)
        self.h = 1.0 / n
        self.kparts = self._spring_parts()
        self.load = np.zeros(n)
        self.load[-1] = 1.0 / self.h

    def _spring_parts(self):
        parts = np.zeros((self.n, self.n, self.n))
        for j in range(self.n):
            # Spring j connects fixed node 0 to node 1 when j=0, and
            # consecutive unknown nodes otherwise.
            if j == 0:
                parts[j, 0, 0] = 1.0 / self.h**2
            else:
                parts[j, j - 1, j - 1] += 1.0 / self.h**2
                parts[j, j, j] += 1.0 / self.h**2
                parts[j, j - 1, j] -= 1.0 / self.h**2
                parts[j, j, j - 1] -= 1.0 / self.h**2
        return parts

    @staticmethod
    def elasticity(theta):
        root = np.sqrt(0.25 * theta * theta + 1.0)
        return 0.5 * theta + root

    @staticmethod
    def elasticity_prime(theta):
        root = np.sqrt(0.25 * theta * theta + 1.0)
        return 0.5 + 0.25 * theta / root

    def force(self, t):
        if self.beta == 0:
            return 1.0
        return max(t, 0.0) ** self.beta

    def project(self, theta):
        norm = np.linalg.norm(theta)
        if norm <= self.radius:
            return theta
        return theta * (self.radius / norm)

    def shell_samples(self, rng, count):
        z = rng.normal(size=(count, self.d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return self.radius * z

    def gaussian_samples(self, rng, count):
        z = rng.normal(size=(count, self.d))
        norm = np.linalg.norm(z, axis=1)
        bad = norm > self.radius
        z[bad] *= (self.radius / norm[bad])[:, None]
        return z

    def matrix(self, theta):
        return np.tensordot(self.elasticity(theta), self.kparts, axes=(0, 0))

    def full_trajectory(self, theta):
        k = self.matrix(theta)
        u = np.zeros((self.nt + 1, self.n))
        # Second-order centered update; zero initial velocity.
        a0 = self.load * self.force(0.0) - k @ u[0]
        u[1] = u[0] + 0.5 * self.dt**2 * a0
        for it in range(1, self.nt):
            a = self.load * self.force(self.times[it]) - k @ u[it]
            u[it + 1] = 2.0 * u[it] - u[it - 1] + self.dt**2 * a
        return u

    @staticmethod
    def orth_add(v, q):
        if v.size:
            q = q - v @ (v.T @ q)
            q = q - v @ (v.T @ q)
        nq = np.linalg.norm(q)
        return np.column_stack((v, q / nq)) if nq > 1e-11 else v

    def enrich(self, v, trajectory, count=1):
        residual = trajectory.T
        if v.size:
            residual = residual - v @ (v.T @ residual)
        left, _, _ = np.linalg.svd(residual, full_matrices=False)
        for j in range(min(count, left.shape[1])):
            v = self.orth_add(v, left[:, j])
        return v

    def rom_trajectory(self, theta, v):
        kr = v.T @ self.matrix(theta) @ v
        fr = v.T @ self.load
        q = np.zeros((self.nt + 1, v.shape[1]))
        q[1] = 0.5 * self.dt**2 * fr * self.force(0.0)
        for it in range(1, self.nt):
            a = fr * self.force(self.times[it]) - kr @ q[it]
            q[it + 1] = 2.0 * q[it] - q[it - 1] + self.dt**2 * a
        return q @ v.T

    def indicator_batch(self, theta, v):
        """Squared time-integrated full dynamic residual for a batch."""
        theta = np.atleast_2d(theta)
        bsz = theta.shape[0]
        dval = self.elasticity(theta)
        kr_parts = np.einsum("ni,jnk,kl->jil", v, self.kparts, v,
                             optimize=True)
        kr = np.einsum("bj,jil->bil", dval, kr_parts, optimize=True)
        pav = np.einsum("mn,jnr->jmr", np.eye(self.n) - v @ v.T,
                        np.einsum("jnk,kr->jnr", self.kparts, v), optimize=True)
        # K(theta)V is constant throughout the time integration.  Forming it
        # once avoids summing over all parameter directions at every step.
        kpv = np.einsum("bj,jnr->bnr", dval, pav, optimize=True)
        pforce = (np.eye(self.n) - v @ v.T) @ self.load
        fr = v.T @ self.load
        rdim = v.shape[1]
        qm = np.zeros((bsz, rdim))
        q = np.zeros((bsz, rdim))
        qnext = 0.5 * self.dt**2 * self.force(0.0) * fr[None, :]
        value = np.zeros(bsz)
        for it in range(1, self.nt + 1):
            qm, q = q, qnext
            kuv = np.einsum("bnr,br->bn", kpv, q, optimize=True)
            residual = self.force(self.times[it]) * pforce[None, :] - kuv
            value += self.dt * np.einsum("bn,bn->b", residual, residual)
            if it < self.nt:
                accel = self.force(self.times[it]) * fr[None, :] - np.einsum(
                    "bij,bj->bi", kr, q, optimize=True)
                qnext = 2.0 * q - qm + self.dt**2 * accel
        return value

    def indicator_grad(self, theta, v):
        """Indicator and exact forward-sensitivity gradient."""
        dval = self.elasticity(theta)
        dp = self.elasticity_prime(theta)
        av = np.einsum("jnk,kr->jnr", self.kparts, v)
        projector = np.eye(self.n) - v @ v.T
        pav = np.einsum("mn,jnr->jmr", projector, av, optimize=True)
        kpv = np.einsum("j,jnr->nr", dval, pav, optimize=True)
        pforce = projector @ self.load
        krp = np.einsum("ni,jnk,kl->jil", v, self.kparts, v,
                        optimize=True)
        kr = np.einsum("j,jil->il", dval, krp, optimize=True)
        fr = v.T @ self.load
        rdim = v.shape[1]
        qm = np.zeros(rdim); q = np.zeros(rdim)
        qnext = 0.5 * self.dt**2 * self.force(0.0) * fr
        sm = np.zeros((self.d, rdim)); s = np.zeros_like(sm)
        snext = np.zeros_like(sm)
        value = 0.0
        grad = np.zeros(self.d)
        for it in range(1, self.nt + 1):
            qm, q = q, qnext
            sm, s = s, snext
            residual = self.force(self.times[it]) * pforce - np.einsum(
                "j,jnr,r->n", dval, pav, q, optimize=True)
            # dr_l = -D'_l K_l V q - K V s_l, all projected.
            direct = dp[:, None] * np.einsum("jnr,r->jn", pav, q,
                                              optimize=True)
            indirect = np.einsum("nr,lr->ln", kpv, s, optimize=True)
            dr = -direct - indirect
            value += self.dt * float(residual @ residual)
            grad += 2.0 * self.dt * (dr @ residual)
            if it < self.nt:
                accel = self.force(self.times[it]) * fr - kr @ q
                qnext = 2.0 * q - qm + self.dt**2 * accel
                forcing_s = dp[:, None] * np.einsum("jil,l->ji", krp, q,
                                                     optimize=True)
                saccel = -np.einsum("il,jl->ji", kr, s) - forcing_s
                snext = 2.0 * s - sm + self.dt**2 * saccel
        return value, grad

    def sphere_indicator_grad(self, theta, v):
        """Indicator and its tangent gradient on the Gaussian-action sphere."""
        value, grad = self.indicator_grad(theta, v)
        grad = grad - theta * (theta @ grad) / (theta @ theta)
        return value, grad

    def true_error(self, theta, v, full=None):
        if full is None:
            full = self.full_trajectory(theta)
        reduced = self.rom_trajectory(theta, v)
        den = np.linalg.norm(full)
        return float(np.linalg.norm(full - reduced) / max(den, 1e-14))


def initial_basis(problem, modes=2):
    trajectory = problem.full_trajectory(np.zeros(problem.d))
    return problem.enrich(np.empty((problem.n, 0)), trajectory, modes)


def train(problem, method, rounds, pool, seed, learning_rate):
    rng = np.random.default_rng(seed)
    v = initial_basis(problem)
    selected = []
    parameters = []
    start = time.perf_counter()
    for _ in range(rounds):
        if method == "continuous":
            theta0 = problem.shell_samples(rng, 1)[0]
            fg = lambda z: problem.sphere_indicator_grad(z, v)
            theta, value = adam_ascent(fg, theta0, problem.project,
                                       steps=T_INNER, lr=learning_rate)
        else:
            candidates = problem.shell_samples(rng, pool)
            values = problem.indicator_batch(candidates, v)
            index = int(np.argmax(values))
            theta, value = candidates[index], float(values[index])
        trajectory = problem.full_trajectory(theta)
        v = problem.enrich(v, trajectory, 1)
        selected.append(value)
        parameters.append(theta.tolist())
    return v, time.perf_counter() - start, selected, parameters


def audit(problem, trained, gaussian=500, shell=500, adversarial=1, seed=901):
    rng = np.random.default_rng(seed)
    points = list(problem.gaussian_samples(rng, gaussian))
    points.extend(problem.shell_samples(rng, shell))
    for v, _, _, _ in trained.values():
        for z0 in problem.shell_samples(rng, adversarial):
            fg = lambda z: problem.sphere_indicator_grad(z, v)
            z, _ = adam_ascent(fg, z0, problem.project, steps=T_INNER, lr=0.12)
            points.append(z)
    full = [problem.full_trajectory(z) for z in points]
    result = {}
    for name, (v, seconds, selected, _) in trained.items():
        errors = np.array([problem.true_error(z, v, uf)
                           for z, uf in zip(points, full)])
        indicators = np.sqrt(problem.indicator_batch(np.asarray(points), v))
        result[name] = {
            "rank": int(v.shape[1]),
            "gaussian_mean": float(errors[:gaussian].mean()),
            "gaussian_q99": float(np.quantile(errors[:gaussian], 0.99)),
            "shell_max": float(errors[gaussian:gaussian + shell].max()),
            "shared_adversarial_max": float(errors.max()),
            "indicator_error_corr": float(np.corrcoef(indicators, errors)[0, 1]),
            "search_seconds": seconds,
            "selected_indicator_squared": selected,
        }
    return result


def gradient_check():
    problem = RandomElasticRod(n=8, dt=0.005)
    v = initial_basis(problem)
    rng = np.random.default_rng(2)
    theta = problem.shell_samples(rng, 1)[0]
    value, grad = problem.indicator_grad(theta, v)
    direction = rng.normal(size=problem.d)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    fd = (problem.indicator_batch(theta + eps * direction, v)[0] -
          problem.indicator_batch(theta - eps * direction, v)[0]) / (2 * eps)
    return value, float(grad @ direction), float(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--pool-large", type=int, default=10000)
    ap.add_argument("--learning-rate", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--audit", type=int, default=500)
    ap.add_argument("--output", type=Path, default=Path("random_rod_result.json"))
    args = ap.parse_args()
    check = gradient_check()
    if abs(check[1] - check[2]) > 1e-5 * max(1.0, abs(check[2])):
        raise RuntimeError(f"gradient check failed: {check}")
    problem = RandomElasticRod()
    trained = {}
    for name, method, pool in (("pool100", "pool", 100),
                               (f"pool{args.pool_large}", "pool", args.pool_large),
                               ("continuous", "continuous", 0)):
        trained[name] = train(problem, method, args.rounds, pool, args.seed,
                              args.learning_rate)
        print(name, "trained", trained[name][1], flush=True)
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "T_inner": T_INNER,
        "gradient_check": check,
        "audit": audit(problem, trained, gaussian=args.audit, shell=args.audit),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
