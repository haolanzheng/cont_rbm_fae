"""Reproducible CPU pilot: Matern input fields for elliptic FAE/RB experiments.

No FAE or greedy optimization is trained here. Covariance samples use circulant
embedding, not a low-rank KL truncation. Only tiny FFT roundoff is clipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/continuous_greedy_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.special import gamma, kv
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits


def covariance(r, nu, length):
    """Unit variance; conventional sqrt(2 nu) r / length Matern convention."""
    q = np.sqrt(2 * nu) * r / length
    c = np.ones_like(q)
    pos = q > 0
    if nu == 0.5:
        return np.exp(-q)
    if nu == 1.5:
        return (1 + q) * np.exp(-q)
    if nu == 2.5:
        return (1 + q + q*q/3) * np.exp(-q)
    c[pos] = 2**(1-nu) / gamma(nu) * q[pos]**nu * kv(nu, q[pos])
    return c


class MaternSampler:
    def __init__(self, n, nu, length):
        self.n, self.nu, self.length = n, nu, length
        m = 4*n
        for _ in range(5):
            x = np.minimum(np.arange(m), m-np.arange(m))/n
            c = covariance(np.hypot(x[:, None], x[None, :]), nu, length)
            eigen = np.fft.rfft2(c).real
            tolerance = 1e-12 * eigen.max()
            if eigen.min() >= -tolerance:
                self.m = m
                self.minimum_eigenvalue = float(eigen.min())
                self.relative_clipped_mass = float(-eigen[eigen < 0].sum()/eigen.sum())
                self.root = np.sqrt(np.maximum(eigen, 0))
                return
            m *= 2
        raise RuntimeError("Circulant embedding not positive; increase padding.")

    def draw(self, rng, count):
        out = np.empty((count, self.n+1, self.n+1))
        for i in range(count):
            white = rng.normal(size=(self.m, self.m))
            field = np.fft.irfft2(np.fft.rfft2(white)*self.root, s=white.shape)
            out[i] = field[:self.n+1, :self.n+1]
        return out


def coefficient(g, amplitude=np.log(10.0)):
    """Bounded log-conductivity. No per-sample rescaling or thresholding."""
    return np.exp(amplitude*np.tanh(g))


class PoissonP1:
    """P1 Galerkin on uniform triangulation, homogeneous Dirichlet boundary."""
    def __init__(self, n):
        self.n = n
        x = np.linspace(0, 1, n+1)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        self.xy = np.column_stack((xx.ravel(), yy.ravel()))
        base = (np.arange(n)[:, None]*(n+1) + np.arange(n)[None, :]).ravel()
        self.tri = np.concatenate((np.stack((base, base+1, base+n+2), axis=1),
                                   np.stack((base, base+n+2, base+n+1), axis=1)))
        pts = self.xy[self.tri]
        transform = np.concatenate((np.ones((*pts.shape[:2], 1)), pts), axis=2)
        grad = np.linalg.inv(transform)[:, 1:, :].transpose(0, 2, 1)
        area = 1/(2*n*n)
        self.ke = area*np.einsum("eai,ebi->eab", grad, grad)
        me = area*(np.ones((3, 3))+np.eye(3))/12
        self.rows = np.repeat(self.tri, 3, axis=1).ravel()
        self.cols = np.tile(self.tri, (1, 3)).ravel()
        self.size = (n+1)**2
        self.mass = sparse.coo_matrix((np.tile(me, (len(self.tri), 1, 1)).ravel(),
                                      (self.rows, self.cols)), shape=(self.size, self.size)).tocsr()
        self.stiffness = self.matrix(np.ones(self.size))
        self.interior = np.flatnonzero((self.xy[:, 0] > 0) & (self.xy[:, 0] < 1)
                                      & (self.xy[:, 1] > 0) & (self.xy[:, 1] < 1))
        self.rhs = self.mass@np.ones(self.size)

    def matrix(self, a):
        # a_h is the nodal P1 interpolant, integrated exactly on each triangle.
        a_element = np.asarray(a).ravel()[self.tri].mean(axis=1)
        return sparse.coo_matrix(((a_element[:, None, None]*self.ke).ravel(),
                                 (self.rows, self.cols)), shape=(self.size, self.size)).tocsr()

    def solve(self, a, source=None):
        mat = self.matrix(a)
        b = self.rhs if source is None else self.mass@source
        ii = self.interior
        interior_mat = mat[ii][:, ii]
        u = np.zeros(self.size)
        u[ii] = spsolve(interior_mat, b[ii])
        relative_residual = np.linalg.norm(interior_mat@u[ii]-b[ii])/np.linalg.norm(b[ii])
        return u, float(relative_residual)


def p1_prolong(u, coarse_n, fine_n):
    v = np.asarray(u).reshape(coarse_n+1, coarse_n+1)
    s = np.linspace(0, coarse_n, fine_n+1)
    jj, ii = np.meshgrid(s, s, indexing="ij")
    i, j = np.minimum(ii.astype(int), coarse_n-1), np.minimum(jj.astype(int), coarse_n-1)
    xi, eta = ii-i, jj-j
    return np.where(xi >= eta,
                    (1-xi)*v[j, i]+(xi-eta)*v[j, i+1]+eta*v[j+1, i+1],
                    (1-eta)*v[j, i]+xi*v[j+1, i+1]+(eta-xi)*v[j+1, i]).ravel()


def spectrum(samples, matrix, centered):
    x = samples.reshape(len(samples), -1)
    if centered:
        x = x-x.mean(axis=0, keepdims=True)
    gram = x@(matrix@x.T)/len(x)
    vals = np.maximum(np.linalg.eigvalsh((gram+gram.T)/2)[::-1], 0)
    fractions = vals/vals.sum()
    tail = np.sqrt(np.maximum(1-np.cumsum(fractions), 0))
    ranks = {str(p): int(np.searchsorted(np.cumsum(fractions), p)+1) for p in (.9, .95, .99)}
    return {"centered": centered, "squared_energy_ranks": ranks, "relative_tail": tail.tolist()}


def manufactured_check():
    errors = []
    residuals = []
    for n in (16, 32):
        model = PoissonP1(n)
        exact = np.sin(np.pi*model.xy[:, 0])*np.sin(np.pi*model.xy[:, 1])
        u, res = model.solve(np.full(model.size, 2.0), 4*np.pi**2*exact)
        e = u-exact
        errors.append(float(np.sqrt(e@(model.mass@e))))
        residuals.append(res)
    assert max(residuals) < 1e-9
    assert errors[1] < .4*errors[0]
    return {"nodal_interpolant_L2_errors": errors, "refinement_ratio": errors[0]/errors[1],
            "maximum_relative_linear_residual": max(residuals)}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2)+"\n")


def plot_panel(path, fields, titles, value_label, cmap="viridis", shared_limits=None):
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 6.0), constrained_layout=True)
    for ax, field, title in zip(axs.ravel(), fields, titles):
        im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                       interpolation="nearest", vmin=None if shared_limits is None else shared_limits[0],
                       vmax=None if shared_limits is None else shared_limits[1])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x_1$", fontsize=9)
        ax.set_ylabel("$x_2$", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, shrink=.78, label=value_label)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(out, screen_count, train_count, val_count, test_count, selected_nu, selected_length):
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out/"run.log"), logging.StreamHandler()])
    start = time.perf_counter()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    settings = {"pde": "-div(a grad u)=1 on (0,1)^2; u=0 on boundary", "cpu_threads": 1,
                "screen_grid_intervals": 64, "screen_samples": screen_count,
                "selected_grid_intervals": 128, "selected_nu": selected_nu,
                "selected_length": selected_length, "grf_mean": 0, "grf_variance": 1,
                "matern_convention": "q=sqrt(2*nu)*r/length; C=2^(1-nu)/Gamma(nu)*q^nu*K_nu(q)",
                "coefficient": "a=exp(log(10)*tanh(g)); 0.1<a<10",
                "splits": {"train": train_count, "validation": val_count, "test": test_count},
                "seeds": {"screen": 20261122, "train": 20261123, "validation": 20261124,
                          "test": 20261125, "mesh_check": 20261126},
                "purpose": "input-generation and FOM/POD pilot only; no FAE or greedy advantage claim"}
    save_json(out/"settings.json", settings)
    checks = {"manufactured_solution": manufactured_check()}
    logging.info("Manufactured solution check: %s", checks["manufactured_solution"])
    choices = [(.5, .1), (.5, .2), (.5, .3), (1., .2), (1.5, .2), (2.5, .3)]
    model = PoissonP1(64)
    results, images, titles = [], [], []
    for nu, length in choices:
        sampler = MaternSampler(64, nu, length)
        g = sampler.draw(np.random.default_rng(20261122), screen_count)
        a = coefficient(g)
        u, residuals = zip(*(model.solve(ai) for ai in a))
        u = np.asarray(u)
        max_residual = max(residuals)
        assert max_residual < 1e-9
        result = {"nu": nu, "length": length, "embedding_grid": sampler.m,
                  "minimum_embedding_eigenvalue": sampler.minimum_eigenvalue,
                  "relative_clipped_spectral_mass": sampler.relative_clipped_mass,
                  "empirical_mean_g": float(g.mean()), "empirical_second_moment_g": float(np.mean(g*g)),
                  "coefficient_range": [float(a.min()), float(a.max())],
                  "maximum_relative_linear_residual": max_residual,
                  "coefficient_L2": spectrum(a, model.mass, True),
                  "solution_L2": spectrum(u, model.mass, False),
                  "solution_H1_seminorm": spectrum(u, model.stiffness, False)}
        results.append(result)
        images.append(np.log10(a[0]))
        titles.append(rf"$\nu={nu:g},\quad \ell={length:g}$")
        logging.info("screen nu=%s length=%s ranks95 a/uL2/uH1=%s/%s/%s", nu, length,
                     *(result[k]["squared_energy_ranks"]["0.95"] for k in
                       ("coefficient_L2", "solution_L2", "solution_H1_seminorm")))
        save_json(out/"screening.json", results)
    plot_panel(out/"roughness_comparison", images, titles, r"$\log_{10}a$", shared_limits=(-1, 1))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, key, title in zip(axs, ("coefficient_L2", "solution_L2", "solution_H1_seminorm"),
                              ("Coefficient: centered $L^2$ POD", "Solution: $L^2$ POD", "Solution: $H^1_0$ POD")):
        for result in results:
            tail = np.asarray(result[key]["relative_tail"])
            end = min(64, len(tail)-1)
            ax.semilogy(np.arange(1, end+1), tail[:end], marker="o", markevery=8, ms=3,
                        label=rf"$\nu={result['nu']:g},\ell={result['length']:g}$")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Relative root-mean-square projection error")
        ax.set_ylim(.005, 1)
        ax.grid(alpha=.2)
    axs[0].legend(ncol=2)
    fig.savefig(out/"compression_comparison.pdf", bbox_inches="tight")
    fig.savefig(out/"compression_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    model = PoissonP1(128)
    sampler = MaternSampler(128, selected_nu, selected_length)
    preview_a, preview_u = [], []
    for split, count, seed in (("train", train_count, 20261123), ("validation", val_count, 20261124),
                               ("test", test_count, 20261125)):
        g = sampler.draw(np.random.default_rng(seed), count)
        a = coefficient(g)
        u = np.empty_like(g)
        residuals = []
        for i, ai in enumerate(a):
            ui, residual = model.solve(ai)
            u[i] = ui.reshape(129, 129)
            residuals.append(residual)
        assert max(residuals) < 1e-9
        np.savez_compressed(out/f"{split}.npz", g=g.astype(np.float32), a=a.astype(np.float32),
                            u=u.astype(np.float32), x=np.linspace(0, 1, 129), seed=seed)
        checks[split] = {"samples": count, "maximum_relative_linear_residual": max(residuals),
                         "embedding_grid": sampler.m, "grf_mean": float(g.mean()),
                         "grf_second_moment": float(np.mean(g*g))}
        if split == "test":
            preview_a = list(np.log10(a[:6]))
            preview_u = list(u[:6])
            checks["selected_test_POD"] = {"coefficient_L2": spectrum(a, model.mass, True),
                                           "solution_L2": spectrum(u, model.mass, False),
                                           "solution_H1_seminorm": spectrum(u, model.stiffness, False)}
        logging.info("Saved %s: %s samples on 129x129 nodes", split, count)
    plot_panel(out/"selected_coefficients", preview_a, [f"Sample {j+1}" for j in range(6)],
               r"$\log_{10}a$", shared_limits=(-1, 1))
    plot_panel(out/"selected_solutions", preview_u, [f"Sample {j+1}" for j in range(6)], "$u$")

    # Coupled fields: all resolutions are restrictions of the same 257x257 sample.
    fine_sampler = MaternSampler(256, selected_nu, selected_length)
    fine_g = fine_sampler.draw(np.random.default_rng(20261126), 4)
    models = {n: PoissonP1(n) for n in (64, 128, 256)}
    mesh = []
    for gi in fine_g:
        fields = {n: coefficient(gi[::256//n, ::256//n]) for n in models}
        solutions = {n: models[n].solve(fields[n])[0] for n in models}
        record = {}
        for coarse, fine in ((64, 128), (128, 256)):
            err = p1_prolong(solutions[coarse], coarse, fine)-solutions[fine]
            mass, stiffness = models[fine].mass, models[fine].matrix(fields[fine])
            uf = solutions[fine]
            record[f"{coarse}_to_{fine}"] = {
                "relative_L2": float(np.sqrt(err@(mass@err)/(uf@(mass@uf)))),
                "relative_a_energy": float(np.sqrt(err@(stiffness@err)/(uf@(stiffness@uf))))}
        mesh.append(record)
    checks["coupled_mesh_comparison"] = mesh
    checks["elapsed_seconds"] = time.perf_counter()-start
    checks["storage"] = "float32 snapshots; generation, PDE solves, diagnostics float64"
    save_json(out/"checks.json", checks)
    logging.info("Mesh 128->256 relative a-energy errors: %s", [x["128_to_256"]["relative_a_energy"] for x in mesh])
    logging.info("Finished in %.1fs", checks["elapsed_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-samples", type=int, default=96)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=.2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.screen_samples, args.train, args.validation, args.test, args.nu, args.length)
