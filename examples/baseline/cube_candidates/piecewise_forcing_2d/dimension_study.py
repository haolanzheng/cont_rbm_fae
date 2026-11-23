"""Five-dimensional-size study of the same 2D block-source Poisson model.

Rectangular equal-area partitions extend the square partition implementation.
Existing d=16,64 training paths are reused; new paths use identical rules.
All N at fixed d are evaluated on one shared test union.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_forcing_2d_matplotlib')
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import model, ascend, append, residual_gram, extrema

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'runs/piecewise_forcing_2d_20261112/extended_N'
OUT = ROOT / 'runs/piecewise_forcing_2d_20261112/dimension_study'
PARTITIONS = [(4, 4), (4, 6), (4, 8), (6, 8), (8, 8)]
METHODS = ['gd80', 'pool256', 'pool4096', 'pool10000']
SEEDS = list(range(1000, 1004))


def rectangular_model(mx, my, cutoff=1024):
    if mx == my:
        return model(mx, cutoff)
    k = np.arange(1, cutoff + 1, dtype=float)
    def axis(m):
        r = np.arange(1, m + 1)
        Q = np.sin(np.pi * ((np.arange(m) + .5) / m)[:, None] * r)
        Q /= np.linalg.norm(Q, axis=0)
        residue = np.arange(1, cutoff + 1) % (2 * m)
        alias = np.minimum(residue, 2 * m - residue)
        amplitude = 2 * np.sqrt(2) * np.sin(np.pi * k / (2 * m)) / (np.pi * k)
        mass = np.zeros((cutoff, m))
        valid = alias > 0
        mass[np.flatnonzero(valid), alias[valid] - 1] = (amplitude**2 * np.where(alias == m, m, m / 2))[valid]
        return Q, mass
    Qx, massx = axis(mx)
    Qy, massy = axis(my)
    inverse_lambda = 1 / (np.pi**2 * (k[:, None]**2 + k[None, :]**2))
    eigenvalues = massx.T @ inverse_lambda @ massy
    omitted_mass = np.maximum(0., 1 / (mx * my) - massx.sum(0)[:, None] * massy.sum(0)[None, :])
    tail = omitted_mass / (np.pi**2 * ((cutoff + 1)**2 + 1))
    U = np.kron(Qx, Qy)
    B = np.sqrt(eigenvalues.ravel())[:, None] * U.T
    G = B.T @ B
    checks = dict(cutoff=cutoff, relative_gram_tail_bound=float(np.max(tail / eigenvalues)),
                  orthogonality_defect=float(np.linalg.norm(U.T @ U - np.eye(mx * my))),
                  constant_load_energy_squared=float(np.ones(mx * my) @ G @ np.ones(mx * my)))
    return B, G, checks


def validate_rectangular():
    checks = []
    for mx, my, cutoff in [(4, 6, 47), (4, 8, 49), (6, 8, 53)]:
        B, G, _ = rectangular_model(mx, my, cutoff)
        k = np.arange(1, cutoff + 1, dtype=float)
        def integrals(m):
            left = np.arange(m) / m
            return np.sqrt(2) * (np.cos(np.pi * k[:, None] * left) - np.cos(np.pi * k[:, None] * (left + 1 / m))) / (np.pi * k[:, None])
        F = np.einsum('ki,lj->klij', integrals(mx), integrals(my)).reshape(cutoff**2, mx * my)
        lam = (np.pi**2 * (k[:, None]**2 + k[None, :]**2)).ravel()
        direct_G = F.T @ (F / lam[:, None])
        relative = float(np.linalg.norm(direct_G - G) / np.linalg.norm(direct_G))
        rng = np.random.default_rng(154 + mx * my)
        Z = rng.uniform(-1, 1, (mx * my, 5))
        snapshots = (F / lam[:, None]) @ Z
        c = rng.uniform(-1, 1, mx * my)
        true = (F @ c) / lam
        reduced = snapshots @ np.linalg.solve(snapshots.T @ (lam[:, None] * snapshots), snapshots.T @ (F @ c))
        direct_error = float(np.sum(lam * (true - reduced)**2))
        Q = np.linalg.qr(B @ Z)[0]
        compressed_error = float(c @ residual_gram(B, G, Q) @ c)
        difference = abs(direct_error - compressed_error) / direct_error
        assert relative < 1e-12 and difference < 1e-11
        checks.append(dict(partition=[mx, my], Gram_relative_error=relative, Galerkin_relative_error=difference))
    return checks


def study(mx, my):
    started = time.perf_counter()
    d = mx * my
    max_n = 3 * d // 4
    B, G, checks = rectangular_model(mx, my)
    B2, _, checks2 = rectangular_model(mx, my, 2048)
    checks['double_cutoff_relative_eigenvalue_change'] = float(np.max(np.abs(np.sum(B2**2, axis=1) / np.sum(B**2, axis=1) - 1)))
    checks['double_cutoff_relative_tail_bound'] = checks2['relative_gram_tail_bound']
    saved = np.load(OLD / f'd{d}_snapshots.npz') if d in (16, 64) else None
    initial_squared = float(np.ones(d) @ G @ np.ones(d))
    assert G.min() > 0
    spaces, parameters, training = {}, {}, []
    for seed in SEEDS:
        candidates = np.random.default_rng(seed).uniform(-1, 1, (10000, d))
        for method in METHODS:
            tick = time.perf_counter()
            Q = np.zeros((d, 0))
            selected, records = [], []
            init = np.random.default_rng(seed + 10000)
            old_z = None
            if saved is not None:
                prefix = f'z_s{seed}_{method}_n'
                key = max((k for k in saved.files if k.startswith(prefix)), key=lambda k: int(k.rsplit('n', 1)[1]))
                old_z = saved[key][:max_n]
                assert len(old_z) == max_n
            if method.startswith('pool'):
                pool = candidates[:int(method[4:])]
                pool_values = pool @ B.T
                scores = np.sum(pool_values**2, axis=1)
            for n in range(max_n):
                H = residual_gram(B, G, Q)
                if old_z is not None:
                    z = old_z[n]
                elif method == 'gd80':
                    z = ascend(H, init.uniform(-1, 1, d), 80)
                else:
                    z = pool[np.argmax(scores)]
                if old_z is not None and method.startswith('pool'):
                    assert np.max(abs(z - pool[np.argmax(scores)])) < 1e-12
                J = float(z @ H @ z)
                Q, q = append(B, Q, z)
                selected.append(z.copy())
                records.append(dict(N=n + 1, J=J))
                if method.startswith('pool'):
                    scores -= (pool_values @ q)**2
            spaces[(seed, method)] = Q
            parameters[f'z_s{seed}_{method}'] = np.asarray(selected)
            training.append(dict(seed=seed, method=method, reused_existing_path=old_z is not None,
                                 execution_seconds=time.perf_counter() - tick, rounds=records))
        print(f'd={d}: seed {seed}, training ready', flush=True)
    np.savez_compressed(OUT / f'd{d}_snapshots.npz', **parameters)

    # One test set, shared by every method, seed and rank for this d.
    rng = np.random.default_rng(82000 + d)
    tests = [rng.uniform(-1, 1, (4096, d)), rng.choice([-1., 1.], (4096, d)),
             np.ones((1, d)), -np.ones((1, d))]
    if d == 16:
        bits = (np.arange(2**d, dtype=np.uint32)[:, None] >> np.arange(d)) & 1
        tests.append(2 * bits.astype(float) - 1)
    else:
        for (seed, method), Q in spaces.items():
            for n in range(1, max_n + 1):
                H = residual_gram(B, G, Q[:, :n])
                tests.append(extrema(H, rng.uniform(-1, 1, (64, d))))
    test = np.unique(np.vstack(tests), axis=0)
    np.savez_compressed(OUT / f'd{d}_test.npz', parameters=test)
    U = test @ B.T
    unorm = np.sum(U**2, axis=1)
    eigenvalues = np.linalg.eigvalsh(G)[::-1]
    bounds = np.sqrt(np.maximum(0., np.r_[eigenvalues.sum(), eigenvalues.sum() - np.cumsum(eigenvalues)[:max_n]]) / initial_squared)
    rows = []
    for (seed, method), Q in spaces.items():
        coeff = U @ Q
        e2 = np.maximum(0., unorm[:, None] - np.cumsum(coeff**2, axis=1))
        curve = np.r_[1., np.sqrt(e2.max(axis=0) / initial_squared)]
        assert np.all(np.diff(curve) <= 1e-12)
        direct = U - coeff @ Q.T
        assert abs(np.sum(direct**2, axis=1).max() - e2[:, -1].max()) < 1e-12
        rows.append(dict(seed=seed, method=method, error_over_initial_maximum=curve.tolist()))
    summary = []
    for n in range(max_n + 1):
        r = dict(N=n, any_space_lower_bound=float(bounds[n]))
        for method in METHODS:
            values = [x['error_over_initial_maximum'][n] for x in rows if x['method'] == method]
            r[method] = dict(median=float(np.median(values)), minimum=float(min(values)), maximum=float(max(values)))
        summary.append(r)
    report = dict(settings=dict(d=d, partition=[mx, my], maximum_N=max_n, alpha=.6, T=80,
                                seeds=SEEDS, methods=METHODS, test_count=len(test),
                                finite_test_is_global=(d == 16),
                                norm='energy error / initial global maximum energy error',
                                initial_global_maximum=float(np.sqrt(initial_squared)),
                                test='Common union across all ranks, methods and seeds: uniform interior, vertices, constant signs and independent vertex searches; complete vertex enumeration at d16.',
                                device='CPU, float64, one BLAS thread'),
                  checks=checks, training=training, curves=rows, summary=summary,
                  elapsed_seconds=time.perf_counter() - started)
    (OUT / f'd{d}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(d=d, test_count=len(test), elapsed=report['elapsed_seconds'], final=summary[-1])), flush=True)
    return report


def plot(reports):
    colors = ['#245681', '#929AA2', '#D09B38', '#38877A']
    styles = ['-', ':', '--', '-.']
    labels = ['GD, T=80', 'Pool 256', 'Pool 4096', 'Pool 10000']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), sharey=True)
    for ax, report in zip(axes.flat, reports):
        rows = report['summary']
        n = np.array([r['N'] for r in rows])
        for method, color, style, label in zip(METHODS, colors, styles, labels):
            ax.plot(n, [r[method]['median'] for r in rows], color=color, ls=style,
                    lw=2.2 if method == 'gd80' else 1.7, label=label)
        s = report['settings']
        ax.set_title(f"d = {s['d']}  ({s['partition'][0]} x {s['partition'][1]} blocks)")
        ax.set_yscale('log')
        ax.set_ylim(.09, 1.06)
        ax.set_xlim(0, s['maximum_N'])
        ax.set_xticks(np.unique(np.linspace(0, s['maximum_N'], 5).round().astype(int)))
        ax.set_xlabel('Reduced dimension N')
        ax.grid(alpha=.18, which='both')
    for ax in axes[:, 0]:
        ax.set_ylabel('Max. energy error / initial maximum')
    axes[1, 2].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc='center', frameon=False, fontsize=12)
    axes[1, 2].text(.5, .16, 'Median of 4 seeds; shared test set\nGlobal maximum: d=16\nTested maximum: d=24,32,48,64',
                    transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=10, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT / 'error_vs_N.png', dpi=190)
    fig.savefig(OUT / 'error_vs_N.pdf')
    plt.close(fig)
    # Same relative reduced dimension, to avoid mixing d with N.
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for report in reports:
        d = report['settings']['d']
        ax.plot([r['N'] / d for r in report['summary']], [r['gd80']['median'] for r in report['summary']], label=f'd={d}', lw=1.8)
    ax.set_yscale('log'); ax.set_xlabel('N / d'); ax.set_ylabel('Maximum test energy error / initial maximum')
    ax.set_xlim(0, .75); ax.grid(alpha=.18, which='both'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / 'gd_error_vs_fraction.png', dpi=190); fig.savefig(OUT / 'gd_error_vs_fraction.pdf'); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    audit = validate_rectangular()
    (OUT / 'validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Rectangular sine/Galerkin checks passed', audit, flush=True)
    reports = [study(mx, my) for mx, my in PARTITIONS]
    plot(reports)
    (OUT / 'summary.json').write_text(json.dumps([dict(settings=r['settings'], checks=r['checks'], summary=r['summary']) for r in reports], indent=2) + '\n')
    print('COMPLETE', flush=True)
