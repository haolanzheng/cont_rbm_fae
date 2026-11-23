"""The same 2D block-source model on the unit round parameter sphere.

Keep B, alpha=.6, T=80, four seeds and nested pool sizes unchanged.
Worst energy errors and n-widths are computed spectrally, not by test samples.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_forcing_2d_matplotlib')
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dimension_study import rectangular_model, PARTITIONS, METHODS, SEEDS
from run import append, residual_gram

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.baseline.geometry_cpu import search, uniform_parameters

OUT = ROOT / 'runs/piecewise_forcing_2d_20261112/sphere_study'
CUBE = ROOT / 'runs/piecewise_forcing_2d_20261112/dimension_study'


def study(mx, my):
    started = time.perf_counter()
    d = mx * my
    max_n = 3 * d // 4
    B, G, checks = rectangular_model(mx, my)
    eigenvalues = np.linalg.eigvalsh(G)[::-1]
    initial = float(np.sqrt(eigenvalues[0]))
    width = np.sqrt(eigenvalues[:max_n + 1]) / initial
    curves, parameters = [], {}
    for seed in SEEDS:
        pool = uniform_parameters(np.random.default_rng(seed), 10000, d, 'sphere')
        for method in METHODS:
            tick = time.perf_counter()
            Q = np.zeros((d, 0))
            init = np.random.default_rng(seed + 10000)
            selected, rounds = [], []
            errors = [1.]
            if method.startswith('pool'):
                candidates = pool[:int(method[4:])]
                solutions = candidates @ B.T
                scores = np.sum(solutions**2, axis=1)
            for n in range(max_n):
                H = residual_gram(B, G, Q)
                if method == 'gd80':
                    z0 = uniform_parameters(init, 1, d, 'sphere')[0]
                    z, audit = search(H, z0, steps=80, domain='sphere', alpha=.6)
                    assert audit['max_feasibility_defect'] < 1e-12
                else:
                    z = candidates[np.argmax(scores)]
                    audit = {}
                assert abs(np.linalg.norm(z) - 1) < 1e-12
                selected.append(z.copy())
                J = float(z @ H @ z)
                maximum_J = float(np.linalg.eigvalsh(H)[-1])
                Q, q = append(B, Q, z)
                if method.startswith('pool'):
                    scores -= (solutions @ q)**2
                Hnew = residual_gram(B, G, Q)
                error = float(np.sqrt(max(0., np.linalg.eigvalsh(Hnew)[-1])) / initial)
                assert error + 1e-10 >= width[n + 1]
                assert error <= errors[-1] + 1e-12
                # Independent singular-value check of exact worst error.
                R = B - Q @ (Q.T @ B)
                direct_error = float(np.linalg.svd(R, compute_uv=False)[0] / initial)
                assert abs(error - direct_error) < 1e-10
                errors.append(error)
                rounds.append(dict(N=n + 1, selected_J=J, selected_J_over_maximum=J / maximum_J, audit=audit))
            parameters[f'z_s{seed}_{method}'] = np.asarray(selected)
            curves.append(dict(seed=seed, method=method, error_over_initial_maximum=errors,
                               execution_seconds=time.perf_counter() - tick, rounds=rounds))
        print(f'd={d}: seed {seed} complete', flush=True)
    summary = []
    for n in range(max_n + 1):
        row = dict(N=n, n_width_over_initial_maximum=float(width[n]))
        for method in METHODS:
            values = [r['error_over_initial_maximum'][n] for r in curves if r['method'] == method]
            row[method] = dict(median=float(np.median(values)), minimum=float(min(values)), maximum=float(max(values)))
        summary.append(row)
    cube = json.loads((CUBE / f'd{d}.json').read_text())
    report = dict(settings=dict(d=d, partition=[mx, my], parameter_domain='unit Euclidean sphere',
                                metric='standard round metric', maximum_N=max_n, alpha=.6, T=80,
                                seeds=SEEDS, pool_sizes=[256, 4096, 10000],
                                initialization='independent normalized Gaussian direction per round',
                                pool='fixed nested normalized Gaussian directions',
                                initial_maximum_energy_error=initial,
                                error='exact maximum over sphere / initial maximum; finite sine model',
                                update='round-sphere exponential map of alpha*v/sqrt(1+||v||^2), v=2(Hz-(z^THz)z)/||H||_F',
                                device='CPU, float64, one BLAS thread'), checks=checks,
                  curves=curves, summary=summary, cube_summary=cube['summary'],
                  elapsed_seconds=time.perf_counter() - started)
    np.savez_compressed(OUT / f'd{d}_snapshots.npz', **parameters)
    (OUT / f'd{d}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(d=d, final=summary[-1], cube_final=cube['summary'][-1], elapsed=report['elapsed_seconds'])), flush=True)
    return report


def plot(reports):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = ['#245681', '#929AA2', '#D09B38', '#38877A']
    styles = ['-', ':', '--', '-.']
    labels = ['GD, T=80', 'Pool 256', 'Pool 4096', 'Pool 10000']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), sharey=True)
    for ax, report in zip(axes.flat, reports):
        rows = report['summary']; s = report['settings']
        n = np.array([r['N'] for r in rows])
        for method, color, style, label in zip(METHODS, colors, styles, labels):
            ax.plot(n, [r[method]['median'] for r in rows], color=color, ls=style,
                    lw=2.2 if method == 'gd80' else 1.6, label=label)
        ax.plot(n, [r['n_width_over_initial_maximum'] for r in rows], color='#202020',
                ls='--', lw=1.2, label='Exact n-width')
        ax.set_title(f"d = {s['d']}  ({s['partition'][0]} x {s['partition'][1]} blocks)")
        ax.set_yscale('log'); ax.set_xlim(0, s['maximum_N']); ax.set_ylim(.09, 1.06)
        ax.set_xticks(np.unique(np.linspace(0, s['maximum_N'], 5).round().astype(int)))
        ax.set_xlabel('Reduced dimension N'); ax.grid(alpha=.18, which='both')
    for ax in axes[:, 0]:ax.set_ylabel('Worst energy error / initial maximum')
    axes[1, 2].axis('off')
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(h, l, loc='center', frameon=False, fontsize=12)
    axes[1, 2].text(.5, .12, 'Unit sphere; median of 4 seeds\nWorst errors computed by eigenvalues\nSame PDE, alpha=0.6, T=80', transform=axes[1, 2].transAxes,
                    ha='center', va='center', fontsize=10, linespacing=1.5)
    fig.tight_layout(); fig.savefig(OUT / 'sphere_error_vs_N.png', dpi=190); fig.savefig(OUT / 'sphere_error_vs_N.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for report in reports:
        d = report['settings']['d']
        for ax, rows in zip(axes, [report['cube_summary'], report['summary']]):
            ax.plot([r['N'] for r in rows][1:], [r['gd80']['median'] for r in rows][1:], label=f'd={d}', lw=1.8)
    for ax, title in zip(axes, ['Cube: GD (tested maximum)', 'Sphere: GD (exact maximum)']):
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(title)
        ax.set_xlabel('Reduced dimension N'); ax.grid(alpha=.18, which='both'); ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('Energy error / initial maximum of each domain')
    fig.tight_layout(); fig.savefig(OUT / 'cube_sphere_GD_loglog.png', dpi=190); fig.savefig(OUT / 'cube_sphere_GD_loglog.pdf'); plt.close(fig)


def plot_domain_methods(reports):
    """Matched rows compare all search methods on both parameter domains."""
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = ['#245681', '#929AA2', '#D09B38', '#38877A']
    styles = ['-', ':', '--', '-.']
    labels = ['GD, T=80', 'Pool 256', 'Pool 4096', 'Pool 10000']
    fig, axes = plt.subplots(len(reports), 2, figsize=(10.4, 13.4), sharey=True)
    for row, report in enumerate(reports):
        d = report['settings']['d']
        for col, records in enumerate([report['cube_summary'], report['summary']]):
            ax = axes[row, col]
            n = np.array([r['N'] for r in records][1:])
            for method, color, style, label in zip(METHODS, colors, styles, labels):
                ax.plot(n, [r[method]['median'] for r in records][1:],
                        color=color, ls=style, lw=2.2 if method == 'gd80' else 1.7,
                        label=label)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, n[-1]); ax.set_ylim(.09, 1.02)
            ax.set_title(('Cube' if col == 0 else 'Sphere') + f': d={d}', fontsize=12)
            ticks = sorted(set([x for x in [1, 2, 4, 8, 16, 32] if x <= n[-1] / 1.25] + [int(n[-1])]))
            ax.set_xticks(ticks, labels=[str(x) for x in ticks])
            ax.minorticks_off(); ax.grid(alpha=.18)
            if col == 0: ax.set_ylabel(r'Energy error / $E_0$')
            if row == len(reports) - 1: ax.set_xlabel('Reduced dimension N')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.))
    fig.text(.5, .012, 'Median of 4 seeds; each domain uses its own initial maximum E_0.\n'
             'Cube: tested maximum (global for d=16). Sphere: exact maximum from eigenvalues.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, .975])
    fig.savefig(OUT / 'cube_sphere_all_methods.png', dpi=190)
    fig.savefig(OUT / 'cube_sphere_all_methods.pdf')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    reports = [study(mx, my) for mx, my in PARTITIONS]
    plot(reports)
    plot_domain_methods(reports)
    (OUT / 'summary.json').write_text(json.dumps([dict(settings=r['settings'], checks=r['checks'], summary=r['summary']) for r in reports], indent=2) + '\n')
    print('COMPLETE', flush=True)
