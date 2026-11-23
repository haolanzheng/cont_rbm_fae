"""Compare cube boundary updates for the piecewise-source Poisson example.

Use the published model, initial parameters, and shared test sets. Training
and diagnostics are separate; this study reports errors, not wall times.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path
import numpy as np
from run import model, ascend, append, residual_gram

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'runs/piecewise_forcing_2d_20261112'
METHODS = ('clipping', 'active_face', 'adaptive_normal')
SEEDS = range(1000, 1004)


def search(H, initial, method, steps=80, alpha=.6):
    if method == 'clipping':
        assert alpha == .6
        return ascend(H, initial, steps)
    z = initial.copy()
    ell = max(float(np.linalg.norm(H)), 1e-30)
    hz = H @ z
    best, score = z.copy(), float(z @ hz)
    for _ in range(steps):
        v = 2 * hz / ell
        if method == 'active_face':
            v[(np.abs(z) >= 1 - 1e-14) & (z * v > 0)] = 0
            step = alpha * v / np.sqrt(1 + v @ v)
            moving = step != 0
            distances = np.where(step > 0, 1-z, 1+z)
            fraction = min(1., float(np.min(distances[moving] /
                             np.abs(step[moving])))) if moving.any() else 0.
            z = np.clip(z + fraction * step, -1, 1)
        elif method == 'adaptive_normal':
            distance = np.where(v >= 0, 1-z, 1+z)
            z = z + alpha * np.minimum(1., distance) * v / np.sqrt(1 + v @ v)
        else:
            raise ValueError(method)
        assert np.max(np.abs(z)) <= 1 + 1e-14
        hz = H @ z
        value = float(z @ hz)
        if value > score:
            score, best = value, z.copy()
    return best


def reference_parameters(d, seed):
    if d == 4:
        rows = json.loads((BASE/'slides_summary/d4.json').read_text())['curves']
        return np.asarray(next(r['parameters'] for r in rows
                          if r['seed'] == seed and r['method'] == 'gd80'))
    folder = 'dimension_study' if d == 16 else 'square_dimension_study'
    with np.load(BASE/f'{folder}/d{d}_snapshots.npz') as saved:
        return saved[f'z_s{seed}_gd80'].copy()


def study(d, output):
    B, G, checks = model(int(np.sqrt(d)))
    test = np.load(BASE/f'slides_summary/d{d}_test.npz')['parameters']
    values = test @ B.T
    norm_squared = np.sum(values**2, axis=1)
    previous = json.loads((BASE/'slides_summary/shared_test_errors.json').read_text())[str(d)]
    max_n = len(reference_parameters(d, 1000))
    records, parameters = [], {}
    for seed in SEEDS:
        for method in METHODS:
            initial = np.random.default_rng(seed+10000)
            Q = np.zeros((d, 0))
            selected = []
            for _ in range(max_n):
                H = residual_gram(B, G, Q)
                z = search(H, initial.uniform(-1, 1, d), method)
                Q, _ = append(B, Q, z)
                selected.append(z)
            selected = np.asarray(selected)
            squared = norm_squared[:, None] - np.column_stack((
                np.zeros(len(test)), np.cumsum((values @ Q)**2, axis=1)))
            errors = np.sqrt(np.maximum(0., squared.max(axis=0)))
            # Independent final-error computation, without subtraction.
            direct = np.linalg.norm(values - (values @ Q) @ Q.T, axis=1).max()
            np.testing.assert_allclose(errors[-1], direct, rtol=1e-10, atol=1e-13)
            np.testing.assert_allclose(Q.T @ Q, np.eye(max_n), atol=1e-12)
            if method == 'clipping':
                np.testing.assert_array_equal(selected, reference_parameters(d, seed))
                reference = next(r['energy_error'] for r in previous['curves']
                                 if r['seed'] == seed and r['method'] == 'gd80')
                np.testing.assert_allclose(errors, reference[:max_n+1], atol=1e-12, rtol=1e-10)
            records.append(dict(seed=seed, method=method, energy_error=errors.tolist()))
            parameters[f'z_s{seed}_{method}'] = selected
    np.savez_compressed(output/f'd{d}_parameters.npz', **parameters)
    result = dict(d=d, max_N=max_n, final_N=d//2, model_checks=checks,
                  test_count=len(test), curves=records)
    (output/f'd{d}.json').write_text(json.dumps(result, indent=2))
    medians = {method: float(np.median([r['energy_error'][d//2] for r in records
                                      if r['method'] == method])) for method in METHODS}
    result['final_medians'] = medians
    print(json.dumps(dict(d=d, N=d//2, errors=medians, clipping_matches_saved=True)), flush=True)
    return result


def report(results, output, figure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':11,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    styles = [('clipping', 'Clipping', '#0072B2', 'o', '-'),
              ('active_face', 'Active face', '#C74432', 's', '--'),
              ('adaptive_normal', 'Adaptive normal', '#009E73', '^', '-.')]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), constrained_layout=True)
    for ax, d in zip(axes.flat, [4, 16, 64, 256]):
        result = next(r for r in results if r['d'] == d)
        for method, label, color, marker, ls in styles:
            curves = [r['energy_error'] for r in result['curves'] if r['method'] == method]
            median = np.median(curves, axis=0)
            ax.semilogy(np.arange(len(median)), median, color=color, ls=ls, marker=marker,
                        markevery=max(1, len(median)//9), ms=4, lw=1.6, label=label)
        ax.set(title=f'$d={d}$', xlabel='$N$', ylabel='Maximum test error')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(alpha=.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure)
    plt.close(fig)
    rows = []
    for r in results:
        errors = r['final_medians']; best = min(errors.values())
        values = [('$\\mathbf{%.5f}$' if value == best else '$%.5f$') % value
                  for value in errors.values()]
        rows.append(f"{r['d']} & {r['final_N']} & " + ' & '.join(values) + r'\\')
    (output/'table_rows.tex').write_text('\n'.join(rows)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/piecewise_boundary_updates_20261121')
    parser.add_argument('--figure', type=Path, default=ROOT/'paper/writeup/figs/piecewise_boundary_updates_error_vs_N.pdf')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [study(d, args.output) for d in [4, 16, 25, 64, 100, 144, 256]]
    (args.output/'summary.json').write_text(json.dumps(dict(
        settings=dict(T=80, alpha=.6, cutoff=1024, seeds=list(SEEDS), methods=METHODS,
                      device='CPU, float64, one BLAS thread',
                      test='saved shared 4096 interior + 4096 vertex parameters'),
        results=results), indent=2))
    report(results, args.output, args.figure)
