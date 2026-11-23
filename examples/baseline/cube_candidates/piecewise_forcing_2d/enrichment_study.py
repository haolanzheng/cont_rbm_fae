"""Continue the existing cube-source greedy paths to N=d-1.

Measure worst-error reductions separately on each seed's nested sequence.
All paths at a given d share one test set; final rank-one errors are exact
for the finite spectral model. Previously recorded prefixes are preserved.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_forcing_2d_matplotlib')
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dimension_study import rectangular_model, PARTITIONS, METHODS, SEEDS
from run import ascend, append, residual_gram, extrema

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'runs/piecewise_forcing_2d_20261112'
PREVIOUS = BASE / 'dimension_study'
OUT = BASE / 'enrichment_study'


def aggregate(values):
    values = np.asarray(values)
    return dict(median=np.median(values, axis=0).tolist(),
                minimum=np.min(values, axis=0).tolist(),
                maximum=np.max(values, axis=0).tolist())


def study(mx, my):
    tick = time.perf_counter()
    d = mx * my
    B, G, checks = rectangular_model(mx, my)
    normalization = float(np.sqrt(np.ones(d) @ G @ np.ones(d)))
    assert G.min() > 0
    saved = np.load(PREVIOUS / f'd{d}_snapshots.npz')
    spaces, snapshots, records, final_vectors = {}, {}, [], {}
    for seed in SEEDS:
        pool = np.random.default_rng(seed).uniform(-1, 1, (10000, d))
        for method in METHODS:
            init = np.random.default_rng(seed + 10000)
            Q = np.zeros((d, 0))
            prefix = saved[f'z_s{seed}_{method}']
            selected, rounds = [], []
            if method.startswith('pool'):
                candidates = pool[:int(method[4:])]
                U_pool = candidates @ B.T
                scores = np.sum(U_pool**2, axis=1)
            for n in range(d - 1):
                H = residual_gram(B, G, Q)
                # Consume the original stream even for the reused prefix.
                start = init.uniform(-1, 1, d) if method == 'gd80' else None
                if n < len(prefix):
                    z = prefix[n].copy()
                    if method.startswith('pool'):
                        assert np.max(np.abs(z - candidates[np.argmax(scores)])) < 1e-12
                elif method == 'gd80':
                    z = ascend(H, start, 80)
                else:
                    z = candidates[np.argmax(scores)].copy()
                J = float(z @ H @ z)
                assert J > 1e-20
                Q, q = append(B, Q, z)
                selected.append(z)
                rounds.append(dict(N=n + 1, J_selected=J, prefix_reused=n < len(prefix)))
                if method.startswith('pool'):
                    scores -= (U_pool @ q)**2
            assert np.max(np.abs(np.asarray(selected)[:len(prefix)] - prefix)) == 0
            spaces[(seed, method)] = Q
            snapshots[f'z_s{seed}_{method}'] = np.asarray(selected)
            # Complete orthogonal basis: one remaining solution-space direction.
            w = np.linalg.qr(Q, mode='complete')[0][:, -1]
            h = B.T @ w
            final_vectors[(seed, method)] = h
            records.append(dict(seed=seed, method=method, rounds=rounds))
        print(f'd={d}: seed {seed}, continued to N={d-1}', flush=True)
    np.savez_compressed(OUT / f'd{d}_snapshots.npz', **snapshots)

    # Keep all previously tested points and add searches for the extended ranks.
    tests = [np.load(PREVIOUS / f'd{d}_test.npz')['parameters']]
    rng = np.random.default_rng(172000 + d)
    if d > 16:
        for (seed, method), Q in spaces.items():
            for n in range(3 * d // 4 + 1, d):
                R = B - Q[:, :n] @ (Q[:, :n].T @ B)
                tests.append(extrema(R.T @ R, rng.uniform(-1, 1, (128, d)), steps=160))
    for h in final_vectors.values():
        corner = np.where(h >= 0, 1., -1.)
        tests.extend([corner[None, :], -corner[None, :]])
    test = np.unique(np.vstack(tests), axis=0)
    np.savez_compressed(OUT / f'd{d}_test.npz', parameters=test)
    U = test @ B.T
    unorm = np.sum(U**2, axis=1)
    rows = []
    checks['max_basis_orthogonality_defect'] = 0.
    checks['max_final_rank_one_vs_direct_relative_error'] = 0.
    checks['max_subtracted_vs_direct_gram_relative_error'] = 0.
    for (seed, method), Q in spaces.items():
        coeff = U @ Q
        squared = np.maximum(0., unorm[:, None] - np.cumsum(coeff**2, axis=1))
        curve = np.r_[1., np.sqrt(squared.max(axis=0)) / normalization]
        h = final_vectors[(seed, method)]
        final_exact = float(np.sum(np.abs(h)) / normalization)
        R = B - Q @ (Q.T @ B)
        direct = float(np.sqrt(np.sum((test @ R.T)**2, axis=1).max()) / normalization)
        mismatch = abs(final_exact - direct) / final_exact
        assert mismatch < 1e-10
        assert abs(curve[-1] - final_exact) / final_exact < 1e-10
        curve[-1] = final_exact
        assert np.all(np.diff(curve) <= 1e-12)
        gram_difference = np.linalg.norm(residual_gram(B, G, Q) - R.T @ R) / np.linalg.norm(R.T @ R)
        assert gram_difference < 1e-10
        checks['max_basis_orthogonality_defect'] = max(checks['max_basis_orthogonality_defect'], float(np.linalg.norm(Q.T @ Q - np.eye(d-1))))
        checks['max_final_rank_one_vs_direct_relative_error'] = max(checks['max_final_rank_one_vs_direct_relative_error'], mismatch)
        checks['max_subtracted_vs_direct_gram_relative_error'] = max(checks['max_subtracted_vs_direct_gram_relative_error'], float(gram_difference))
        drop = curve[:-1] - curve[1:]
        relative_drop = drop / curve[:-1]
        rows.append(dict(seed=seed, method=method, error_over_E0=curve.tolist(),
                         drop_over_E0=drop.tolist(), relative_drop=relative_drop.tolist(),
                         final_exact_energy_error=final_exact * normalization,
                         final_exact_error_over_E0=final_exact))
    summary = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        summary[method] = {key: aggregate([r[key] for r in selected])
                           for key in ['error_over_E0', 'drop_over_E0', 'relative_drop']}
    paired = {}
    for method in METHODS[1:]:
        comparisons = []
        for seed in SEEDS:
            a = next(r for r in rows if r['method'] == 'gd80' and r['seed'] == seed)
            b = next(r for r in rows if r['method'] == method and r['seed'] == seed)
            comparisons.append(dict(seed=seed,
                final_relative_improvement=1 - a['final_exact_error_over_E0'] / b['final_exact_error_over_E0'],
                rounds_with_larger_GD_absolute_drop=int(np.sum(np.array(a['drop_over_E0']) > b['drop_over_E0'])),
                rounds_with_lower_GD_error=int(np.sum(np.array(a['error_over_E0'][1:]) < b['error_over_E0'][1:]))))
        paired[method] = comparisons
    report = dict(settings=dict(d=d, partition=[mx, my], maximum_N=d-1,
                               alpha=.6, T=80, seeds=SEEDS, methods=METHODS,
                               initial_global_maximum=normalization, test_count=len(test),
                               all_ranks_global=(d == 16), final_rank_global=True,
                               test='Previous common test union, extended-rank vertex searches, and exact final-rank maximizing vertices; shared across all ranks, methods and seeds.',
                               device='CPU, float64, one BLAS thread',
                               interpretation='Each method follows its own greedy path; not a common-V_n one-step comparison.'),
                  checks=checks, training=records, curves=rows, summary=summary,
                  paired_comparisons=paired, elapsed_seconds=time.perf_counter()-tick)
    (OUT / f'd{d}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(d=d, tests=len(test), seconds=report['elapsed_seconds'],
                         final_medians={m: summary[m]['error_over_E0']['median'][-1] for m in METHODS})), flush=True)
    return report


def plots(reports):
    colors = ['#0072B2', '#555555', '#009E73', '#D55E00']
    styles = ['-', ':', '-.', '--']
    labels = ['GD, T=80', 'Pool 256', 'Pool 4096', 'Pool 10000']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    for name, key, ylabel in [
        ('error_vs_N', 'error_over_E0', r'$E_N/E_0$'),
        ('drop_per_basis', 'drop_over_E0', r'$(E_{N-1}-E_N)/E_0$'),
        ('relative_drop_per_basis', 'relative_drop', r'$(E_{N-1}-E_N)/E_{N-1}$ (%)')]:
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4))
        for ax, report in zip(axes.flat, reports):
            d = report['settings']['d']
            n = np.arange(d) if key == 'error_over_E0' else np.arange(1, d)
            for method, color, style, label in zip(METHODS, colors, styles, labels):
                values = np.asarray(report['summary'][method][key]['median'])
                if key == 'relative_drop':
                    values = 100 * values
                ax.plot(n, values, color=color, ls=style, label=label,
                        lw=2.2 if method == 'gd80' else 1.6)
            ax.set_title(f'd = {d}')
            ax.set_xlabel('Reduced dimension N')
            ax.set_ylabel(ylabel)
            ax.set_xlim(0 if key == 'error_over_E0' else 1, d-1)
            ax.set_xticks(np.unique(np.linspace(0 if key == 'error_over_E0' else 1, d-1, 5).round().astype(int)))
            ax.set_yscale('log')
            ax.grid(alpha=.16, which='both')
        axes[1, 2].axis('off')
        handles, texts = axes[0, 0].get_legend_handles_labels()
        axes[1, 2].legend(handles, texts, loc='upper left', frameon=False, fontsize=12)
        note = ('Median over 4 seeds\nEach method follows its own greedy path\n\nShared test set for each d\nGlobal maximum at all N: d=16\nGlobal maximum at N=d-1: all d')
        if key != 'error_over_E0':
            note += '\n\nDrops computed before seed aggregation'
        axes[1, 2].text(.04, .43, note, transform=axes[1, 2].transAxes,
                        ha='left', va='top', fontsize=10, linespacing=1.6)
        fig.tight_layout()
        fig.savefig(OUT / f'{name}.png', dpi=180)
        fig.savefig(OUT / f'{name}.pdf')
        plt.close(fig)
    with (OUT / 'final_errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'N', 'method', 'median_E_over_E0', 'min_E_over_E0', 'max_E_over_E0', 'median_absolute_E'])
        for r in reports:
            for method in METHODS:
                s = r['summary'][method]['error_over_E0']
                writer.writerow([r['settings']['d'], r['settings']['maximum_N'], method,
                    s['median'][-1], s['minimum'][-1], s['maximum'][-1],
                    s['median'][-1]*r['settings']['initial_global_maximum']])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    reports = [study(mx, my) for mx, my in PARTITIONS]
    plots(reports)
    (OUT / 'summary.json').write_text(json.dumps([
        {k: r[k] for k in ['settings', 'checks', 'summary', 'paired_comparisons']}
        for r in reports], indent=2) + '\n')
    print('COMPLETE', flush=True)
