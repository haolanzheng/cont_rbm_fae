"""Evaluate saved greedy spaces at N=d/2 and ceil(2d/3); no retraining."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
from run import model, residual_gram, extrema

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'runs/piecewise_forcing_2d_20261112/extended_N'
OUT = ROOT / 'runs/piecewise_forcing_2d_20261112/moderate_dimensions'
METHODS = ['gd80', 'pool256', 'pool4096', 'pool10000']


def evaluate(d):
    data = json.loads((SOURCE / f'd{d}.json').read_text())
    snapshots = np.load(SOURCE / f'd{d}_snapshots.npz')
    B, G, checks = model(int(np.sqrt(d)), data['checks']['cutoff'])
    original_spaces = {key: np.linalg.qr(B @ snapshots[key].T)[0]
                       for key in snapshots.files}
    ranks = [d // 2, (2 * d + 2) // 3]
    spaces = {}
    for seed in data['settings']['seeds']:
        for method in METHODS:
            prefix = f'z_s{seed}_{method}_n'
            key = max((k for k in snapshots.files if k.startswith(prefix)),
                      key=lambda k: int(k.rsplit('n', 1)[1]))
            z = snapshots[key]
            for n in ranks:
                Q = np.linalg.qr(B @ z[:n].T)[0]
                assert np.linalg.norm(Q.T @ Q - np.eye(n)) < 1e-10
                spaces[(seed, method, n)] = Q

    # Reconstruct the old shared test set before adding searches for new ranks.
    rng = np.random.default_rng(71000 + d)
    tests = [rng.uniform(-1, 1, (2048, d)), rng.choice([-1., 1.], (2048, d)),
             np.ones((1, d)), -np.ones((1, d))]
    if d == 16:
        bits = (np.arange(2**d, dtype=np.uint32)[:, None] >> np.arange(d)) & 1
        tests.append(2 * bits.astype(float) - 1)
    for Q in original_spaces.values():
        tests.append(extrema(residual_gram(B, G, Q), rng.uniform(-1, 1, (64, d))))
    old_test = np.unique(np.vstack(tests), axis=0)
    assert len(old_test) == data['settings']['test_count']
    initial_max_squared = float(np.ones(d) @ G @ np.ones(d))
    assert G.min() > 0  # Thus the constant-sign load attains the initial maximum.
    old_u = old_test @ B.T
    reproduction = []
    for (seed, method, n), Q in spaces.items():
        if n != d // 2:
            continue
        e = old_u - (old_u @ Q) @ Q.T
        value = float(np.sqrt(np.sum(e * e, axis=1).max() / initial_max_squared))
        old = next(r['max_test_error_normalized'] for r in data['results']
                   if r['seed'] == seed and r['method'] == method and r['N'] == n)
        reproduction.append(abs(value - old))
    assert max(reproduction) < 1e-10
    for Q in spaces.values():
        tests.append(extrema(residual_gram(B, G, Q), rng.uniform(-1, 1, (64, d))))
    test = np.unique(np.vstack(tests), axis=0)
    test_u = test @ B.T
    rows = []
    for (seed, method, n), Q in spaces.items():
        e = test_u - (test_u @ Q) @ Q.T
        H = residual_gram(B, G, Q)
        maximum = float(np.sqrt(np.sum(e * e, axis=1).max()))
        rows.append(dict(d=d, N=n, seed=seed, method=method,
                         maximum_test_energy_error=maximum,
                         maximum_test_error_over_initial_maximum=maximum / np.sqrt(initial_max_squared),
                         uniform_mean_square_error_ratio=float(np.trace(H) / np.trace(G)),
                         global_upper_over_initial_maximum=float(np.sqrt(d * np.linalg.eigvalsh(H)[-1] / initial_max_squared))))
    summary = []
    eigenvalues = np.linalg.eigvalsh(G)[::-1]
    approximation_lower_bounds = {str(n): float(np.sqrt(eigenvalues[n:].sum() / initial_max_squared))
                                  for n in ranks}
    for n in ranks:
        for method in METHODS:
            r = [x for x in rows if x['N'] == n and x['method'] == method]
            values = [x['maximum_test_error_over_initial_maximum'] for x in r]
            summary.append(dict(d=d, N=n, method=method, median=float(np.median(values)),
                                minimum=float(min(values)), maximum=float(max(values)),
                                median_absolute_error=float(np.median([x['maximum_test_energy_error'] for x in r])),
                                median_uniform_mean_square_error_ratio=float(np.median([x['uniform_mean_square_error_ratio'] for x in r]))))
    return dict(d=d, ranks=ranks, test_count=len(test), original_test_count=len(old_test),
                original_result_reproduction_max_error=max(reproduction),
                initial_maximum_energy_error=float(np.sqrt(initial_max_squared)),
                any_N_dimensional_space_lower_bound_over_initial_maximum=approximation_lower_bounds,
                global_maximum_computed=(d == 16), checks=checks, results=rows, summary=summary)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    results = [evaluate(d) for d in (16, 64)]
    output = dict(protocol='Saved four-seed GD80 and nested fixed-pool trajectories, alpha=.6. Prefix spaces at d/2 and ceil(2d/3). Shared original test set augmented by independent vertex searches for all new spaces. No retraining. Reported maxima are global for the truncated d16 model and tested lower bounds for d64.', dimensions=results)
    (OUT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k not in ('results', 'summary')}, indent=2))
        print(json.dumps(result['summary'], indent=2))
