"""Square-domain Poisson with independently controlled piecewise constant loads.

CPU/float64 screen. Uniform blocks, z in [-1,1]^(m*m), zero Dirichlet data.
The energy Gram is computed by a sine series with a rigorous analytic tail
bound (formulas evaluated in floating point, not interval arithmetic).
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np

DEFAULT = Path(__file__).resolve().parents[4] / 'runs/piecewise_forcing_2d_20261112'


def model(m, cutoff=1024):
    k = np.arange(1, cutoff + 1, dtype=float)
    r = np.arange(1, m + 1, dtype=float)
    x = (np.arange(m) + .5) / m
    Q = np.sin(np.pi * x[:, None] * r[None, :])
    Q /= np.linalg.norm(Q, axis=0)
    # Each integrated sine belongs to exactly one discrete sine direction.
    residue = np.arange(1, cutoff + 1) % (2 * m)
    alias = np.minimum(residue, 2 * m - residue)
    amp = 2 * np.sqrt(2) * np.sin(np.pi * k / (2 * m)) / (np.pi * k)
    mass = np.zeros((cutoff, m))
    valid = alias > 0
    norm_squared = np.where(alias == m, m, m / 2)
    mass[np.flatnonzero(valid), alias[valid] - 1] = amp[valid]**2 * norm_squared[valid]
    inverse_eigenvalue = 1 / (np.pi**2 * (k[:, None]**2 + k[None, :]**2))
    eig = mass.T @ inverse_eigenvalue @ mass
    captured = mass.sum(axis=0)
    tail = np.maximum(0., 1 / m**2 - captured[:, None] * captured[None, :])
    tail /= np.pi**2 * ((cutoff + 1)**2 + 1)
    relative_bound = float(np.max(tail / eig))
    U = np.kron(Q, Q)
    eig = eig.ravel()
    B = np.sqrt(eig)[:, None] * U.T
    G = B.T @ B
    return B, G, {'cutoff': cutoff, 'relative_gram_tail_bound': relative_bound,
                  'orthogonality_defect': float(np.linalg.norm(U.T @ U - np.eye(m*m))),
                  'constant_load_energy_squared': float(np.ones(m*m) @ G @ np.ones(m*m))}


def ascend(H, z, steps):
    """Projected ascent with one H@z per iterate, shared by score and gradient.

    The objective, step rule and best-iterate selection are unchanged. Scaling
    the vector H@z avoids forming 2*H on every update.
    """
    z = z.copy()
    ell = max(float(np.linalg.norm(H)), 1e-30)
    hz = H @ z
    best = z.copy()
    score = float(z @ hz)
    for _ in range(steps):
        g = 2 * hz / ell
        z = np.clip(z + .6 * g / np.sqrt(1 + g @ g), -1, 1)
        hz = H @ z
        value = float(z @ hz)
        if value > score:
            score, best = value, z.copy()
    return best


def append(B, Q, z):
    v = B @ z
    for _ in range(2):
        v -= Q @ (Q.T @ v)
    v /= np.linalg.norm(v)
    return np.column_stack((Q, v)), v


def residual_gram(B, G, Q):
    L = B.T @ Q
    H = G - L @ L.T
    return (H + H.T) / 2


def extrema(H, starts, steps=80):
    z = starts.copy()
    for _ in range(steps):
        new = np.where(z @ H >= 0, 1., -1.)
        if np.array_equal(new, z):
            break
        z = new
    return z


def run(m, seeds, cutoff, output, max_fraction=.5):
    start = time.perf_counter()
    d = m*m
    B, G, checks = model(m, cutoff)
    B2, G2, check2 = model(m, 2*cutoff)
    # Simultaneous diagonalization permits a direct eigenvalue comparison.
    checks['double_cutoff_max_relative_eigenvalue_change'] = float(np.max(
        np.abs(np.sum(B2*B2, axis=1) / np.sum(B*B, axis=1) - 1)))
    checks['double_cutoff_relative_tail_bound'] = check2['relative_gram_tail_bound']
    maximum_n = min(d-1, int(d*max_fraction))
    checkpoints = sorted(set([n for n in [4, 8, 16, 32, 48, 64, 96, 128, 192, 224]
                              if n <= maximum_n] + [maximum_n]))
    methods = ['gd80', 'pool256', 'pool4096', 'pool10000']
    spaces, parameters, training = {}, {}, []
    for seed in seeds:
        pool = np.random.default_rng(seed).uniform(-1, 1, (10000, d))
        for method in methods:
            t0 = time.perf_counter()
            Q = np.zeros((d, 0))
            init = np.random.default_rng(seed+10000)
            selected, records = [], []
            if method.startswith('pool'):
                candidates = pool[:int(method[4:])]
                values = candidates @ B.T
                scores = np.sum(values**2, axis=1)
            for n in range(max(checkpoints)):
                H = residual_gram(B, G, Q)
                if method.startswith('gd'):
                    z = ascend(H, init.uniform(-1, 1, d), int(method[2:]))
                else:
                    z = candidates[np.argmax(scores)]
                J = float(z @ H @ z)
                mean_gain = float(z @ H @ H @ z / (3*J))
                Q, q = append(B, Q, z)
                selected.append(z.copy())
                records.append({'N': n+1, 'J': J, 'mean_squared_error_drop': mean_gain})
                if method.startswith('pool'):
                    scores -= (values @ q)**2
                if n+1 in checkpoints:
                    key = (seed, method, n+1)
                    spaces[key] = Q.copy()
                    parameters[key] = np.asarray(selected).copy()
            training.append({'seed': seed, 'method': method,
                             'wall_seconds': time.perf_counter()-t0, 'rounds': records})
        print(json.dumps({'m': m, 'd': d, 'seed': seed, 'status': 'trained',
                          'elapsed': time.perf_counter()-start}), flush=True)
    rng = np.random.default_rng(71000+d)
    tests = [rng.uniform(-1, 1, (2048, d)), rng.choice([-1., 1.], (2048, d)),
             np.ones((1, d)), -np.ones((1, d))]
    if d <= 16:
        bits = (np.arange(2**d, dtype=np.uint32)[:, None] >> np.arange(d)) & 1
        tests.append(2*bits.astype(float)-1)
    for Q in spaces.values():
        H = residual_gram(B, G, Q)
        tests.append(extrema(H, rng.uniform(-1, 1, (64, d))))
    test = np.unique(np.vstack(tests), axis=0)
    test_u = test @ B.T
    normalization = np.sqrt(np.ones(d) @ G @ np.ones(d))
    rows = []
    for (seed, method, n), Q in spaces.items():
        residual = test_u - (test_u @ Q) @ Q.T
        squared = np.sum(residual**2, axis=1)
        H = residual_gram(B, G, Q)
        upper = np.sqrt(d * max(0., np.linalg.eigvalsh(H)[-1]))
        rows.append({'seed': seed, 'method': method, 'N': n,
                     'max_test_error': float(np.sqrt(squared.max())),
                     'max_test_error_normalized': float(np.sqrt(squared.max())/normalization),
                     'global_upper_normalized': float(upper/normalization),
                     'uniform_mean_squared_error': float(np.trace(H)/3),
                     'basis_orthogonality_defect': float(np.linalg.norm(Q.T@Q-np.eye(n)))})
    # All snapshot coordinates are retained for independent recomputation.
    np.savez_compressed(output/f'd{d}_snapshots.npz', **{
        f'z_s{s}_{method}_n{n}': z for (s, method, n), z in parameters.items()})
    result = {'settings': {'physical_domain': '(0,1)^2', 'm': m, 'd': d,
              'source': 'sum z_j indicator(cell_j)', 'parameter_domain': '[-1,1]^d',
              'seeds': seeds, 'alpha': .6, 'T': 80, 'maximum_N': maximum_n,
              'device': 'CPU, float64, one BLAS thread',
              'pools': 'fixed uniform, nested sizes 256/4096/10000',
              'test_count': len(test), 'error_normalization': 'energy norm of solution to f=1',
              'test': 'same union for every space: interior, vertices, independent vertex searches; all vertices for d16',
              'finite_test_is_global': d <= 16, 'elapsed_seconds': time.perf_counter()-start},
              'checks': checks, 'training': training, 'results': rows}
    (output/f'd{d}.json').write_text(json.dumps(result, indent=2))
    for n in checkpoints:
        print(json.dumps({'d': d, 'N': n, 'median_errors': {
            method: float(np.median([r['max_test_error_normalized'] for r in rows
                     if r['method']==method and r['N']==n])) for method in methods}}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--m', nargs='+', type=int, default=[4, 8, 16])
    p.add_argument('--cutoff', type=int, default=1024)
    p.add_argument('--max-fraction', type=float, default=.5)
    p.add_argument('--output', type=Path, default=DEFAULT)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [run(m, list(range(1000,1004)), args.cutoff, args.output, args.max_fraction) for m in args.m]
    (args.output/'summary.json').write_text(json.dumps([
        {'settings': a['settings'], 'checks': a['checks'], 'results': a['results']} for a in summaries], indent=2))
