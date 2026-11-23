"""Small CPU screening, with saved data and explicit global error bounds."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .normalized_load import NormalizedLoad

OUT = Path(__file__).resolve().parents[3] / 'runs/cube_candidate_screen_20261104'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=80)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    prefix = 'normalized_load' if args.steps == 80 else f'normalized_load_T{args.steps}'
    OUT.mkdir(parents=True, exist_ok=True)
    settings = dict(dimensions=[16, 32, 64], seeds=list(range(1000, 1004)),
                    methods=['gd', 'pool256', 'pool4096', 'pool10000'],
                    alpha=.6, T=args.steps, N='d/2, including center snapshot',
                    norm='reference H1_0 energy; exact analytic solution',
                    normalizer='Frobenius norm of augmented residual Gram',
                    test='16384 scrambled Sobol points, coordinate axes, common union of spectral candidates and 64-start T160 searches on every final space',
                    initialization='same actual center snapshot for all methods',
                    device='CPU float64, one BLAS thread', width='not claimed')
    save(f'{prefix}_settings.json', settings)
    model = NormalizedLoad(7)
    rng = np.random.default_rng(7241)
    q = model.append(model.initial_basis(), rng.uniform(-1, 1, 7))
    h = model.matrix(q)
    z = rng.uniform(-.7, .7, 7)
    _, grad = model.objective(h, z, True)
    diff = np.array([(model.objective(h, z+1e-6*ei)-model.objective(h, z-1e-6*ei))/2e-6 for ei in np.eye(7)])
    validation = dict(gradient_relative_error=float(np.linalg.norm(diff-grad)/np.linalg.norm(grad)))
    assert validation['gradient_relative_error'] < 1e-7
    save(f'{prefix}_validation.json', validation)
    rows, curves = [], []
    with threadpool_limits(limits=1):
        for d in settings['dimensions']:
            model = NormalizedLoad(d)
            test = 2*qmc.Sobol(d, scramble=True, seed=202611401+d).random_base2(14)-1
            test = np.vstack((test, np.eye(d), -np.eye(d)))
            test_solutions = model.solution(test)
            starts = 2*qmc.Sobol(d, scramble=True, seed=202611402+d).random_base2(6)-1
            local_rows, local_bases, extra = [], [], []
            for seed in settings['seeds']:
                for method in settings['methods']:
                    rng = np.random.default_rng(seed+1000003)
                    q = model.initial_basis()
                    elapsed = 0.
                    start = time.perf_counter()
                    if method != 'gd':
                        pool = rng.uniform(-1, 1, (int(method[4:]), d))
                        pool_solutions = model.solution(pool)
                        pool_values = np.sum(pool_solutions[:, 1:]**2, axis=1)
                    elapsed += time.perf_counter()-start
                    selected = []
                    for n in range(1, d//2):
                        start = time.perf_counter()
                        if method == 'gd':
                            z, _ = model.search(q, rng.uniform(-1, 1, d), steps=args.steps)
                        else:
                            z = pool[int(np.argmax(pool_values))].copy()
                        q = model.append(q, z)
                        selected.append(z)
                        if method != 'gd':
                            pool_values = np.maximum(pool_values-(pool_solutions@q[:, -1])**2, 0.)
                        elapsed += time.perf_counter()-start
                        lower, upper, z_eig = model.error_bounds(q)
                        curves.append(dict(d=d, method=method, seed=seed, N=n+1,
                                           E_lower=lower, E_upper=upper))
                    lower, upper, z_eig = model.error_bounds(q)
                    e_test = float(np.linalg.norm(test_solutions-(test_solutions@q)@q.T, axis=1).max())
                    row = dict(d=d, method=method, seed=seed, N=d//2,
                               E_test=e_test, E_lower=lower, E_upper=upper,
                               seconds=elapsed, orthogonality=float(np.linalg.norm(q.T@q-np.eye(q.shape[1]))))
                    local_rows.append(row)
                    local_bases.append(q)
                    extra.append(z_eig[None, :])
                    extra.append(model.search(q, starts, steps=160)[0])
                    np.savez_compressed(OUT/f'{prefix}_d{d}_{method}_s{seed}.npz', Q=q, selected_parameters=selected)
            extra = np.vstack(extra)
            extra_solutions = model.solution(extra)
            np.savez_compressed(OUT/f'{prefix}_test_extra_d{d}.npz', parameters=extra)
            for row, q in zip(local_rows, local_bases):
                found = float(np.linalg.norm(extra_solutions-(extra_solutions@q)@q.T, axis=1).max())
                row['E_test_augmented'] = max(row['E_test'], found)
                assert row['E_test_augmented'] <= row['E_upper']*(1+1e-9)
                print(row, flush=True)
            rows.extend(local_rows)
            save(f'{prefix}_results.json', rows)
            save(f'{prefix}_curves.json', curves)
    print('Normalized load screening complete.', flush=True)


if __name__ == '__main__':
    main()
