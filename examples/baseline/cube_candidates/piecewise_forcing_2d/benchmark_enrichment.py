"""Time complete greedy training from V_0 to V_{d-1}, without prefix reuse.

Pool candidates are precomputed in the compressed energy coordinates and
their errors updated by the existing exact rank-one formula. GD evaluates
the existing quadratic objective and gradient. Excludes shared spectral
model construction, evaluation on test points, validation, plotting and I/O.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_forcing_2d_matplotlib')
import csv
import json
import platform
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from dimension_study import rectangular_model, PARTITIONS, METHODS, SEEDS
from run import ascend, append, residual_gram

BASE = Path(__file__).resolve().parents[4] / 'runs/piecewise_forcing_2d_20261112/enrichment_study'
REPEATS = 9


def train(B, G, seed, method):
    cpu0 = process_time()
    t0 = perf_counter()
    d = B.shape[0]
    Q = np.empty((d, 0))
    selected = []
    if method == 'gd80':
        rng = np.random.default_rng(seed + 10000)
    else:
        M = int(method[4:])
        pool = np.random.default_rng(seed).uniform(-1, 1, (M, d))
        pool_u = pool @ B.T
        scores = np.sum(pool_u**2, axis=1)
    setup = perf_counter() - t0
    search = 0.
    enrichment = 0.
    for n in range(d - 1):
        ts = perf_counter()
        if method == 'gd80':
            H = residual_gram(B, G, Q)
            z = ascend(H, rng.uniform(-1, 1, d), 80)
        else:
            z = pool[np.argmax(scores)]
        search += perf_counter() - ts
        ts = perf_counter()
        Q, q = append(B, Q, z)
        selected.append(z)
        enrichment += perf_counter() - ts
        if method != 'gd80' and n < d - 2:
            ts = perf_counter()
            scores -= (pool_u @ q)**2
            search += perf_counter() - ts
    total = perf_counter() - t0
    cpu = process_time() - cpu0
    return Q, np.asarray(selected), dict(total_seconds=total, cpu_seconds=cpu,
                setup_seconds=setup, search_seconds=search, enrichment_seconds=enrichment)


def main():
    reports = []
    for mx, my in PARTITIONS:
        d = mx * my
        # Warm the shared model, then time its construction separately.
        rectangular_model(mx, my)
        model_times = []
        for _ in range(5):
            tick = perf_counter()
            B, G, _ = rectangular_model(mx, my)
            model_times.append(perf_counter() - tick)
        saved = np.load(BASE / f'd{d}_snapshots.npz')
        errors = json.loads((BASE / f'd{d}.json').read_text())
        validation = []
        # Warm every method/seed and validate training without counting checks.
        for method in METHODS:
            for seed in SEEDS:
                Q, Z, _ = train(B, G, seed, method)
                difference = float(np.max(np.abs(Z - saved[f'z_s{seed}_{method}'])))
                assert difference < 1e-12
                w = np.linalg.qr(Q, mode='complete')[0][:, -1]
                energy = float(np.sum(np.abs(B.T @ w)))
                reference = next(r for r in errors['curves'] if r['method'] == method and r['seed'] == seed)
                relative_error = abs(energy / reference['final_exact_energy_error'] - 1)
                assert relative_error < 1e-11
                validation.append(dict(method=method, seed=seed, max_parameter_difference=difference,
                                       final_error_relative_difference=relative_error))
        timings = []
        order_rng = np.random.default_rng(612000 + d)
        jobs = [(method, seed) for method in METHODS for seed in SEEDS]
        for repeat in range(REPEATS):
            for index in order_rng.permutation(len(jobs)):
                method, seed = jobs[index]
                _, _, measured = train(B, G, seed, method)
                timings.append(dict(method=method, seed=seed, repeat=repeat, **measured))
        summary = {}
        for method in METHODS:
            rows = [r for r in timings if r['method'] == method]
            s = {}
            for key in ['total_seconds', 'cpu_seconds', 'setup_seconds', 'search_seconds', 'enrichment_seconds']:
                values = np.array([r[key] for r in rows])
                s[key] = dict(median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                              q75=float(np.quantile(values, .75)), minimum=float(values.min()), maximum=float(values.max()))
            summary[method] = s
        report = dict(d=d, N=d-1, timings=timings, summary=summary, validation=validation,
                      shared_model_construction_seconds=model_times)
        reports.append(report)
        print(json.dumps(dict(d=d, N=d-1, median_ms={m: 1000*summary[m]['total_seconds']['median'] for m in METHODS},
                         shared_model_median_ms=1000*float(np.median(model_times)))), flush=True)
    try:
        from threadpoolctl import threadpool_info
        libraries = threadpool_info()
    except ImportError:
        libraries = []
    output = dict(settings=dict(platform=platform.platform(), machine=platform.machine(),
                   python=platform.python_version(), numpy=np.__version__, threadpools=libraries,
                   threads=1, dtype='float64', seeds=SEEDS, repeats=REPEATS,
                   T=80, alpha=.6, final_N='d-1',
                   included='Initialization and candidate sampling; pool precomputation; all greedy searches and score maintenance; snapshot construction and reorthogonalization.',
                   excluded='Shared B/G spectral model construction (separately timed), test errors, validation, plotting, I/O, interpreter/import startup.',
                   implementation='Same NumPy GD80 and efficient cached pool search as the accuracy experiment; pool-only work omits unused H assembly; final unused pool score update omitted.',
                   timing='Sequential randomized method/seed order; one untimed validated warmup per pair; 9 complete training repeats per seed; no saved-prefix reuse.'), reports=reports)
    (BASE / 'timing.json').write_text(json.dumps(output, indent=2) + '\n')
    with (BASE / 'timing.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d','N','method','median_ms','q25_ms','q75_ms','setup_median_ms','search_median_ms','enrichment_median_ms'])
        for r in reports:
            for m in METHODS:
                s=r['summary'][m]
                writer.writerow([r['d'],r['N'],m]+[1000*s['total_seconds'][k] for k in ['median','q25','q75']]
                    +[1000*s[k]['median'] for k in ['setup_seconds','search_seconds','enrichment_seconds']])
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
