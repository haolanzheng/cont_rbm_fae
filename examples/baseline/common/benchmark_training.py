"""Audit-free CPU timings with selected-parameter equivalence checks.

Run from the repository root:
 .venv_cpu/bin/python -m examples.baseline.common.benchmark_training --output runs/...
Previous logs remain unchanged. Matrix assembly, candidate generation, search,
and enrichment are timed; diagnostics and equivalence checks are outside timing.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from ..geometry_cpu import CoordinateROM, rom_options, search, pool_search, uniform_parameters
from ..cube.search import search as cube_search
from ..run_extended_comparison_cpu import tensor_candidates

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / 'runs/analytic_hf_a06_t40_20261022'
DIMS = [4, 6, 8, 16, 32, 48, 64]
SEEDS = list(range(1000, 1008))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sample(rng, d, domain, count=1):
    return rng.uniform(-1, 1, (count, d)) if domain == 'cube' else uniform_parameters(rng, count, d, domain)


def train(G0, H0, opts, domain, method, seed, legacy=None):
    start = time.perf_counter()
    rom = (legacy.CoordinateROM if legacy is not None else CoordinateROM)(G0, H0, **opts)
    rng = np.random.default_rng(seed)
    d = len(G0)
    pool = None
    if method.startswith('pool'):
        pool = sample(np.random.default_rng(seed + 1000003), d, domain, int(method[4:]))
    elif method.startswith('grid'):
        h = float(method[4:])
        if domain == 'cube':
            axis = np.linspace(-1, 1, round(2 / h) + 1)
            pool = np.stack(np.meshgrid(*([axis] * d), indexing='ij'), axis=-1).reshape(-1, d)
        else:
            pool = tensor_candidates(d, h, domain)[0]
    params = []
    for _ in range(d // 2):
        H, _ = rom.matrices() if legacy is not None else rom.matrices(include_error=False)
        if method == 'gd':
            initial = sample(rng, d, domain)[0]
            if domain == 'cube':
                fn = legacy.cube_search if legacy is not None else cube_search
                kwargs = {} if legacy is not None else {'diagnostics': False}
                z, _ = fn(H, initial, alpha=.6, T=80, **kwargs)
            else:
                fn = legacy.search if legacy is not None else search
                kwargs = {} if legacy is not None else {'diagnostics': False}
                z, _ = fn(H, initial, alpha=.6, steps=40, domain=domain,
                          omega=1., ball_step_rule='normalized', **kwargs)
        else:
            z, _ = pool_search(H, pool)
        rom.append(z)
        params.append(z)
    elapsed = 1000 * (time.perf_counter() - start)
    return elapsed, np.asarray(params)


def load_legacy(folder):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    geometry = load('previous_geometry', folder / 'geometry_cpu.py')
    geometry.cube_search = load('previous_cube', folder / 'cube/search.py').search
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-source', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'timings.json').exists():
        raise FileExistsError('Preserve existing timing logs; use a fresh output directory.')
    old = load_legacy(args.previous_source)
    high = json.loads((MODEL / 'results.json').read_text())['runs']
    low = json.loads((MODEL / 'low_dimension_results.json').read_text())['runs']
    cube = json.loads((ROOT / 'runs/cube_slides_a06_t80_20261102/results.json').read_text())['runs']
    def archived(domain, d, method, seed):
        if domain == 'cube':
            row = next(r for r in cube if (r['d'], r['method'], r['seed']) == (d, method, seed))
        else:
            key = method if d < 8 else (('exp_rgd40' if domain == 'sphere' else 'normalized_pg40') if method == 'gd' else 'fixed_' + method)
            row = next(r for r in (low if d < 8 else high)
                       if (r['domain'], r['dimension'], r['method'], r['seed']) == (domain, d, key, seed)
                       and (d >= 8 or r['orientation'] == 'rotated'))
        return np.array([r['parameter'] for r in row['rounds'][:d // 2]])
    rows = []; checks = []; raw = []; paired = []
    with threadpool_limits(limits=1):
        metadata = {'platform': platform.platform(), 'threads': threadpool_info(),
                    'dimensions': DIMS, 'seeds': SEEDS, 'repeats': args.repeats,
                    'alpha': .6, 'sphere_ball_steps': 40, 'cube_steps': 80,
                    'diagnostics_timed': False, 'dtype': 'float64',
                    'scope': 'Candidate generation, search matrix formation, search, and enrichment through N=d/2; no test errors or diagnostics.',
                    'sources': {f: hashlib.sha256((ROOT/'examples/baseline'/f).read_bytes()).hexdigest()
                                for f in ['geometry_cpu.py', 'cube/search.py', 'common/benchmark_training.py']}}
        assert all(t['num_threads'] == 1 for t in metadata['threads'])
        dump(out/'metadata.json', metadata)
        shuffle = np.random.default_rng(271828)
        for d in DIMS:
            a = np.load(MODEL / f'model_d{d}.npz')
            for domain in ['sphere', 'ball', 'cube']:
                rotated = domain != 'cube' and d < 8
                G0 = a['rotated_G0'] if rotated else a['G0']
                H0 = a['rotated_H0'] if rotated else a['H0']
                opts = {k: a['rotated_'+k] for k in rom_options(a)} if rotated else rom_options(a)
                methods = ['gd', 'pool256', 'pool4096', 'pool10000']
                if d < 8:
                    methods += ['grid1', 'grid0.5', 'grid0.25']
                # Validate whole greedy histories, not only one isolated step.
                for method in methods:
                    seeds = ([1000] if domain == 'cube' else [0]) if method.startswith('grid') else SEEDS
                    for seed in seeds:
                        _, z = train(G0, H0, opts, domain, method, seed)
                        ref = archived(domain, d, method, seed)
                        err = float(np.max(np.abs(z-ref)))
                        np.testing.assert_allclose(z, ref, atol=1e-9, rtol=1e-9)
                        if method == 'gd':
                            _, old_z = train(G0,H0,opts,domain,method,seed,legacy=old)
                            np.testing.assert_array_equal(z, old_z)
                        checks.append(dict(d=d, domain=domain, method=method, seed=seed, max_parameter_difference=err))
                # Warm validation runs above precede randomized, interleaved timing.
                jobs = [(m, seed, repeat) for m in methods
                        for seed in (([1000] if domain=='cube' else [0]) if m.startswith('grid') else SEEDS)
                        for repeat in range(5 if m.startswith('grid') else args.repeats)]
                jobs += [('legacy_gd', seed, repeat) for seed in SEEDS for repeat in range(args.repeats)]
                shuffle.shuffle(jobs)
                group = []
                for method, seed, repeat in jobs:
                    elapsed, _ = train(G0,H0,opts,domain,'gd' if method=='legacy_gd' else method,
                                       seed,legacy=old if method=='legacy_gd' else None)
                    record = dict(domain=domain,d=d,N=d//2,method=method,seed=seed,repeat=repeat,wall_ms=elapsed)
                    group.append(record);raw.append(record)
                for method in methods + ['legacy_gd']:
                    entries = [r for r in group if r['method']==method]
                    per_seed = [float(np.median([r['wall_ms'] for r in entries if r['seed']==seed]))
                                for seed in sorted({r['seed'] for r in entries})]
                    row = dict(domain=domain,d=d,N=d//2,method=method,wall_ms_median=float(np.median(per_seed)))
                    (paired if method=='legacy_gd' else rows).append(row)
                dump(out/'raw_timings.json', raw);dump(out/'validation.json', checks)
                dump(out/'timings_partial.json', rows);dump(out/'legacy_gd_timings.json', paired)
                new_ms = next(r['wall_ms_median'] for r in rows if (r['d'],r['domain'],r['method'])==(d,domain,'gd'))
                old_ms = paired[-1]['wall_ms_median']
                print(f'{domain} d={d}: GD {old_ms:.3f} -> {new_ms:.3f} ms ({old_ms/new_ms:.2f}x); histories verified',flush=True)
        dump(out/'timings.json', rows)
        print(f'Completed: {len(checks)} histories checked; {len(raw)} timed runs.', flush=True)

if __name__ == '__main__':
    main()
