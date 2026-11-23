"""Decompose fixed-round baseline search time on one CPU thread.

Run with the repository's CPU environment:
  .venv_cpu/bin/python -m examples.baseline.common.benchmark_search_cost
Uses saved FEM residual matrices; does not change training or slide artifacts.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json
import platform
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from ..geometry_cpu import search,pool_search,sphere_exp,uniform_parameters,sym

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/search_cost_breakdown_20261113'


def profile_sphere(H,initial,T=40):
    """Same arithmetic as geometry_cpu.search, with coarse block timers."""
    parts=np.zeros(4)
    start=time.perf_counter_ns()
    H=sym(H);scale=float(np.linalg.norm(H,'fro'))
    points=np.atleast_2d(np.asarray(initial,dtype=np.float64)).copy()
    points/=np.linalg.norm(points,axis=1)[:,None]
    best=points.copy();best_values=np.full(len(points),-np.inf)
    parts[0]=time.perf_counter_ns()-start
    for t in range(T+1):
        start=time.perf_counter_ns()
        hz=points@H.T
        values=np.sum(points*hz,axis=1)
        parts[1]+=time.perf_counter_ns()-start
        start=time.perf_counter_ns()
        improved=values>best_values
        best[improved]=points[improved]
        best_values=np.maximum(best_values,values)
        parts[2]+=time.perf_counter_ns()-start
        if t==T:break
        start=time.perf_counter_ns()
        derivative=2.0*(hz-values[:,None]*points)/scale
        tangent=.6*derivative/np.sqrt(1.0+np.sum(derivative**2,axis=1,keepdims=True))
        points=sphere_exp(points,tangent)
        parts[3]+=time.perf_counter_ns()-start
    return best[0],parts


def benchmark(jobs,repeats=7,target=.02):
    counts={};measurements={name:[] for name in jobs}
    for name,f in jobs.items():
        for _ in range(5):f()
        n=1
        while True:
            t=time.perf_counter()
            for _ in range(n):f()
            elapsed=time.perf_counter()-t
            if elapsed>=.003 or n>=16384:break
            n*=2
        counts[name]=max(1,min(32768,int(target*n/max(elapsed,1e-9))))
    rng=np.random.default_rng(1311)
    for repeat in range(repeats):
        for name in rng.permutation(list(jobs)):
            f=jobs[name];n=counts[name];t=time.perf_counter_ns()
            for _ in range(n):f()
            measurements[name].append((time.perf_counter_ns()-t)/n/1e6)
    return {k:dict(median_ms=float(np.median(v)),min_ms=min(v),max_ms=max(v),
                   repetitions_per_sample=counts[k],samples_ms=v) for k,v in measurements.items()}


def main():
    OUT.mkdir(exist_ok=True)
    records=[]
    with threadpool_limits(limits=1):
        metadata=dict(platform=platform.platform(),numpy=np.__version__,blas=threadpool_info(),
            dimensions=[16,32,64,128],T=40,alpha=.6,pool_sizes=[256,4096,10000],
            domain='sphere',fixed_space='saved GD seed1000, n=d/2-1',
            timing='warm repeated single-thread search; excludes H construction and sampling',
            profile_note='Separate instrumented runs; timer overhead may perturb microsecond blocks.')
        for d in metadata['dimensions']:
            source=ROOT/f'runs/consistent_fem_20261113/basis_sphere_d{d}_gd_s1000.npz'
            H=np.ascontiguousarray(np.load(source)['final_search_H'])
            rng=np.random.default_rng(1311+d)
            initial=uniform_parameters(rng,1,d,'sphere')[0]
            z=initial[None,:]
            pools={M:uniform_parameters(rng,M,d,'sphere') for M in metadata['pool_sizes']}
            jobs={'single_matvec':lambda:z@H.T,
                  'single_value':lambda:np.sum(z*(z@H.T),axis=1),
                  'gd_full':lambda:search(H,initial,steps=40,domain='sphere',alpha=.6,diagnostics=False)}
            for M,Z in pools.items():
                Q=Z@H.T
                jobs[f'pool{M}_matmul']=lambda Z=Z:Z@H.T
                jobs[f'pool{M}_postprocess']=lambda Z=Z,Q=Q:np.argmax(np.sum(Z*Q,axis=1))
                jobs[f'pool{M}_full']=lambda Z=Z:pool_search(H,Z)
            result=benchmark(jobs)
            expected=search(H,initial,steps=40,domain='sphere',alpha=.6,diagnostics=False)[0]
            measured=[]
            for _ in range(100):
                actual,p=profile_sphere(H,initial)
                np.testing.assert_array_equal(actual,expected)
                measured.append(p)
            profile=np.median(measured,axis=0)/1e6
            record=dict(d=d,source=str(source),benchmarks=result,
                profiled_median_ms=dict(zip(['setup','objective_values','best_selection','gradient_and_update'],profile)),
                profile_matches_original=True)
            records.append(record)
            (OUT/'results.json').write_text(json.dumps(dict(metadata=metadata,records=records),indent=2))
            print(d,'GD ms',result['gd_full']['median_ms'],
                  'pool ms',[result[f'pool{M}_full']['median_ms'] for M in pools],
                  'profile ms',profile.tolist(),flush=True)
    print('Saved',OUT/'results.json',flush=True)

if __name__=='__main__':main()
