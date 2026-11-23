"""Validate and time energy-dual searches with fixed-pool score reuse.

Uses the displayed models, candidates, starts, and update budgets. New output
folder preserves all historical measurements. Timed work includes sampling,
caching, score maintenance, search and enrichment; diagnostics run separately.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from .run_consistent_fem import train, coordinates, TABLE_DIMS, METHODS, GRIDS, SEEDS

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'runs/energy_residual_20261120'

def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    old=json.loads((BASE/'results.json').read_text());checks=[];raw=[];times=[]
    models={}
    with threadpool_limits(limits=1):
        for d in sorted({r['d'] for r in old}):
            models[d]=dict(np.load(BASE/f'model_d{d}.npz'))
        for row in old:
            if row['method'] not in METHODS+GRIDS:continue
            dom,d,m,s=(row[k] for k in ['domain','d','method','seed'])
            a=models[d];C=coordinates(a,dom);N=row['rank']
            rom,r,_,_=train(C,a['K'],dom,m,s,N,diagnostics=False)
            old_z=np.asarray(row['parameters']);new_z=np.asarray(r['parameters'])
            delta=float(np.max(np.abs(old_z-new_z))) if new_z.shape==old_z.shape else 999.
            Qold=np.load(BASE/f'basis_{dom}_d{d}_{m}_s{s}.npz')['Q']
            # Compare all saved reduced dimensions, including late tiny errors.
            targets=range(1,N+1) if delta>1e-12 else [N]
            projector_error=max(float(np.linalg.norm(rom.Q[:,:n]@rom.Q[:,:n].T-Qold[:,:n]@Qold[:,:n].T,2)) for n in targets)
            rec=dict(domain=dom,d=d,method=m,seed=s,N=N,rank=r['rank'],max_parameter_difference=delta,projector_difference=projector_error)
            if delta>1e-12:
                Z=np.load(BASE/f'test_{dom}_d{d}.npz')['parameters'];n=d//2
                E=C-rom.Q[:,:n]@(rom.Q[:,:n].T@C)
                err=float(np.sqrt(np.max(np.sum((Z@E.T)**2,axis=1))))
                rec.update(test_max_error=err,old_test_max_error=row['test_max_error'],relative_error_change=err/row['test_max_error']-1)
                print('DIFFERENCE',json.dumps(rec),flush=True)
            checks.append(rec)
            if r['rank']!=N or projector_error>1e-9:raise RuntimeError(rec)
        dump(out/'validation.json',checks)
        print('VALIDATED',len(checks),'histories; changed',sum(r['max_parameter_difference']>1e-12 for r in checks),flush=True)
        dump(out/'metadata.json',dict(platform=platform.platform(),numpy=np.__version__,threads=threadpool_info(),seeds=list(SEEDS),repeats=args.repeats,alpha=.6,T_sphere_ball=40,T_cube=80,
            source_results=str(BASE),included='Sampling/grid generation; pool energy-coordinate precomputation; rank-one score updates and stable refreshes; GD search-matrix construction; searches; basis enrichment.',
            excluded='Shared HF construction; error evaluation; probability diagnostics; verification; file I/O.',
            summary='Median of per-seed medians; randomized interleaved sequential runs after validation warmups.',
            sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['examples/baseline/common/cached_pool.py','examples/baseline/common/run_consistent_fem.py','examples/baseline/common/benchmark_cached_pool.py']}))
        if args.validate_only:return
        for d in sorted(set(TABLE_DIMS+[6])):
            a=models[d]
            jobs=[(dom,m,s,k) for dom in ['sphere','ball','cube'] for m in METHODS+(GRIDS if d<8 else []) for s in ([0] if m.startswith('grid') else SEEDS) for k in range(args.repeats)]
            np.random.default_rng(934000+d).shuffle(jobs)
            for dom,m,s,k in jobs:
                C=coordinates(a,dom)
                # Capture the complete training call, including pool setup and refreshes.
                tick=time.perf_counter();rom,r,_,_=train(C,a['K'],dom,m,s,d//2,diagnostics=False)
                wall=1000*(time.perf_counter()-tick)
                raw.append(dict(domain=dom,d=d,method=m,seed=s,repeat=k,wall_ms=wall,rank=r['rank']))
            for dom in ['sphere','ball','cube']:
                for m in METHODS+(GRIDS if d<8 else []):
                    rr=[r for r in raw if (r['d'],r['domain'],r['method'])==(d,dom,m)]
                    vals=[np.median([r['wall_ms'] for r in rr if r['seed']==s]) for s in sorted({r['seed'] for r in rr})]
                    times.append(dict(d=d,domain=dom,method=m,wall_ms=float(np.median(vals)),rank_min=min(r['rank'] for r in rr),rank_max=max(r['rank'] for r in rr)))
            dump(out/'timings.json',times);dump(out/'raw_timings.json',raw)
            print('TIMED',d,[(r['domain'],r['method'],round(r['wall_ms'],3)) for r in times if r['d']==d and r['method'] in ('gd','pool10000')],flush=True)

if __name__=='__main__':main()
