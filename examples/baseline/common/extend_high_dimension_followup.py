"""Repeat high-dimensional timings and match the existing test-set convention."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .extend_high_dimensions import train,evaluate,dump
from ..geometry_cpu import CoordinateROM,rom_options,uniform_parameters
from ..cube.boundary_strategies import search as boundary_search
from ..cube.test_set import make_vertex_test_set
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'runs/high_dimension_tables_20261113'

def main():
    rows=json.loads((OUT/'results.json').read_text())['runs'];raw=[];summary=[]
    with threadpool_limits(limits=1):
        # Match the archived sphere/ball test-seed convention.
        for d in [128,256]:
            a=dict(np.load(OUT/f'model_d{d}.npz'))
            for domain in ['sphere','ball']:
                Z=uniform_parameters(np.random.default_rng(202610180+100*d+(domain=='ball')),16384,d,domain)
                np.savez_compressed(OUT/f'test_{domain}_d{d}.npz',parameters=Z)
                for r in [r for r in rows if (r['d'],r['domain'])==(d,domain)]:
                    rom=CoordinateROM(a['G0'],a['H0'],**rom_options(a));rom.Q=np.load(OUT/f"basis_{domain}_d{d}_{r['method']}_s{r['seed']}.npz")['Q']
                    r['test_max_error'],r['exact_domain_error']=evaluate(a,rom,Z,domain)
            dump(OUT/'results.json',dict(runs=rows))
            print('Test convention matched',d,flush=True)
            for domain in ['sphere','ball','cube']:
                jobs=[(m,s,k) for m in ['gd','pool256','pool4096','pool10000'] for s in range(1000,1008) for k in range(3)]
                np.random.default_rng(819+d).shuffle(jobs)
                for m,seed,repeat in jobs:
                    ms,rom,params,stop=train(a,domain,m,seed)
                    ref=next(r for r in rows if (r['domain'],r['d'],r['method'],r['seed'])==(domain,d,m,seed))
                    np.testing.assert_array_equal(params,ref['parameters'])
                    raw.append(dict(domain=domain,d=d,method=m,seed=seed,repeat=repeat,wall_ms=ms,rank=rom.Q.shape[1],reached_target=stop is None))
                for m in ['gd','pool256','pool4096','pool10000']:
                    rr=[r for r in raw if (r['domain'],r['d'],r['method'])==(domain,d,m)]
                    per_seed=[np.median([r['wall_ms'] for r in rr if r['seed']==s]) for s in range(1000,1008)]
                    summary.append(dict(domain=domain,d=d,method=m,wall_ms_median=float(np.median(per_seed)),all_reached_target=all(r['reached_target'] for r in rr),rank_min=min(r['rank'] for r in rr),rank_max=max(r['rank'] for r in rr)))
                dump(OUT/'raw_timings.json',raw);dump(OUT/'timings.json',summary)
                print('Repeated timings complete',domain,d,flush=True)
        cfg=json.loads((OUT/'config.json').read_text());cfg['sphere_ball_test_seed']='202610180+100*d+(domain==ball)';cfg['timing_repetitions']=3;dump(OUT/'config.json',cfg)
    print('All follow-up runs complete',flush=True)

if __name__=='__main__':main()
