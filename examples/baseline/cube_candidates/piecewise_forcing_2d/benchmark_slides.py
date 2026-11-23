"""Full greedy-training timing at N=floor(d/2), matching slide experiments.

Model construction, error tests, validation and file I/O are outside timers.
Pool scores use cached energy coordinates and exact rank-one updates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json, itertools, platform, argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from equal_time_pool import train
from run import model, append

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'runs/piecewise_forcing_2d_20261112'
OUT=BASE/'slides_summary'
SEEDS=range(1000,1004)
METHODS=['gd80','pool256','pool4096','pool10000']
REPEATS=5

def run(B,G,seed,method):
    if not method.startswith('grid'):
        return train(B,G,seed,0 if method=='gd80' else int(method[4:]))
    tick=perf_counter();d=len(B);h=float(method[4:]);N=d//2
    points=np.array(list(itertools.product(np.linspace(-1,1,round(2/h)+1),repeat=d)))
    U=points@B.T;scores=np.sum(U**2,axis=1);Q=np.zeros((d,0));selected=[]
    for n in range(N):
        z=points[np.argmax(scores)];Q,q=append(B,Q,z);selected.append(z.copy())
        if n<N-1:scores-=(U@q)**2
    return Q,np.array(selected),dict(wall_seconds=perf_counter()-tick)

def main(output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'timing.json').exists():
        raise FileExistsError('Preserve previous timing logs; choose a new --output directory.')
    reports=[];table=[]
    for d in [4,16,25,64,100,144,256]:
        B,G,_=model(round(np.sqrt(d)));N=d//2
        methods=METHODS+(['grid1','grid0.5','grid0.25'] if d==4 else [])
        jobs=[(method,seed) for method in methods for seed in (SEEDS if not method.startswith('grid') else [1000])]
        if d==4:
            saved=json.loads((OUT/'d4.json').read_text())
        else:
            saved=np.load(BASE/('dimension_study' if d==16 else 'square_dimension_study')/f'd{d}_snapshots.npz')
        validation=[]
        for method,seed in jobs:
            Q,Z,_=run(B,G,seed,method)
            ref=np.array(next(r['parameters'] for r in saved['curves'] if r['seed']==seed and r['method']==method))[:N] if d==4 else saved[f'z_s{seed}_{method}'][:N]
            if not method.startswith('grid'):
                diff=float(np.max(np.abs(Z-ref)));assert diff<1e-10,(d,method,seed,diff)
                validation.append(dict(method=method,seed=seed,max_parameter_difference=diff))
            else:
                Qref=np.zeros((d,0))
                for z in ref: Qref,_=append(B,Qref,z)
                vertices=np.array(list(itertools.product([-1.,1.],repeat=d)))
                e=np.linalg.norm(vertices@(B-Q@(Q.T@B)).T,axis=1).max()
                er=np.linalg.norm(vertices@(B-Qref@(Qref.T@B)).T,axis=1).max()
                assert abs(e-er)<1e-12
                validation.append(dict(method=method,seed=seed,max_error_difference=float(abs(e-er))))
        rng=np.random.default_rng(934000+d);timings=[]
        for repeat in range(REPEATS):
            for index in rng.permutation(len(jobs)):
                method,seed=jobs[index];_,_,t=run(B,G,seed,method)
                timings.append(dict(method=method,seed=seed,repeat=repeat,wall_seconds=t['wall_seconds']))
        med={m:float(np.median([r['wall_seconds'] for r in timings if r['method']==m])) for m in methods}
        reports.append(dict(d=d,N=N,timings=timings,median_seconds=med,validation=validation))
        table.append(f'{d} & {N} & '+' & '.join(f'${1000*med[m]:.2f}$' if m in med else '---' for m in METHODS+['grid1','grid0.5','grid0.25'])+r'\\')
        print(json.dumps(dict(d=d,median_ms={m:1000*v for m,v in med.items()})),flush=True)
    try:
        from threadpoolctl import threadpool_info
        libraries=threadpool_info()
    except ImportError: libraries=[]
    (output/'timing.json').write_text(json.dumps(dict(settings=dict(platform=platform.platform(),numpy=np.__version__,threadpools=libraries,repeats=REPEATS,seeds=list(SEEDS),T=80,alpha=.6,N='floor(d/2)',
        included='Sampling/grid construction, pool precomputation, search, score maintenance and basis enrichment.',
        excluded='Shared spectral model construction, test evaluation, validation, plotting, I/O.',
        order='Randomized sequential runs after validated warmup; single BLAS thread.'),reports=reports),indent=2)+'\n')
    (output/'timing_rows.tex').write_text('\n'.join(table)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    main(parser.parse_args().output)
