"""Compare snapshot directions on fixed saved spaces using independent pools.

No greedy trajectory is retrained. GD80 and nested fresh random pools see
the same H_n. Probe seeds are independent of every saved training seed.
The scale-invariant trace decrease is compared with its best-direction
value lambda_max(H_n); this is an exact average-square-error diagnostic,
not a guarantee ordering the next cube worst error.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from run import model, ascend, append

BASE=Path(__file__).resolve().parents[4]/'runs/piecewise_forcing_2d_20261112'
OLD=BASE/'square_dimension_study'
OUT=BASE/'dimension_mechanism'


def metrics(H, eig, vectors, z, k):
    hz=H@z;J=float(z@hz);squared_norm=float(z@z)
    return dict(J=J,source_norm_squared=squared_norm/len(z),
        mean_gain_over_lambda_max=float(hz@hz/(J*eig[-1])),
        direction_Rayleigh_over_lambda_max=J/(squared_norm*eig[-1]),
        top_eigenspace_fraction=float(np.sum(eig[-k:]*(vectors[:,-k:].T@z)**2)/J))


def main():
    reports=[]
    for d in [25,64,144,256]:
        B,_,_=model(int(np.sqrt(d)))
        saved=np.load(OLD/f'd{d}_snapshots.npz')
        N=d//2;rows=[]
        for seed in [1000,1001,1002,1003]:
            # This pool was never used to build either saved reduced space.
            pool=np.random.default_rng(1700000+1000*d+seed).uniform(-1,1,(10000,d))
            pool_u=pool@B.T
            for anchor in ['gd80','pool10000']:
                Q=np.zeros((d,0))
                for n,z in enumerate(saved[f'z_s{seed}_{anchor}']):
                    if n in [N//2,N-1]:
                        R=B-Q@(Q.T@B);H=R.T@R
                        eig,vectors=np.linalg.eigh(H)
                        scores=np.sum(pool_u**2,axis=1)-np.sum((pool_u@Q)**2,axis=1)
                        candidates={}
                        for M in [256,4096,10000]:
                            candidates[f'pool{M}']=pool[np.argmax(scores[:M])]
                        start=np.random.default_rng(2700000+10000*d+100*seed+2*n+(anchor=='pool10000')).uniform(-1,1,d)
                        candidates['gd80']=ascend(H,start,80)
                        k=max(1,int(np.ceil((d-n)*.1)))
                        measured={name:metrics(H,eig,vectors,candidate,k) for name,candidate in candidates.items()}
                        hz=pool[:256]@H
                        typical=np.sum(hz**2,axis=1)/(np.sum(hz*pool[:256],axis=1)*eig[-1])
                        rows.append(dict(seed=seed,anchor=anchor,n=n,residual_rank=d-n,
                            effective_rank_trace_over_lambda_max=float(np.trace(H)/eig[-1]),
                            effective_rank_trace_squared_over_frobenius_squared=float(np.trace(H)**2/np.sum(H**2)),
                            top_eigenspace_dimension=k,
                            typical_random_mean_gain_over_lambda_max=float(np.mean(typical)),
                            candidates=measured))
                    Q,_=append(B,Q,z)
        summaries=[]
        for anchor in ['gd80','pool10000']:
            for n in [N//2,N-1]:
                subset=[r for r in rows if r['anchor']==anchor and r['n']==n]
                summaries.append(dict(anchor=anchor,n=n,
                    effective_rank=float(np.median([r['effective_rank_trace_over_lambda_max'] for r in subset])),
                    typical_random_mean_gain=float(np.median([r['typical_random_mean_gain_over_lambda_max'] for r in subset])),
                    candidates={name:{key:float(np.median([r['candidates'][name][key] for r in subset]))
                                     for key in subset[0]['candidates'][name]}
                                for name in ['gd80','pool256','pool4096','pool10000']},
                    gd_higher_mean_gain_than_pool10000=sum(r['candidates']['gd80']['mean_gain_over_lambda_max']>r['candidates']['pool10000']['mean_gain_over_lambda_max'] for r in subset)))
        report=dict(d=d,N=N,rows=rows,summary=summaries)
        reports.append(report)
        (OUT/f'fresh_pool_d{d}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(d=d,summary=[dict(anchor=s['anchor'],n=s['n'],effective_rank=s['effective_rank'],
             gd=s['candidates']['gd80']['mean_gain_over_lambda_max'],
             pool=s['candidates']['pool10000']['mean_gain_over_lambda_max'],
             typical=s['typical_random_mean_gain'],wins=s['gd_higher_mean_gain_than_pool10000']) for s in summaries])),flush=True)
    (OUT/'fresh_pool_summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    main()
