"""Full greedy retraining with matched starts and larger per-search GD budgets.

Test errors are evaluated only after training; no budget is selected by them.
Existing T=80 and pool results are retained as references, without rerunning.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
from .model import ThermalBlock
from .run import train, dump
from .correlation_prior import train as prior_train
from .contrast_correlation import maps

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/thermal_block_cube_20261103'
PREV=ROOT/'runs/thermal_block_contrast_correlation_20261103'
OUT=ROOT/'runs/thermal_block_step_budget_20261103'
SEEDS=list(range(1000,1008))

def cases():
    result=maps()
    centers=np.array([[(j+.5)/7,(i+.5)/2] for i in range(2) for j in range(7)])
    dist=np.sum((centers[:,None]-centers[None,:])**2,axis=2)
    for length in [0.,.15,.35]:
        K=np.eye(14) if length==0 else np.exp(-dist/(2*length**2))
        v,w=np.linalg.eigh(K);L=(w*np.sqrt(np.maximum(v,0)))@w.T
        result.append(dict(name=f'prior_{length:g}',group='prior',length=length,
                           center=50.5,jacobian=49.5*np.eye(14),L=L))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',nargs='+',type=int,default=[160,320,640]);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    orig=np.load(BASE/'test_d14.npz');Z=orig['Z'];groups=orig['groups']
    existing=json.loads((PREV/'results.json').read_text());prior=json.loads((PREV/'prior_results.json').read_text())
    references=[dict(r,T=80) for r in existing]
    references.extend(dict(r,case=f"prior_{r['length']:g}",T=80) for r in prior)
    dump(OUT/'references.json',references)
    configs=cases()
    dump(OUT/'config.json',dict(steps=[80]+args.steps,seeds=SEEDS,d=14,N=24,cells=28,alpha=.6,threads=1,device='CPU',test_size=len(Z),test_sha256=hashlib.sha256(Z.tobytes()).hexdigest(),cases=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in configs],objective='log squared reference-energy dual residual',comparison='full greedy retraining; same per-round random starts; existing pool results unchanged',timing='training only, excluding tests and common assembly',stopping='at most T iterations; original displacement and Armijo early stopping retained'))
    records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    done={(r['case'],r['T'],r['seed']) for r in records}
    pbase=ThermalBlock(2,7,28)
    for case in configs:
        p=copy.copy(pbase);p.coefficient_center=np.full(14,case['center']);p.coefficient_jacobian=case['jacobian']
        if case['name']=='contrast_10': data=orig
        else: data=np.load(PREV/('test_contrast_100.npz' if case['group']=='prior' else f"test_{case['name']}.npz"))
        U=data['U'];norm2=data['norm2']
        jobs=[(T,s) for T in args.steps for s in SEEDS];np.random.default_rng(401).shuffle(jobs)
        for T,seed in jobs:
            if (case['name'],T,seed) in done:continue
            if case['group']=='prior':rb,seconds,rounds=prior_train(p,'gd',seed,case['L'],steps=T)
            else:rb,seconds,rounds=train(p,'gd',seed,24,steps=T)
            rank=rb.n
            np.savez_compressed(OUT/f"basis_{case['name']}_T{T}_s{seed}.npz",V=rb.V[:,:rank])
            e=rb.errors(Z,U,norm2)
            curve=[dict(N=rank,max_error=float(e.max()),max_relative_error=float((e/np.sqrt(norm2)).max()),interior_max=float(e[groups==0].max()),boundary_max=float(e[(groups>0)&(groups<4)].max()),worst_index=int(np.argmax(e)))]
            records.append(dict(case=case['name'],method='gd',d=14,T=T,seed=seed,rank=rank,training_seconds=seconds,rounds=rounds,curve=curve))
            dump(OUT/'results.json',records)
            print('done',case['name'],T,seed,'error',curve[0]['max_error'],'seconds',round(seconds,3),flush=True)
    print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
