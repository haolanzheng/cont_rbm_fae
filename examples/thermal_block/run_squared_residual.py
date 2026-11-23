"""Rerun the existing ten settings with squared loss and baseline updates."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import copy,hashlib,json
from pathlib import Path
import numpy as np
from .model import ThermalBlock
from .run import dump
from .step_budget import cases,BASE,PREV,SEEDS
from .squared_residual import train,validate
ROOT=Path(__file__).resolve().parents[2]
NORMALIZATION=os.environ.get('THERMAL_NORMALIZATION','gram')
OUTPUT_NAMES={'gram':'thermal_block_squared_gram_20261103','source':'thermal_block_squared_residual_20261103','none':'thermal_block_squared_unscaled_20261103'}
OUT=ROOT/'runs'/OUTPUT_NAMES[NORMALIZATION]
NORMALIZER_DESCRIPTIONS={'gram':'ell_n=||B_n^T X^-1 B_n||_F, B_n=[f,-A_1 v_1,...,-A_d v_n], fixed during each search','source':'fixed source dual norm squared','none':'ell_n=1; no objective normalization'}
STEPS=[80,160,320,640]

def main():
    OUT.mkdir(exist_ok=True)
    orig=np.load(BASE/'test_d14.npz');Z=orig['Z'];groups=orig['groups']
    old=ROOT/'runs/thermal_block_step_budget_20261103'
    references=json.loads((old/'references.json').read_text())
    dump(OUT/'pool_references.json',[r for r in references if r['method']!='gd'])
    dump(OUT/'log_references.json',[r for r in references if r['method']=='gd']+json.loads((old/'results.json').read_text()))
    cfg=cases()
    dump(OUT/'config.json',dict(d=14,N=24,cells=28,seeds=SEEDS,steps=STEPS,alpha=.6,device='CPU',threads=1,objective='r^T X^-1 r / ell_n',normalizer=NORMALIZER_DESCRIPTIONS[NORMALIZATION],update='z_next=clip(z+alpha*gradient/sqrt(1+gradient@gradient),-1,1)',search='exactly T steps, best visited point, no logarithm or Armijo backtracking',test_size=len(Z),test_sha256=hashlib.sha256(Z.tobytes()).hexdigest(),timing='complete training excluding common assembly and test evaluation',cases=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in cfg],pool_reuse='monotonic transformation of scores preserves candidate ordering; previous pools retained'))
    records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    done={(r['case'],r['T'],r['seed']) for r in records};pbase=ThermalBlock(2,7,28)
    for case in cfg:
        p=copy.copy(pbase);p.coefficient_center=np.full(14,case['center']);p.coefficient_jacobian=case['jacobian']
        dump(OUT/f"validation_{case['name']}.json",validate(p,NORMALIZATION))
        data=orig if case['name']=='contrast_10' else np.load(PREV/('test_contrast_100.npz' if case['group']=='prior' else f"test_{case['name']}.npz"))
        U=data['U'];norm2=data['norm2'];jobs=[(T,s) for T in STEPS for s in SEEDS];np.random.default_rng(401).shuffle(jobs)
        for T,seed in jobs:
            if (case['name'],T,seed) in done:continue
            rb,seconds,rounds=train(p,seed,T,L=case.get('L'),normalization=NORMALIZATION);rank=rb.n
            np.savez_compressed(OUT/f"basis_{case['name']}_T{T}_s{seed}.npz",V=rb.V[:,:rank])
            e=rb.errors(Z,U,norm2)
            curve=[dict(N=rank,max_error=float(e.max()),max_relative_error=float((e/np.sqrt(norm2)).max()),interior_max=float(e[groups==0].max()),boundary_max=float(e[(groups>0)&(groups<4)].max()),worst_index=int(np.argmax(e)))]
            records.append(dict(case=case['name'],method='gd',d=14,T=T,seed=seed,rank=rank,training_seconds=seconds,rounds=rounds,curve=curve))
            dump(OUT/'results.json',records)
            print('done',case['name'],T,seed,'rank',rank,'error',curve[0]['max_error'],'seconds',round(seconds,3),flush=True)
    print('COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
