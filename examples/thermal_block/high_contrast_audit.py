"""Check inner iteration sensitivity and one nested-mesh high-contrast audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .model import ThermalBlock,RB
from .run import gd,dump
from .refinement import prolongation
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/thermal_block_contrast_correlation_20261103';BASE=ROOT/'runs/thermal_block_cube_20261103'
def main():
 budget=[]
 for C in (100,300):
  p=ThermalBlock(2,7,28);p.coefficient_center=np.full(14,(1+C)/2);p.coefficient_jacobian=(C-1)/2*np.eye(14)
  V=np.load(OUT/f'basis_contrast_{C}_gd_s1000.npz')['V'];rb=RB(p,16)
  for j in range(16):rb.add(V[:,j])
  starts=np.random.default_rng(537).uniform(-1,1,(8,14))
  for i,z in enumerate(starts):
   results={}
   for T in (80,320):
    _,stats=gd(rb,z,steps=T);results[T]=stats
   budget.append(dict(contrast=C,start=i,ratio_indicator_320_over_80=float(np.exp((results[320]['log_residual']-results[80]['log_residual'])/2)),runs=results))
 dump(OUT/'inner_budget_audit.json',budget)
 records=json.loads((OUT/'results.json').read_text());test=np.load(BASE/'test_d14.npz');Z=test['Z'];groups=test['groups'];U=np.load(OUT/'test_contrast_100.npz')['U'];rng=np.random.default_rng(123)
 ids=set(rng.choice(np.flatnonzero(groups==0),64,replace=False).tolist()+rng.choice(np.flatnonzero(groups==3),64,replace=False).tolist())
 ids.update(r['curve'][-1]['worst_index'] for r in records if r['case']=='contrast_100');ids=np.array(sorted(ids))
 coarse=ThermalBlock(2,7,28);fine=ThermalBlock(2,7,56,dense_factor=False)
 for p in (coarse,fine):p.coefficient_center=np.full(14,50.5);p.coefficient_jacobian=49.5*np.eye(14)
 P=prolongation(28);assert sparse.linalg.norm(P.T@fine.X@P-coarse.X)/sparse.linalg.norm(coarse.X)<1e-12
 methods=['gd','pool4096','pool10000','mixed4096'];Vc={m:np.load(OUT/f'basis_contrast_100_{m}_s1000.npz')['V'] for m in methods};Vf={m:P@v for m,v in Vc.items()};data=[]
 for i in ids:
  A=fine.matrix(Z[i]);Ac=coarse.matrix(Z[i]);uf=spsolve(A,fine.f);diff=uf-P@U[:,i];entry=dict(index=int(i),hf_relative_difference=float(np.sqrt(diff@(fine.X@diff)/(uf@(fine.X@uf)))))
  for m in methods:
   vf=Vf[m];vc=Vc[m];y=np.linalg.solve(vf.T@(A@vf),vf.T@fine.f);e=uf-vf@y;yf=np.linalg.solve(vc.T@(Ac@vc),vc.T@coarse.f);ec=U[:,i]-vc@yf
   entry[m]=dict(coarse=float(np.sqrt(max(0,ec@(coarse.X@ec)))),fine=float(np.sqrt(max(0,e@(fine.X@e)))))
  data.append(entry)
 summary=dict(count=len(ids),cells=[28,56],scope='fixed coarse spaces, lifted and reprojected; seed1000 only; not a full mesh convergence study',hf_relative_median=float(np.median([r['hf_relative_difference'] for r in data])),hf_relative_max=max(r['hf_relative_difference'] for r in data),methods={m:dict(coarse_max=max(r[m]['coarse'] for r in data),fine_max=max(r[m]['fine'] for r in data)) for m in methods})
 dump(OUT/'refinement_contrast_100.json',dict(summary=summary,records=data));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
