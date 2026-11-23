"""Nested-grid audit of fixed reduced spaces on shared witness parameters."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .model import ThermalBlock
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/thermal_block_cube_20261103'
def prolongation(cells):
 fine=2*cells;rr=[];cc=[];vv=[]
 for i in range(1,fine):
  for j in range(1,fine):
   x=i/2;y=j/2;ix=min(int(x),cells-1);iy=min(int(y),cells-1);a=x-ix;b=y-iy
   nodes=[(ix,iy,1-a),(ix+1,iy,a-b),(ix+1,iy+1,b)] if a>=b else [(ix,iy,1-b),(ix+1,iy+1,a),(ix,iy+1,b-a)]
   for u,v,w in nodes:
    if 0<u<cells and 0<v<cells and w:rr.append((i-1)*(fine-1)+j-1);cc.append((u-1)*(cells-1)+v-1);vv.append(w)
 return sparse.coo_matrix((vv,(rr,cc)),shape=((fine-1)**2,(cells-1)**2)).tocsc()
def main():
 records=json.loads((OUT/'results.json').read_text());test=np.load(OUT/'test_d14.npz');Z=test['Z'];U=test['U'];groups=test['groups'];rng=np.random.default_rng(123)
 ids=set(rng.choice(np.flatnonzero(groups==0),64,replace=False).tolist()+rng.choice(np.flatnonzero(groups==3),64,replace=False).tolist())
 ids.update(r['direct_error_check'][0]['index'] for r in records if r['d']==14);ids=np.array(sorted(ids))
 coarse=ThermalBlock(2,7,28);fine=ThermalBlock(2,7,56,dense_factor=False);P=prolongation(28)
 stiffness=float(sparse.linalg.norm(P.T@fine.X@P-coarse.X)/sparse.linalg.norm(coarse.X));assert stiffness<1e-12
 methods=['gd','pool4096','pool10000','mixed4096'];V={m:P@np.load(OUT/f'basis_d14_{m}_s1000.npz')['V'] for m in methods}
 data=[]
 for i in ids:
  A=fine.matrix(Z[i]);uf=spsolve(A,fine.f);diff=uf-P@U[:,i];hf_relative=np.sqrt(diff@(fine.X@diff)/(uf@(fine.X@uf)));entry=dict(index=int(i),hf_relative_difference=float(hf_relative))
  for m,v in V.items():
   y=np.linalg.solve(v.T@(A@v),v.T@fine.f);e=uf-v@y;fineerror=np.sqrt(max(0,e@(fine.X@e)))
   vc=np.load(OUT/f'basis_d14_{m}_s1000.npz')['V'];ac=coarse.matrix(Z[i]);yc=np.linalg.solve(vc.T@(ac@vc),vc.T@coarse.f);ec=U[:,i]-vc@yc;ce=np.sqrt(max(0,ec@(coarse.X@ec)))
   entry[m]=dict(coarse=float(ce),fine=float(fineerror))
  data.append(entry)
 summary=dict(cells=[28,56],dofs=[coarse.n,fine.n],count=len(ids),scope='fixed lifted coarse spaces, seed1000; shared witness set, not a full retraining',nested_stiffness_relative_error=stiffness,hf_relative_median=float(np.median([r['hf_relative_difference'] for r in data])),hf_relative_max=float(max(r['hf_relative_difference'] for r in data)),methods={m:dict(coarse_max=max(r[m]['coarse'] for r in data),fine_max=max(r[m]['fine'] for r in data)) for m in methods})
 (OUT/'refinement.json').write_text(json.dumps(dict(summary=summary,records=data),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
