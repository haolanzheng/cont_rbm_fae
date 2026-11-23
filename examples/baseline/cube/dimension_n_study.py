"""Fixed clipping search; error versus reduced dimension, with projection audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/cube_n_matplotlib')
import json
from pathlib import Path
import numpy as np
from .boundary_strategies import search
from ..geometry_cpu import CoordinateROM,rom_options
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/cube_fixed_clip_N_20261102';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'runs/analytic_hf_a06_t40_20261022'
TEST=ROOT/'runs/cube_slides_a06_t80_20261102'
def main():
 rows=[]
 for d in (6,8,16,32,48,64):
  a=np.load(SOURCE/f'model_d{d}.npz');G0=a['G0'];L=np.linalg.cholesky(G0);Z=np.load(TEST/f'test_vertices_d{d}.npz')['parameters'].T
  for seed in range(1000,1008):
   rom=CoordinateROM(G0,a['H0'],**rom_options(a));rng=np.random.default_rng(seed)
   for n in range(d+1):
    P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross)
    Pbest=np.eye(d)-rom.Q@(rom.Q.T@G0)
    error=np.linalg.norm(L.T@P@Z,axis=0)
    best_error=np.linalg.norm(L.T@Pbest@Z,axis=0)
    mismatch=np.linalg.norm(L.T@(P-Pbest)@Z,axis=0)
    rows.append(dict(d=d,N=n,seed=seed,error=float(error.max()),best_projection_error=float(best_error.max()),galerkin_mismatch=float(mismatch.max()),width=float(np.sqrt(sum(.78**(2*j) for j in range(n,d)))),orthogonality=rom.orthogonality_defect()))
    if n<d:
     H,_=rom.matrices();z,_=search(H,rng.uniform(-1,1,d),'clip',steps=80,alpha=.6);rom.append(z)
   np.savez_compressed(OUT/f'basis_d{d}_s{seed}.npz',Q=rom.Q)
   (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(d,seed,'done',flush=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(12,7))
 summary=[]
 for d,ax in zip((6,8,16,32,48,64),axs.flat):
  med=[];bp=[];mm=[];width=[];low=[];high=[]
  for n in range(d+1):
   r=[v for v in rows if v['d']==d and v['N']==n]
   vals=[v['error'] for v in r];med.append(float(np.median(vals)));low.append(min(vals));high.append(max(vals));bp.append(float(np.median([v['best_projection_error'] for v in r])));mm.append(float(np.median([v['galerkin_mismatch'] for v in r])));width.append(r[0]['width'])
   summary.append(dict(d=d,N=n,error=med[-1],best_projection_error=bp[-1],galerkin_mismatch=mm[-1],width=width[-1]))
  ax.semilogy(range(d+1),med,label='Galerkin test maximum');ax.fill_between(range(d+1),low,high,alpha=.15)
  ax.semilogy(range(d+1),bp,'--',label='Best projection in same space');ax.semilogy(range(d),width[:-1],':',label='Exact analytic width');ax.set(title=f'd = {d}',xlabel='N');ax.grid(alpha=.2)
 fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUT/'error_vs_N.png',dpi=180)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
