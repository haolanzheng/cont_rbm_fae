"""Exhaustive vertex maximization at every greedy round, through full rank."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/cube_n_matplotlib')
import json,csv
from pathlib import Path
import numpy as np
from ..geometry_cpu import CoordinateROM,rom_options
from .test_set import make_vertex_test_set
from .search import scores
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/cube_exact_greedy_N_20261102';OUT.mkdir(exist_ok=True)
rows=[];steps=[]
for d in (6,8,16):
 a=np.load(ROOT/f'runs/analytic_hf_a06_t40_20261022/model_d{d}.npz');L=np.linalg.cholesky(a['G0']);Z,exhaustive=make_vertex_test_set(d);assert exhaustive
 for method in ('exact_residual','exact_error'):
  rom=CoordinateROM(a['G0'],a['H0'],**rom_options(a))
  for n in range(d+1):
   H,_=rom.matrices();P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross);Pbest=np.eye(d)-rom.Q@(rom.Q.T@rom.G0)
   errors=np.linalg.norm(L.T@P@Z.T,axis=0);best=np.linalg.norm(L.T@Pbest@Z.T,axis=0)
   width=float(np.sqrt(sum(.78**(2*j) for j in range(n,d))))
   rows.append(dict(d=d,N=n,method=method,error=float(errors.max()),best_projection_error=float(best.max()),width=width,ratio=float(errors.max()/width) if width else None,orthogonality=rom.orthogonality_defect()))
   if n==d:break
   values=scores(H,Z) if method=='exact_residual' else errors**2
   k=int(np.argmax(values));z=Z[k].copy();novelty=rom.append(z)
   steps.append(dict(d=d,n=n,method=method,parameter=z.tolist(),objective_max=float(values[k]),energy_error_at_parameter=float(errors[k]),max_energy_error=float(errors.max()),energy_quality=float(errors[k]/errors.max()),snapshot_novelty=novelty,hessian_min_eigenvalue=float(np.linalg.eigvalsh(H)[0])))
  np.savez_compressed(OUT/f'basis_d{d}_{method}.npz',Q=rom.Q)
  print(d,method,'done',flush=True)
(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(OUT/'selected_parameters.json').write_text(json.dumps(steps,indent=2)+'\n')
with (OUT/'summary.csv').open('w') as fp:
 w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
old=json.loads((ROOT/'runs/cube_fixed_clip_N_20261102/results.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for d,ax in zip((6,8,16),axs):
 for method,label,style in [('exact_residual','Exact residual maximization','-'),('exact_error','Exact error maximization','--')]:
  r=[v for v in rows if v['d']==d and v['method']==method];ax.semilogy([v['N'] for v in r],[v['error'] for v in r],style,lw=1.7,label=label)
 ax.semilogy(range(d+1),[np.median([v['error'] for v in old if v['d']==d and v['N']==n]) for n in range(d+1)],':',lw=2,label='GD (eight-seed median)')
 ax.semilogy(range(d),[np.sqrt(sum(.78**(2*j) for j in range(n,d))) for n in range(d)],'-.',color='#555555',label='Exact analytic width')
 ax.set(xlabel='N',title=f'd = {d}',ylim=(2e-7,4));ax.grid(alpha=.2)
axs[0].set_ylabel('Maximum A-norm error')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9);fig.tight_layout(rect=(0,.13,1,1));fig.savefig(OUT/'exact_greedy_error_vs_N.png',dpi=180)
print('COMPLETE',flush=True)
