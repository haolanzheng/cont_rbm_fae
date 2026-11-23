"""Extend pool and mesh comparisons to N=d on the fixed vertex test sets."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from ..geometry_cpu import CoordinateROM,rom_options
from .search import scores
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'runs/cube_fixed_clip_N_20261102';BASE=ROOT/'runs/cube_slides_a06_t80_20261102'
rows=json.loads((OUT/'comparison_results.json').read_text()) if (OUT/'comparison_results.json').exists() else []
stops=[]
done={(r['d'],r['method'],r['seed']) for r in rows}
for d in (6,16,32,64):
 a=np.load(ROOT/f'runs/analytic_hf_a06_t40_20261022/model_d{d}.npz');G0=a['G0'];L=np.linalg.cholesky(G0);Z=np.load(BASE/f'test_vertices_d{d}.npz')['parameters'].T
 for method in ['pool256','pool4096','pool10000']+(['grid1','grid0.5','grid0.25'] if d==6 else []):
  for seed in ([1000] if method.startswith('grid') else range(1000,1008)):
   if (d,method,seed) in done:continue
   rom=CoordinateROM(G0,a['H0'],**rom_options(a));Y=L.T@Z;crossZ=rom.cross@Z
   if method.startswith('pool'):pool=np.random.default_rng(seed+1000003).uniform(-1,1,(int(method[4:]),d))
   else:
    levels=np.linspace(-1,1,round(2/float(method[4:]))+1);pool=np.stack(np.meshgrid(*([levels]*d),indexing='ij'),axis=-1).reshape(-1,d)
   for n in range(d+1):
    rows.append(dict(d=d,N=n,seed=seed,method=method,error=float(np.linalg.norm(Y,axis=0).max())))
    if n==d:break
    H,_=rom.matrices();z=pool[np.argmax(scores(H,pool))].copy()
    try:rom.append(z)
    except RuntimeError:
     stops.append(dict(d=d,method=method,seed=seed,N=n,reason='selected snapshot does not enrich'));break
    q=rom.Q[:,-1];Y-=np.outer(L.T@q,q@crossZ)
   np.savez_compressed(OUT/f'basis_d{d}_{method}_s{seed}.npz',Q=rom.Q)
   (OUT/'comparison_results.json').write_text(json.dumps(rows,indent=2)+'\n');print(d,method,seed,flush=True)
(OUT/'comparison_stops.json').write_text(json.dumps(stops,indent=2)+'\n')
print('COMPLETE',flush=True)
