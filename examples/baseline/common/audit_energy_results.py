"""Verify saved energy-dual runs, common data, feasibility and objective identity."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import hashlib,json
import numpy as np
from .run_energy_residual import OUT,OLD
from .run_consistent_fem import DIMS,BOUNDARY_DIMS,CURVE_DIMS,METHODS,GRIDS,SEEDS

def main():
 rows=json.loads((OUT/'results.json').read_text());expected=set()
 for d in DIMS:
  for dom in ['sphere','ball','cube']:
   methods=METHODS+(GRIDS if d<8 else [])+(['active_face','normal_scaled'] if dom=='cube' and d in BOUNDARY_DIMS else [])
   expected.update((d,dom,m,s) for m in methods for s in ([0] if m.startswith('grid') else SEEDS))
 keys=[(r['d'],r['domain'],r['method'],r['seed']) for r in rows]
 assert len(keys)==len(set(keys)) and set(keys)==expected
 models={d:dict(np.load(OUT/f'model_d{d}.npz')) for d in DIMS}
 for d,a in models.items():
  with np.load(OLD/f'model_d{d}.npz') as old:
   for key in ['C','G0','rotation']:np.testing.assert_array_equal(a[key],old[key])
  np.testing.assert_array_equal(a['K'],np.eye(d))
  for dom in ['sphere','ball','cube']:
   name=f'test_{dom}_d{d}.npz'
   assert hashlib.sha256((OUT/name).read_bytes()).digest()==hashlib.sha256((OLD/name).read_bytes()).digest()
 discrepancies=[]
 for r in rows:
  d,dom,m,s=r['d'],r['domain'],r['method'],r['seed'];a=models[d]
  C=a['C']@a['rotation'] if d<8 and dom!='cube' else a['C']
  saved=np.load(OUT/f'basis_{dom}_d{d}_{m}_s{s}.npz');Q=saved['Q'];z=saved['selected'];H=saved['final_search_H']
  assert r['stop'] is None and r['rank']>=d//2
  P=Q[:,:d//2-1];E=C-P@(P.T@C);G=E.T@E
  discrepancies.append(float(np.linalg.norm(H-G)/max(np.linalg.norm(G),1e-300)))
  Z=np.array(r['parameters'])
  if dom=='sphere':assert np.max(abs(np.linalg.norm(Z,axis=1)-1))<1e-12
  elif dom=='ball':assert np.max(np.linalg.norm(Z,axis=1))<=1+1e-12
  else:assert np.max(abs(Z))<=1+1e-12
 assert max(discrepancies)<1e-12
 result=dict(runs=len(rows),dimensions=DIMS,unchanged_models_and_test_sets=True,early_stops=0,
  max_objective_error_relative=max(discrepancies),max_orthogonality_defect=max(r['orthogonality_defect'] for r in rows))
 (OUT/'final_audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
