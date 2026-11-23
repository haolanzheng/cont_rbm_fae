import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from examples.baseline.cube_candidates.filtered_conductivity import FilteredConductivity
from examples.baseline.cube_candidates.run_filtered_conductivity import curves
P=Path(__file__).resolve().parents[3]/'runs/filtered_conductivity_20261111'

def main():
 rows=json.loads((P/'results_confirm.json').read_text());out=[]
 # Preserve original curves and test data; new results use one enriched shared set.
 (P/'results_before_enriched_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
 for d in [32,64,128]:
  tag=f'd{d}_c10_ell0.2';model=FilteredConductivity(d,10,.2,256)
  old=np.load(P/f'{tag}_test.npz')['parameters'];rng=np.random.default_rng(421)
  probes=old[rng.choice(len(old),1024,replace=False)]
  random=qmc.Sobol(d,scramble=True,seed=944).random_base2(3)
  found=[];local=[r for r in rows if r['d']==d]
  for row in local:
   data=np.load(P/f"{tag}_{row['method']}_s{row['seed']}.npz");q=data['Q']
   for n in [8,12,13,14,15,16,17,18,19,20,24,28,32]:
    qb=q[:,:n-1];vals=model.values(qb,probes,kind='relative',chunk=512)
    starts=np.vstack((random,probes[np.argsort(vals)[-4:]]))
    z,_=model.search(qb,starts,steps=160,kind='relative');found.append(z)
   print('enriched',d,row['method'],row['seed'],flush=True)
  common=np.vstack((old,*found));np.savez_compressed(P/f'{tag}_test_enriched.npz',parameters=common)
  for row in local:
   q=np.load(P/f"{tag}_{row['method']}_s{row['seed']}.npz")['Q']
   ab,rel=curves(model,q,common)
   assert np.all(rel>=np.array(row['E_relative'])-1e-11)
   row.update(E_test=ab.tolist(),E_relative=rel.tolist(),test_count=len(common));out.append(row)
   print('checked',d,row['method'],row['seed'],rel[15],rel[23],rel[31],flush=True)
  (P/'results_enriched.json').write_text(json.dumps(out,indent=2)+'\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
