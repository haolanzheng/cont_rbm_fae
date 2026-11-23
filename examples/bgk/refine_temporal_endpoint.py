"""Check snapshot accuracy at N25, where late snapshot directions are small."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,concurrent.futures
import numpy as np
from examples.bgk.bgk_temporal import OUT,context
from examples.bgk.bgk_decisive import dump,evaluate
from examples.bgk.bgk_strict import StrictROM
from examples.bgk.boltzmann_bgk_screen import BGKModel,Config,relative_l2

def run(name):
 x=json.loads((OUT/f'{name}.json').read_text());sn=np.load(OUT/f'{name}.npz');mus=sn['mus'][:25]
 m=BGKModel(Config(fom_tol=1e-13,inner_maxiter=320));ss=[];gaps=[]
 for i,mu in enumerate(mus):
  f,info=m.solve(*mu);assert info['converged'];ss.append(f);gaps.append(relative_l2(m,f,sn['snapshots'][i]))
 par,truth,_=context();rom=StrictROM(m,ss)
 diag=evaluate(rom,par[:256],truth[:256],{'typical':[0,256]})
 idx=int(np.argmax(diag['errors']));f,info=m.solve(*par[idx]);assert info['converged']
 strict=StrictROM(m,ss,solver='scipy',tol=1e-13);ans=strict.solve_inner(*par[idx])
 checked=relative_l2(m,strict.field(ans['y']),f)
 row=dict(name=name,rank=25,max_snapshot_change=max(gaps),original_max=x['diagnostics'][-1]['typical']['max'],
   refined_snapshots_max=diag['typical']['max'],fully_refined_worst_point_error=checked,
   max_error_ratio=diag['typical']['max']/x['diagnostics'][-1]['typical']['max'],
   truth_change_at_worst=relative_l2(m,f,truth[idx]),argmax=par[idx].tolist())
 dump(OUT/f'endpoint_refined_{name}.json',row);return row

if __name__=='__main__':
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  rows=list(ex.map(run,['new_adam','new_pool']))
 dump(OUT/'BGK_TEMPORAL_ENDPOINT_PRECISION.json',rows)
 print(json.dumps(rows,indent=2))
