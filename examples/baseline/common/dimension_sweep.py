"""Dense dimension sweep at N=d/2; same CPU searches, final test maximum only.

Existing five dimensions are reused. Intermediate models are leading principal
blocks of the same nested 64-mode analytic family (no change to PDE or grid).
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from ..geometry_cpu import CoordinateROM,rom_options,uniform_parameters,search
from ..cube.search import search as cube_search,scores
from ..cube.test_set import make_vertex_test_set

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 dims=[12,20,24,28,36,40,44,52,56,60];seeds=list(range(1000,1008));base=np.load(args.source/'model_d64.npz');records=[]
 path=out/'results.json'
 if path.exists():records=json.loads(path.read_text())['runs']
 existing={(r['domain'],r['d'],r['method'],r['seed']) for r in records}
 meta={'dimensions':dims,'seeds':seeds,'alpha':.6,'updates':{'sphere':40,'ball':40,'cube':80},'rho':.78,'pool_sizes':[256,4096,10000],'N':'d/2','dtype':'float64','threads':1,'model_source':str(args.source/'model_d64.npz'),'model_sha256':hashlib.sha256((args.source/'model_d64.npz').read_bytes()).hexdigest(),'metric':'maximum absolute A-energy error on fixed test set / exact analytic width'}
 (out/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
 for d in dims:
  N=d//2;model={k:base[k][:d,:d] for k in ['G0','H0','cross','residual_error_cross','residual_error_gram']};np.savez_compressed(out/f'model_d{d}.npz',**model)
  for domain in ['sphere','ball','cube']:
   T=meta['updates'][domain];Z=(make_vertex_test_set(d,65536,202610270)[0] if domain=='cube' else uniform_parameters(np.random.default_rng(202610270+d),16384,d,domain));np.savez_compressed(out/f'test_{domain}_d{d}.npz',parameters=Z)
   width=float(np.sqrt(sum(.78**(2*j) for j in range(N,d)))) if domain=='cube' else .78**N
   for method in ['gd','pool256','pool4096','pool10000']:
    for seed in seeds:
     if (domain,d,method,seed) in existing:continue
     rng=np.random.default_rng(seed);rom=CoordinateROM(model['G0'],model['H0'],**rom_options(model));rounds=[]
     begin=time.perf_counter();pool=None
     if method!='gd':
      prng=np.random.default_rng(seed+1000003);m=int(method[4:]);pool=prng.uniform(-1,1,(m,d)) if domain=='cube' else uniform_parameters(prng,m,d,domain)
     elapsed=time.perf_counter()-begin
     for n in range(N):
      begin=time.perf_counter();H,_=rom.matrices()
      if method=='gd':
       z0=rng.uniform(-1,1,d) if domain=='cube' else uniform_parameters(rng,1,d,domain)[0]
       z,_=cube_search(H,z0,alpha=.6,T=T) if domain=='cube' else search(H,z0,steps=T,domain=domain,alpha=.6,ball_step_rule='normalized')
      else:z=pool[np.argmax(scores(H,pool))]
      rom.append(z);elapsed+=time.perf_counter()-begin;rounds.append(z.tolist())
     _,G=rom.matrices();err=float(np.sqrt(max(0,scores(G,Z).max())))
     rec={'domain':domain,'d':d,'N':N,'method':method,'seed':seed,'test_max_error':err,'exact_width':width,'ratio':err/width,'test_count':len(Z),'wall_ms':elapsed*1000,'rounds':rounds,'orthogonality_defect':rom.orthogonality_defect()};records.append(rec)
     np.savez_compressed(out/f'basis_{domain}_d{d}_{method}_s{seed}.npz',Q=rom.Q)
    path.write_text(json.dumps({'config':meta,'runs':records},indent=2)+'\n')
   print('Complete',d,domain,flush=True)
if __name__=='__main__':main()
