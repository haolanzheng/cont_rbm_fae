"""Training timings, repeated-point checks and probability data for the FEM rerun."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time
import numpy as np
from scipy.special import betaincc
from .run_consistent_fem import OUT,model,coordinates,train,dump,TABLE_DIMS,METHODS,GRIDS,SEEDS
from ..consistent_fem import EnergyROM
from .box_bounds import maximum


def main():
 rows=json.loads((OUT/'results.json').read_text());timings=[];raw=[];audits=[]
 for d in TABLE_DIMS:
  a=model(d)
  jobs=[(dom,m,s,k) for dom in ['sphere','ball','cube'] for m in METHODS+(GRIDS if d==4 else []) for s in ([0] if m.startswith('grid') else SEEDS) for k in range(3)]
  np.random.default_rng(819+d).shuffle(jobs)
  for dom,m,s,k in jobs:
   C=coordinates(a,dom);rom,r,_,_=train(C,a['K'],dom,m,s,d//2)
   ref=next(x for x in rows if (x['d'],x['domain'],x['method'],x['seed'])==(d,dom,m,s))
   np.testing.assert_array_equal(r['parameters'],ref['parameters'][:d//2])
   raw.append(dict(d=d,domain=dom,method=m,seed=s,repeat=k,wall_ms=r['elapsed_ms'][-1],rank=r['rank']))
  for dom in ['sphere','ball','cube']:
   for m in METHODS+(GRIDS if d==4 else []):
    rr=[r for r in raw if (r['d'],r['domain'],r['method'])==(d,dom,m)]
    vals=[np.median([r['wall_ms'] for r in rr if r['seed']==s]) for s in sorted({r['seed'] for r in rr})]
    timings.append(dict(d=d,domain=dom,method=m,wall_ms=float(np.median(vals)),rank_min=min(r['rank'] for r in rr),rank_max=max(r['rank'] for r in rr)))
  dump('timings.json',timings);dump('raw_timings.json',raw);print('TIMINGS',d,flush=True)
 # Directly check that saved selected snapshots have vanishing residual.
 for r in rows:
  if r['d']!=128:continue
  a=model(128);C=coordinates(a,r['domain']);rom=EnergyROM(C,a['K'])
  rom.Q=np.load(OUT/f"basis_{r['domain']}_d128_{r['method']}_s{r['seed']}.npz")['Q']
  E=rom.error_factor();H=rom.matrix();z=np.array(r['parameters']);v=z@E.T
  selected=np.einsum('ij,ij->i',v@a['K'],v)
  audits.append(dict(domain=r['domain'],method=r['method'],seed=r['seed'],rank=r['rank'],selected_residual_max=float(np.sqrt(max(selected.max(),0))),remaining_residual_max=float(np.sqrt(np.linalg.eigvalsh(H)[-1])),test_max_error=r['test_max_error']))
 dump('selected_residual_checks.json',audits)
 theory=[]
 for d in [8,16,32,48,64]:
  a=model(d);rom=EnergyROM(a['C'],a['K'])
  for n in range(d//2):rom.append(np.linalg.eigh(rom.matrix())[1][:,-1])
  H=rom.matrix();e=np.linalg.eigvalsh(H);sigma=e[-1]/np.linalg.norm(H);ratio=e[-2]/e[-1]
  np.savez_compressed(OUT/f'reference_d{d}_n{d//2}.npz',H=H,Q=rom.Q)
  for dom in ['sphere','ball']:
   b=(d-1)/2 if dom=='sphere' else (d+1)/2
   def growth(alpha):
    if dom=='sphere':return 1+2*alpha*sigma*(1-.9**2)/np.sqrt(1+sigma**2)*(1-alpha**2*sigma**2/(6*(1+sigma**2))-alpha*.9**2*sigma)
    eta=alpha/np.sqrt(1+4*.9**2*sigma**2)
    return (1+2*eta*sigma)/np.sqrt(1+4*.9**2*eta*sigma*(1+eta*sigma))
   alpha=np.linspace(0,1,301);T=np.arange(101)
   p=1. if ratio>=.9**2 else float(betaincc(.5,b,(.9**2-ratio)/(1-ratio)))
   theory.append(dict(domain=dom,d=d,sigma=float(sigma),lambda2_ratio=float(ratio),alpha=alpha.tolist(),step_lower=betaincc(.5,b,.9**2*growth(alpha)**(-80)).tolist(),T=T.tolist(),time_lower=betaincc(.5,b,.9**2*growth(.6)**(-2*T)).tolist(),lower=float(betaincc(.5,b,.9**2*growth(.6)**(-80))),pool_upper={str(M):float(-np.expm1(M*np.log1p(-p))) if p<1 else 1. for M in [256,4096,10000]}))
 dump('probability_theory.json',theory)
 empirical=[]
 for dom in ['sphere','ball','cube']:
  for d in [8,16,32,48,64]:
   for m in METHODS:
    for r in [r for r in rows if (r['domain'],r['d'],r['method'])==(dom,d,m)]:
     a=np.load(OUT/f"basis_{dom}_d{d}_{m}_s{r['seed']}.npz");H=a['final_search_H'];z=a['selected'];H=H/np.linalg.norm(H)
     lo,hi=maximum(H) if dom=='cube' else (np.linalg.eigvalsh(H)[-1],)*2
     value=float(z@H@z)
     empirical.append(dict(domain=dom,d=d,method=m,seed=r['seed'],selected_value=value,maximum_lower=float(lo),maximum_upper=float(hi),hit_lower=int(value>=.81*hi),hit_upper=int(value>=.81*lo)))
    dump('probability_empirical.json',empirical)
   print('EMPIRICAL',dom,d,flush=True)
 print('VALIDATION COMPLETE',flush=True)

if __name__=='__main__':main()
