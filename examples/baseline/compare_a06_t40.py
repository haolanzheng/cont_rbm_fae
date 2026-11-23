"""Paired comparison of (alpha,T)=(0.6,40) and (1,20), same models and samples.
PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.compare_a06_t40
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,platform,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .geometry_cpu import CoordinateROM,rom_options,search,ball_normalized_probability_lower
from .run_geometry_cpu import run_greedy
from .run_extended_comparison_cpu import sphere_probability_lower,confidence_interval
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'runs/analytic_hf_a1_t20_20261022';OUT=ROOT/'runs/compare_a06_t40_20261022'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
 OUT.mkdir(exist_ok=True);base=json.loads((SOURCE/'results.json').read_text());cfg=base['config'];allruns=[];summary=[]
 for d in cfg['dimensions']:
  a=np.load(SOURCE/f'model_d{d}.npz');opts=rom_options(a)
  for dom in ['sphere','ball']:
   method='exp_rgd20' if dom=='sphere' else 'normalized_pg20';Z=np.load(SOURCE/f'test_{dom}_d{d}.npz')['test_parameters'];den=np.sum((Z@a['G0'])*Z,axis=1)
   for alpha,T in [(1.,20),(.6,40)]:
    group=[]
    for seed in cfg['seeds']:
     if T==20:r=next(x for x in base['runs'] if (x['dimension'],x['domain'],x['method'],x['seed'])==(d,dom,method,seed)).copy()
     else:r=run_greedy(a['G0'],a['H0'],dom,method,seed,cfg|{'updates':T,'alpha':alpha,'sphere_alpha':alpha},rom_kwargs=opts)
     rom=CoordinateROM(a['G0'],a['H0'],**opts)
     for st in r['rounds']:rom.append(st['parameter'])
     P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross);e=Z@P.T
     r.update(alpha=alpha,updates=T,method='sphere_exp' if dom=='sphere' else 'ball_projected',test_rms=float(np.sqrt(np.mean(np.maximum(np.sum((e@a['G0'])*e,axis=1),0)/den))),final_error_over_width=r['errors'][-1]/float(a['widths'][d//2]))
     assert all(st['h_actions_per_start']==T+1 for st in r['rounds'])
     group.append(r);allruns.append(r)
    rec={'d':d,'domain':dom,'alpha':alpha,'T':T,'median_error_over_width':float(np.median([r['final_error_over_width'] for r in group])),'median_test_rms':float(np.median([r['test_rms'] for r in group]))}
    summary.append(rec);print(rec,flush=True)
  save('runs.json',allruns);save('summary.json',summary)
 oldprob=json.loads((SOURCE/'probability_curves.json').read_text());probs=[]
 for prev in oldprob:
  d=prev['dimension'];dom=prev['domain'];n=prev['rank'];f=np.load(SOURCE/f'frozen_{dom}_d{d}_n{n}.npz');H=np.diag(f['eigenvalues']);starts=f['starts_in_eigencoordinates'];top=f['eigenvalues'][-1]
  for alpha,T in [(1.,20),(.6,40)]:
   if T==20:
    st=next(iter(prev['methods'].values()))['snapshots'][-1];hits=st['hits'];defect=prev['max_feasibility_defect'];increment=prev['minimum_increment_over_lambda1']
   else:
    _,data=search(H,starts,steps=T,domain=dom,alpha=alpha,omega=1,ball_step_rule='normalized',threshold=.81*top,checkpoints=[T]);hits=data['snapshots'][str(T)]['hits'];defect=data['max_feasibility_defect'];increment=data['minimum_objective_increment']/top
   lower,cert=(sphere_probability_lower if dom=='sphere' else ball_normalized_probability_lower)(H,.9,T,alpha)
   p0=prev['direct_analytic_upper'];upper=float(-np.expm1(10000*np.log1p(-p0))) if p0<1 else 1.
   probs.append({'d':d,'domain':dom,'n':n,'alpha':alpha,'T':T,'hits':hits,'count':len(starts),'q_hat':hits/len(starts),'ci95':confidence_interval(hits,len(starts)),'analytic_lower':float(lower),'pool10000_upper':upper,'certificate':cert,'max_feasibility_defect':defect,'minimum_increment_over_lambda1':float(increment)})
  print('Probability complete:',dom,d,n,flush=True);save('probabilities.json',probs)
 a=np.load(SOURCE/'model_d48.npz');times=[]
 jobs=[(dom,alpha,T,seed,rep) for dom in ['sphere','ball'] for alpha,T in [(1.,20),(.6,40)] for seed in cfg['seeds'] for rep in range(3)]
 np.random.default_rng(271828).shuffle(jobs)
 for dom,alpha,T,seed,rep in jobs:
  method='exp_rgd20' if dom=='sphere' else 'normalized_pg20'
  r=run_greedy(a['G0'],a['H0'],dom,method,seed,cfg|{'alpha':alpha,'sphere_alpha':alpha,'updates':T},rom_kwargs=rom_options(a))
  times.append({'d':48,'domain':dom,'alpha':alpha,'T':T,'seed':seed,'repeat':rep,'seconds':r['timed_total_seconds']})
 save('timings.json',times)
 save('metadata.json',{'source':str(SOURCE),'config':cfg,'compared_settings':[{'alpha':1,'T':20},{'alpha':.6,'T':40}],'device':'CPU','threads':1,'platform':platform.platform(),'python':sys.executable,'test_count':16384,'fixed_space_starts':4096,'probability_ranks':'0,d/4,d/2','pairing':'Same models, same seeds, same saved test parameters and probability starts. Current 20-step histories reused; timings freshly interleaved, 3 repetitions per seed, d=48, excluding audits and common preprocessing.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 print('COMPLETE',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
