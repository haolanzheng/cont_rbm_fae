"""Compare optimization budgets on the unchanged analytic-snapshot baseline.
Run: PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.check_iteration_budget
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time,platform,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .run_geometry_cpu import run_greedy
from .geometry_cpu import CoordinateROM,rom_options
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/analytic_hf_d48_20261022'
OUT=ROOT/'runs/iteration_budget_20261022'
BUDGETS=[10,20,50,100]
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 OUT.mkdir(exist_ok=True)
 baseline=json.loads((SOURCE/'results.json').read_text());cfg=baseline['config'];runs=[];summary=[]
 for d in cfg['dimensions']:
  a=np.load(SOURCE/f'model_d{d}.npz');opts=rom_options(a)
  for domain in ['sphere','ball']:
   method='exp_rgd100' if domain=='sphere' else 'normalized_pg100'
   Z=np.load(SOURCE/f'test_{domain}_d{d}.npz')['test_parameters'];den=np.sum((Z@a['G0'])*Z,axis=1)
   for T in BUDGETS:
    group=[]
    for seed in cfg['seeds']:
     if T==100:r=next(x for x in baseline['runs'] if (x['dimension'],x['domain'],x['method'],x['seed'])==(d,domain,method,seed)).copy()
     else:r=run_greedy(a['G0'],a['H0'],domain,method,seed,cfg|{'updates':T},rom_kwargs=opts)
     r['updates']=T;r['method']='sphere_exp' if domain=='sphere' else 'ball_projected'
     rom=CoordinateROM(a['G0'],a['H0'],**opts)
     for st in r['rounds']:rom.append(st['parameter'])
     P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross);e=Z@P.T
     r['test_rms']=float(np.sqrt(np.mean(np.maximum(np.sum((e@a['G0'])*e,axis=1),0)/den)))
     r['final_error_over_width']=r['errors'][-1]/float(a['widths'][d//2]);group.append(r);runs.append(r)
    entry={'d':d,'domain':domain,'T':T,'median_error_over_width':float(np.median([r['final_error_over_width'] for r in group])),
           'median_test_rms':float(np.median([r['test_rms'] for r in group]))}
    summary.append(entry);print(entry,flush=True)
  save('runs.json',runs);save('summary.json',summary)
 # Interleaved timing comparison at d=48; identical CPU, thread count, seeds,
 # and code path. Diagnostic eigensolves and test errors excluded by cost fields.
 d=48;a=np.load(SOURCE/f'model_d{d}.npz');opts=rom_options(a);timings=[]
 jobs=[(domain,T,seed,rep) for domain in ['sphere','ball'] for T in [10,100] for seed in cfg['seeds'] for rep in range(3)]
 np.random.default_rng(271828).shuffle(jobs)
 for domain,T,seed,rep in jobs:
  method='exp_rgd100' if domain=='sphere' else 'normalized_pg100'
  r=run_greedy(a['G0'],a['H0'],domain,method,seed,cfg|{'updates':T},rom_kwargs=opts)
  timings.append({'d':d,'domain':domain,'T':T,'seed':seed,'repeat':rep,'seconds':r['timed_total_seconds']})
 save('timings.json',timings)
 probs=json.loads((SOURCE/'probability_curves.json').read_text());pout=[]
 for r in probs:
  if r['rank']!=r['dimension']//2:continue
  ss=next(iter(r['methods'].values()))['snapshots']
  for st in ss:
   if st['T'] in BUDGETS:pout.append({'d':r['dimension'],'domain':r['domain'],'n':r['rank'],**st})
 save('fixed_space_probabilities.json',pout)
 save('metadata.json',{'source':str(SOURCE),'budgets':BUDGETS,'alpha':.2,'seeds':cfg['seeds'],'test_count':len(Z),'platform':platform.platform(),'python':sys.executable,'device':'CPU','threads':1,'timings':'d=48; 3 shuffled repetitions for each training seed and budget; exact audits excluded','training_100':'Reused original histories; freshly timed in timing comparison.'})
 print('COMPLETE',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
