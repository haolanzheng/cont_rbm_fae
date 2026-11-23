"""Paired alpha sweep at T=10 on the unchanged analytic-snapshot baseline.
PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.check_large_step_scale
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/baseline_alpha_matplotlib')
import json,platform,sys,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .run_geometry_cpu import run_greedy
from .geometry_cpu import CoordinateROM,rom_options,search,ball_normalized_probability_lower
from .run_extended_comparison_cpu import sphere_probability_lower,confidence_interval
ROOT=Path(__file__).resolve().parents[2]; SOURCE=ROOT/'runs/analytic_hf_d48_20261022'
BUDGET=ROOT/'runs/iteration_budget_20261022';OUT=ROOT/'runs/large_step_scale_T10_20261022'
ALPHAS=[.5,.75,1.,1.5,2.]
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 OUT.mkdir(exist_ok=True)
 base=json.loads((SOURCE/'results.json').read_text());cfg=base['config'];old=json.loads((BUDGET/'runs.json').read_text())
 runs=[];summary=[]
 for d in cfg['dimensions']:
  a=np.load(SOURCE/f'model_d{d}.npz');opts=rom_options(a)
  for dom in ['sphere','ball']:
   method='exp_rgd100' if dom=='sphere' else 'normalized_pg100'
   Z=np.load(SOURCE/f'test_{dom}_d{d}.npz')['test_parameters'];den=np.sum((Z@a['G0'])*Z,axis=1)
   for alpha in ALPHAS:
    group=[]
    for seed in cfg['seeds']:
     if alpha==.2:r=next(x for x in old if (x['dimension'],x['domain'],x['updates'],x['seed'])==(d,dom,10,seed)).copy()
     else:
      r=run_greedy(a['G0'],a['H0'],dom,method,seed,cfg|{'updates':10,'sphere_alpha':alpha,'alpha':alpha},rom_kwargs=opts)
      rom=CoordinateROM(a['G0'],a['H0'],**opts)
      for st in r['rounds']:rom.append(st['parameter'])
      P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross);e=Z@P.T
      r['test_rms']=float(np.sqrt(np.mean(np.maximum(np.sum((e@a['G0'])*e,axis=1),0)/den)))
      r['final_error_over_width']=r['errors'][-1]/float(a['widths'][d//2])
     r.update(updates=10,alpha=alpha,method='sphere_exp' if dom=='sphere' else 'ball_projected');runs.append(r);group.append(r)
    reference=[x for x in old if (x['dimension'],x['domain'],x['updates'])==(d,dom,100)]
    item={'d':d,'domain':dom,'alpha':alpha,'T':10,'median_error_over_width':float(np.median([x['final_error_over_width'] for x in group])),
          'median_test_rms':float(np.median([x['test_rms'] for x in group])),
          'reference_T100_error_over_width':float(np.median([x['final_error_over_width'] for x in reference])),
          'reference_T100_test_rms':float(np.median([x['test_rms'] for x in reference]))}
    summary.append(item);print(item,flush=True)
  save('runs.json',runs);save('summary.json',summary)
 # The same saved initial points and eigensystem as the original probability
 # experiment, for each dimension, domain and rank. No new sampling noise.
 previous=json.loads((SOURCE/'probability_curves.json').read_text());probs=[]
 for prev in previous:
  d=prev['dimension'];dom=prev['domain'];n=prev['rank'];f=np.load(SOURCE/f'frozen_{dom}_d{d}_n{n}.npz')
  H=np.diag(f['eigenvalues']);starts=f['starts_in_eigencoordinates'];gamma=.9;threshold=gamma**2*f['eigenvalues'][-1]
  for alpha in ALPHAS:
   point,data=search(H,starts,steps=10,domain=dom,alpha=alpha,omega=1.,checkpoints=[10],threshold=threshold,ball_step_rule='normalized')
   hits=data['snapshots']['10']['hits']
   if dom=='sphere' and alpha>.5:lower,certificate=None,None
   else:lower,certificate=(sphere_probability_lower if dom=='sphere' else ball_normalized_probability_lower)(H,gamma,10,alpha)
   diagnostics=trajectory_diagnostics(f['eigenvalues'],starts,dom,alpha,point)
   if alpha==.5:
    oldprob=json.loads((ROOT/'runs/step_scale_T10_20261022/probabilities.json').read_text())
    oldrow=next(x for x in oldprob if (x['d'],x['domain'],x['n'],x['alpha'])==(d,dom,n,alpha))
    assert hits==oldrow['hits']

   if alpha==.2:
    previous10=next(s for s in next(iter(prev['methods'].values()))['snapshots'] if s['T']==10)
    assert hits==previous10['hits'];assert abs(float(lower)-previous10['analytic_lower'])<1e-12
   ref=next(s for s in next(iter(prev['methods'].values()))['snapshots'] if s['T']==100)
   item={'d':d,'domain':dom,'n':n,'T':10,'alpha':alpha,'hits':hits,'count':len(starts),'q_hat':hits/len(starts),'ci95':confidence_interval(hits,len(starts)),
         'analytic_lower':None if lower is None else float(lower),'reference_T100_q_hat':ref['q_hat'],'reference_T100_lower':ref['analytic_lower'],'certificate':certificate,
         'max_feasibility_defect':data['max_feasibility_defect'],'minimum_increment_over_lambda1':data['minimum_objective_increment']/f['eigenvalues'][-1]}
   item.update(diagnostics);probs.append(item)
  print('Probability sweep complete:',dom,d,n,flush=True);save('probabilities.json',probs)
 # Interleaved, repeated timings for the complete N=24 greedy construction.
 a=np.load(SOURCE/'model_d48.npz');opts=rom_options(a);times=[]
 jobs=[(dom,T,alpha,seed,rep) for dom in ['sphere','ball'] for T,alpha in ([(10,a) for a in ALPHAS]+[(100,.2)]) for seed in cfg['seeds'] for rep in range(3)]
 np.random.default_rng(271828).shuffle(jobs)
 for dom,T,alpha,seed,rep in jobs:
  method='exp_rgd100' if dom=='sphere' else 'normalized_pg100'
  r=run_greedy(a['G0'],a['H0'],dom,method,seed,cfg|{'updates':T,'sphere_alpha':alpha,'alpha':alpha},rom_kwargs=opts)
  times.append({'d':48,'domain':dom,'T':T,'alpha':alpha,'seed':seed,'repeat':rep,'seconds':r['timed_total_seconds']})
 save('timings.json',times)
 save('metadata.json',{'source':str(SOURCE),'config':cfg|{'updates':10,'alphas':ALPHAS},'python':sys.executable,'platform':platform.platform(),
      'comparison':'Same training seeds, same 16384-point test sets, same saved 4096 probability starts per fixed space; CPU float64 one BLAS thread',
      'probability_ranks':'0,d/4,d/2','sphere_theory':'No analytical bound evaluated for alpha>0.5','timings':'d=48; 3 shuffled repetitions per seed, excluding exact diagnostics and common preprocessing',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 plot(summary,probs,cfg['dimensions']);print('COMPLETE',flush=True)
def trajectory_diagnostics(eigenvalues,starts,domain,alpha,returned):
 """Count objective decreases; validate diagonal trajectory against production search."""
 z=starts.copy();scale=np.linalg.norm(eigenvalues);top=eigenvalues[-1]
 if domain=='sphere':z/=np.linalg.norm(z,axis=1,keepdims=True)
 values=np.sum(z*z*eigenvalues,axis=1);best=values.copy();any_drop=np.zeros(len(z),dtype=bool);drops=0;min_inc=0.;selected=[]
 selected.append((values[:8]/top).tolist())
 for _ in range(10):
  hz=z*eigenvalues
  derivative=2*(hz-values[:,None]*z)/scale if domain=='sphere' else 2*hz/scale
  v=alpha*derivative/np.sqrt(1+np.sum(derivative**2,axis=1,keepdims=True))
  if domain=='sphere':
   length=np.linalg.norm(v,axis=1,keepdims=True);z=np.cos(length)*z+np.sinc(length/np.pi)*v;z/=np.linalg.norm(z,axis=1,keepdims=True)
  else:
   z=z+v;z/=np.maximum(1,np.linalg.norm(z,axis=1,keepdims=True))
  new=np.sum(z*z*eigenvalues,axis=1);inc=(new-values)/top;drop=inc < -1e-12
  any_drop|=drop;drops+=int(drop.sum());min_inc=min(min_inc,float(inc.min()));best=np.maximum(best,new);values=new
  selected.append((values[:8]/top).tolist())
 diff=float(np.max(np.abs(np.sum(returned**2*eigenvalues,axis=1)-best))/top)
 assert diff<1e-11,diff
 return {'fraction_trajectories_with_decrease':float(any_drop.mean()),'fraction_updates_decreasing':drops/(10*len(z)),
         'minimum_normalized_increment_checked':min_inc,'median_best_minus_last_over_lambda1':float(np.median((best-values)/top)),
         'first_eight_objective_trajectories_over_lambda1':selected,'diagonal_vs_production_best_value_difference':diff}

def plot(summary,probs,dims):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True)
 colors=['#245E91','#BC762C','#477A68','#B44E49','#655B80']
 for col,dom in enumerate(['sphere','ball']):
  for d,c in zip(dims,colors):
   s=[x for x in summary if x['d']==d and x['domain']==dom];p=[x for x in probs if x['d']==d and x['domain']==dom and x['n']==d//2]
   axes[0,col].plot(ALPHAS,[x['median_test_rms']/x['reference_T100_test_rms'] for x in s],marker='o',color=c,label=f'd={d}')
   axes[1,col].plot(ALPHAS,[x['q_hat'] for x in p],marker='o',color=c)
  axes[0,col].axhline(1,color='black',ls='--',lw=1);axes[0,col].set_title(dom.capitalize());axes[0,col].grid(alpha=.2)
  axes[1,col].set_ylim(0,1.03);axes[1,col].grid(alpha=.2);axes[1,col].set_xlabel(r'Step scale $\alpha$ ($T=10$)');axes[1,col].set_xticks(ALPHAS)
 axes[0,0].set_ylabel('Test error / reference test error\n(reference: T=100, alpha=0.2)')
 axes[1,0].set_ylabel(r'Empirical $q_{10}$ at $n=d/2$, $\gamma=0.9$')
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=5);fig.tight_layout(rect=(0,.055,1,1))
 for ext in ['png','pdf']:fig.savefig(OUT/f'step_scale_comparison.{ext}',dpi=160,bbox_inches='tight')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
