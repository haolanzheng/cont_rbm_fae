"""Temporal mechanism audit: fixed labels, pointwise errors, order intervention.

All truth information is diagnostic only; it never enters greedy selection.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,concurrent.futures,time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from examples.bgk.bgk_decisive import ROOT,OUT as PREV,points,dump,evaluate,stats,LO,SPAN
from examples.bgk.bgk_strict import StrictROM
from examples.bgk.boltzmann_bgk_screen import BGKModel,Config,relative_l2,projected_adam_search

OUT=ROOT/'temporal_output'
OLD=json.loads((ROOT/'mechanism_output/trajectory.json').read_text())

def model():return BGKModel(Config(fom_tol=1e-11,inner_maxiter=240))

def init():
 OUT.mkdir(exist_ok=True)
 par,groups=points();extra=np.array(OLD['parameters']['typical']+OLD['parameters']['boundary'])
 par=np.vstack([par,extra]);groups.update(old_typical=[384,404],old_boundary=[404,420])
 protocol=dict(question='Does endogenous Adam selection migrate from difficult to typical regions, explaining delayed error crossover?',
  test_parameters=par.tolist(),groups=groups,rank_limit=25,rank_safety_singular_ratio=1e-11,
  typical='Uniform parameter law: 256 independent Sobol points, not synonymous with interior away from boundary.',
  fixed_hardness=['Common rank1 relative state error','Independent 64-snapshot affine POD3 relative projection error'],
  hard_quantiles=[.8,.9],selection_windows={'early':[1,8],'late':[9,14]},
  reorder='Keep center first; move original selections 9..14 ahead of 1..8. Identical final span; intervention tests timing only.',
  metrics=['all420 max','uniform256 max/mean/p95','original20 max','fixed initial-error hard/remaining strata','POD3 hard/remaining strata','pointwise marginal log improvement'],
  caution='Retrospective trajectory study. Hardness proxies are operational definitions, not intrinsic PDE difficulty. No data-driven threshold tuning; no universal causal claim.')
 dump(OUT/'protocol.json',protocol)
 m=model()
 if not (OUT/'extra_truth.npy').exists():
  arr=[]
  for mu in extra:
   f,info=m.solve(*mu);assert info['converged'];arr.append(f)
  np.save(OUT/'extra_truth.npy',arr)
 if not (OUT/'reference.npz').exists():
  mus=LO+SPAN*qmc.Sobol(2,scramble=True,seed=451820).random_base2(6)
  arr=[]
  for mu in mus:
   f,info=m.solve(*mu);assert info['converged'];arr.append(f.ravel())
  X=np.array(arr);mean=X.mean(axis=0);u,s,v=np.linalg.svd((X-mean).T,full_matrices=False)
  f,info=m.solve(0.,5.5)
  np.savez_compressed(OUT/'reference.npz',mean=mean,directions=u[:,:3],initial=f,parameters=mus,singular_values=s)
 print('INIT ready',flush=True)

def context():
 p=json.loads((OUT/'protocol.json').read_text());par=np.array(p['test_parameters'])
 truth=np.concatenate([np.load(PREV/'truth.npy'),np.load(OUT/'extra_truth.npy')])
 return par,truth,p['groups']

def hardness(f,ref):
 flat=f.reshape(len(f),-1);norm=np.linalg.norm(flat,axis=1)
 delta=flat-ref['mean'];proj=delta-(delta@ref['directions'])@ref['directions'].T
 return dict(initial=np.linalg.norm(flat-ref['initial'].ravel(),axis=1)/norm,
             pod3=np.linalg.norm(proj,axis=1)/norm)

def run(name):
 path=OUT/f'{name}.json'
 if path.exists() and json.loads(path.read_text()).get('complete'):return name+' cached'
 m=model();ref=np.load(OUT/'reference.npz');par,truth,groups=context();t=time.time()
 if name.startswith('old_'):
  kind=name.split('_')[1];mus=[[0.,5.5]]+[s['mu'] for s in OLD[kind]['selections']]
  if name.endswith('_swapped'):mus=[mus[0]]+mus[9:15]+mus[1:9]
  snaps=[]
  for mu in mus:
   f,info=m.solve(*mu);assert info['converged'];snaps.append(f)
  maxrank=15
 else:
  kind=name.split('_')[1];mus=[[0.,5.5]];snaps=[ref['initial']];maxrank=25
 pool=np.array(OLD['parameters']['pool_training_initial'])
 data=dict(name=name,mus=mus,diagnostics=[],selections=[],complete=False)
 for n in range(1,maxrank+1):
  ss=np.stack(snaps[:n]);sv=np.linalg.svd(ss.reshape(n,-1).T,compute_uv=False)
  ratio=float(sv[-1]/sv[0])
  if ratio<1e-11:
   data['stopped']=dict(rank=n,reason='snapshot numerical rank safeguard',singular_ratio=ratio);break
  rom=StrictROM(m,list(ss));d=evaluate(rom,par,truth,groups);d.update(rank=n,singular_ratio=ratio,inner_stats=stats(rom))
  # Stable explicit projection subtraction rather than norm^2 minus norm^2.
  proj=[]
  for batch in np.array_split(truth.reshape(len(truth),-1),7):
   delta=batch-rom.base_field.ravel();D=rom.directions.reshape(rom.base_field.size,n-1)
   diff=delta-(delta@D)@D.T
   proj.extend((np.linalg.norm(diff,axis=1)/np.linalg.norm(batch,axis=1)).tolist())
  d['projection_errors']=proj;data['diagnostics'].append(d)
  print(name,'N',n,'uniform max',d['typical']['max'],'old20',d['old_typical']['max'],'sec',round(time.time()-t),flush=True)
  if n<maxrank:
   if name.startswith('old_'):mu=np.array(mus[n]);f=snaps[n]
   else:
    rom.cache.clear();rom.solve_records=[]
    if kind=='adam':
     pick=projected_adam_search(rom,((-1.,1.),(1.,10.)),5,50,901+n)
     mu=pick['mu'];data.setdefault('adam_searches',[]).append(dict(rank=n,trace=pick['history']))
    else:
     vals=np.array([rom.solve_inner(*mu)['value'] for mu in pool])
     for oldmu in mus:vals[np.linalg.norm((pool-oldmu)/SPAN,axis=1)<1e-10]=-np.inf
     mu=pool[int(np.argmax(vals))]
    f,info=m.solve(*mu);assert info['converged'];snaps.append(f);mus.append(mu.tolist());data['mus']=mus
   ans=rom.solve_inner(*mu);h=hardness(np.array([f]),ref)
   data['selections'].append(dict(rank=n,mu=np.array(mu).tolist(),pre_state_error=relative_l2(m,rom.field(ans['y']),f),
      initial_error=float(h['initial'][0]),pod3_error=float(h['pod3'][0]),value=ans['value']))
  dump(path,data)
  np.savez_compressed(OUT/f'{name}.npz',snapshots=np.stack(snaps),mus=np.array(mus))
 data['complete']=True;data['seconds']=time.time()-t;dump(path,data)
 return name+' done'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','run']);p.add_argument('--names',nargs='+',default=['old_pool','old_adam','new_pool','new_adam']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.action=='init':init()
 else:
  with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
   for r in ex.map(run,a.names):print(r,flush=True)
