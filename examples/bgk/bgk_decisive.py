#!/usr/bin/env python3
"""Predeclared paired BGK mechanism controls; all states use the full grid."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,concurrent.futures,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from examples.bgk.boltzmann_bgk_screen import BGKModel,Config,FixedROM,relative_l2,projected_adam_search
from examples.bgk.bgk_strict import StrictROM

ROOT=Path(__file__).parent
OUT=ROOT/'decisive_output'
LO=np.array([-1.,1.]);SPAN=np.array([2.,9.])
CORNERS=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
METHODS=['fixed','fixed_corners','fresh','fresh_corners','pilot_only','hybrid']
SEEDS=[901,902,903,904,905,906]
CHECKPOINTS=[5,10,15]

def dump(path,data):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def points():
 typical=qmc.Sobol(2,scramble=True,seed=873441).random_base2(8)
 grid=np.linspace(0,1,17)
 boundary=np.unique(np.array([(x,y) for x in grid for y in [0,1]]+
                            [(x,y) for x in [0,1] for y in grid]),axis=0)
 d=np.array([.001,.005,.025,.125])
 corners=np.array([[c[0]+(1-2*c[0])*x,c[1]+(1-2*c[1])*y]
                   for c in CORNERS for x in d for y in d])
 z=np.vstack([typical,boundary,corners])
 return LO+z*SPAN,{'typical':[0,len(typical)],'boundary':[len(typical),len(typical)+len(boundary)],
                 'corner_neighborhood':[len(typical)+len(boundary),len(z)]}

def truth_one(mu):
 m=BGKModel(Config(fom_tol=1e-11));f,info=m.solve(*mu)
 if not info['converged']:raise RuntimeError('FOM did not converge')
 return f,info

def init(workers):
 OUT.mkdir(exist_ok=True)
 par,groups=points()
 protocol=dict(scope='BGK screening surrogate, not full Boltzmann',seeds=SEEDS,methods=METHODS,
  max_rank=15,checkpoints=CHECKPOINTS,fom_tol=1e-11,inner='full analytic Jacobian, safeguarded GN with scaled scipy TRF fallback',
  initial_mu=[0.,5.5],pool_budget=50,hybrid_pilot=14,hybrid_extra=36,hybrid_starts=4,hybrid_lr=.08,
  test_groups=groups,test_parameters=par.tolist(),
  primary='N15 maximum relative state error over fixed 384-point union',
  secondary=['typical max/mean/p95','boundary max','corner neighborhood max','search CPU/wall times'],
  comparisons=['fixed_corners/fixed: boundary inclusion','fresh/fixed: refresh',
   'fresh_corners/fixed_corners: refresh with matched boundary coverage',
   'hybrid/fresh_corners: matched pilots, 36 extra random points versus 36 Adam updates',
   'hybrid/pilot_only: contribution of updates, unequal evaluation count explicitly'],
  timing='Single BLAS thread, per-process CPU time avoids worker contention; training excludes audits. Equal-time frozen-space comparison is separate.',
  stopping='All predeclared runs to rank15, no stopping based on advantage; six seeds is a mechanism study, not universal statistical certification.')
 if not (OUT/'protocol.json').exists():dump(OUT/'protocol.json',protocol)
 if not (OUT/'truth.npy').exists():
  infos=[];arr=np.lib.format.open_memmap(OUT/'truth.partial.npy',mode='w+',dtype=float,shape=(len(par),100,32,32))
  with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
   for i,(f,info) in enumerate(ex.map(truth_one,par)):
    arr[i]=f;infos.append(info)
    if (i+1)%64==0:print('truth',i+1,'/',len(par),flush=True)
  arr.flush();del arr;(OUT/'truth.partial.npy').rename(OUT/'truth.npy');dump(OUT/'truth_info.json',infos)

def precision():
 m=BGKModel(Config());strictm=BGKModel(Config(fom_tol=1e-11));sn=np.load(ROOT/'mechanism_output/trajectory_snapshots.npz')
 kernel=[];comparison=[];gradients=[];floors=[]
 for n in [1,7,15]:
  rom=StrictROM(m,list(sn['adam'][:n]));y=np.zeros(n-1)
  r,j=FixedROM._residual_jacobian(rom,y,.17,4.51);rr,jj=rom._residual_jacobian(y,.17,4.51)
  kernel.append(dict(rank=n,r_relative=float(np.linalg.norm(r-rr)/np.linalg.norm(r)),
    jac_relative=float(np.linalg.norm(j-jj)/max(np.linalg.norm(j),1e-300))))
 for method in ['pool','adam']:
  for n in [7,10,15]:
   a=StrictROM(m,list(sn[method][:n]));b=StrictROM(m,list(sn[method][:n]),solver='scipy',tol=1e-13)
   for mu in [[.17,4.51],[.98,9.8],[-.9,1.2]]:
    x=a.solve_inner(*mu);z=b.solve_inner(*mu)
    comparison.append(dict(method=method,rank=n,mu=mu,gn_value=x['value'],scipy_value=z['value'],
      value_relative=abs(x['value']/z['value']-1),state_difference=relative_l2(m,a.field(x['y']),b.field(z['y'])),
      solver=x['solver'],gradient_inf=x['coefficient_gradient_inf']))
    h=2e-4;fd=[]
    for k in range(2):
     pp=np.array(mu);mm=pp.copy();pp[k]+=h*SPAN[k];mm[k]-=h*SPAN[k]
     fd.append((math.log(a.solve_inner(*pp)['value'])-math.log(a.solve_inner(*mm)['value']))/(2*h))
    actual=x['gradient']*SPAN/x['value'];fd=np.array(fd)
    gradients.append(dict(method=method,rank=n,mu=mu,analytic=actual.tolist(),finite_difference=fd.tolist(),
       relative=float(np.linalg.norm(actual-fd)/max(np.linalg.norm(fd),1e-12))))
   print('precision',method,n,flush=True)
 old=json.loads((ROOT/'mechanism_output/trajectory.json').read_text());mus=np.array(old['parameters']['typical']+old['parameters']['boundary'])
 refs=[truth_one(mu)[0] for mu in mus]
 for method in ['pool','adam']:
  rom=StrictROM(strictm,list(sn[method]))
  old_error=[];new_error=[];truth_error=[]
  for mu,truth in zip(mus,refs):
   loose,_=m.solve(*mu);z=rom.solve_inner(*mu);f=rom.field(z['y'])
   old_error.append(relative_l2(m,f,loose));new_error.append(relative_l2(m,f,truth));truth_error.append(relative_l2(m,loose,truth))
  floors.append(dict(method=method,old_truth_error=old_error,tight_truth_error=new_error,
                    loose_vs_tight=truth_error,typical_count=20))
  print('truth-floor',method,'old/new typical',max(old_error[:20]),max(new_error[:20]),flush=True)
 result=dict(kernel=kernel,solver_comparison=comparison,gradients=gradients,truth_floor=floors)
 dump(OUT/'precision.json',result)
 assert max(x['r_relative'] for x in kernel)<1e-10
 assert max(x['jac_relative'] for x in kernel)<1e-10
 assert max(x['state_difference'] for x in comparison)<1e-8
 # Finite differences at tiny residuals can have cancellation, reported explicitly.
 print('precision done', 'max gradient relative',max(x['relative'] for x in gradients),flush=True)

def values(rom,z):
 return [rom.solve_inner(*(LO+point*SPAN)) for point in z]

def refine(rom,pilot,answers):
 # Choose up to four distinct promising pilots; no truth data used.
 order=np.argsort([-a['value'] for a in answers]);chosen=[]
 for i in order:
  if all(np.linalg.norm(pilot[i]-pilot[j])>.15 for j in chosen):chosen.append(int(i))
  if len(chosen)==4:break
 for i in order:
  if len(chosen)==4:break
  if int(i) not in chosen:chosen.append(int(i))
 best=int(np.argmax([x['value'] for x in answers]));bz=pilot[best].copy();bv=answers[best]['value']
 pilot_best=bv;trace=[]
 for start in chosen:
  z=pilot[start].copy();ans=answers[start];m=np.zeros(2);v=np.zeros(2)
  for t in range(1,10):
   grad=ans['gradient']*SPAN/max(ans['value'],1e-300)
   m=.9*m+.1*grad;v=.999*v+.001*grad**2
   z=np.clip(z+.08*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8),0,1)
   ans=rom.solve_inner(*(LO+SPAN*z))
   if ans['value']>bv:bz=z.copy();bv=ans['value']
   trace.append(dict(z=z.tolist(),value=ans['value']))
 return bz,bv,dict(pilot_best=pilot_best,gain=bv/max(pilot_best,1e-300),starts=chosen,trace=trace)

def select(rom,method,seed,rank,selected):
 fixed=np.random.default_rng(seed).random((50,2))
 fresh=np.random.default_rng(np.random.SeedSequence([seed,rank,7721])).random((50,2))
 pilot=np.vstack([CORNERS,fresh[:10]])
 t=time.perf_counter();cpu=time.process_time();before=len(rom.solve_records)
 extra={};logical=50
 if method=='original_adam':
  z=projected_adam_search(rom,((-1.,1.),(1.,10.)),5,50,seed+rank)
  mu=z['mu'];best=z['raw_value']
 elif method=='hybrid':
  aa=values(rom,pilot);z,best,extra=refine(rom,pilot,aa);mu=LO+SPAN*z
 else:
  candidates={'fixed':fixed,'fixed_corners':np.vstack([CORNERS,fixed[:46]]),
              'fresh':fresh,'fresh_corners':np.vstack([CORNERS,fresh[:46]]),'pilot_only':pilot}[method]
  aa=values(rom,candidates);vv=np.array([a['value'] for a in aa]);logical=len(candidates)
  if selected:
   for mu0 in selected:
    vv[np.linalg.norm((LO+SPAN*candidates-np.array(mu0))/SPAN,axis=1)<1e-10]=-np.inf
  idx=int(np.argmax(vv));best=float(vv[idx]);mu=LO+SPAN*candidates[idx]
  if method in ['fresh_corners','pilot_only']:
   pb=max(a['value'] for a in aa[:14]);extra=dict(pilot_best=pb,gain=best/max(pb,1e-300))
 record=dict(mu=mu.tolist(),value=best,logical_evaluations=logical,
  unique_solves=len(rom.solve_records)-before,wall_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,**extra)
 return mu,record

def stats(rom):
 rows=rom.solve_records
 return dict(solves=len(rows),fallbacks=sum(x['solver']=='scipy_trf' for x in rows),
  failures=sum(not x['success'] for x in rows),max_coefficient_gradient=max([x['coefficient_gradient_inf'] for x in rows] or [0]),
  max_normal_condition=max([x['normal_condition'] for x in rows] or [0]))

def evaluate(rom,par,truth,groups):
 errors=[];residuals=[]
 for mu,f in zip(par,truth):
  z=rom.solve_inner(*mu);errors.append(relative_l2(rom.model,rom.field(z['y']),f));residuals.append(z['value'])
 ee=np.array(errors);rr=np.array(residuals);out=dict(errors=errors,residuals=residuals)
 for name,(start,end) in {**groups,'all':[0,len(par)]}.items():
  e=ee[start:end];out[name]=dict(max=float(max(e)),mean=float(np.mean(e)),p95=float(np.quantile(e,.95)),
      argmax_mu=par[start+int(np.argmax(e))].tolist())
 out['audit_max_residual']=float(max(rr));out['q90_over_max']=float(np.quantile(rr[:256],.9)/max(rr))
 out['p90']=float(np.mean(rr[:256]>=.9*max(rr)))
 return out

def job(task):
 seed,method=task;path=OUT/f'{method}_{seed}.json';snapfile=OUT/f'{method}_{seed}.npz'
 if path.exists():
  old=json.loads(path.read_text())
  if old.get('complete'):return dict(seed=seed,method=method,already_complete=True)
  raise RuntimeError('Partial checkpoint exists: use explicit resume support before rerun')
 m=BGKModel(Config(fom_tol=1e-11,inner_maxiter=160));f,info=m.solve(0.,5.5);snaps=[f]
 par,groups=points();truth=np.load(OUT/'truth.npy',mmap_mode='r')
 data=dict(seed=seed,method=method,initial_fom=info,selections=[],diagnostics=[],training_cpu_seconds=0.,training_wall_seconds=0.,complete=False)
 for rank in range(1,16):
  t=time.perf_counter();cpu=time.process_time();rom=StrictROM(m,snaps)
  data['training_cpu_seconds']+=time.process_time()-cpu;data['training_wall_seconds']+=time.perf_counter()-t
  if rank in CHECKPOINTS:
   diag=evaluate(rom,par,truth,groups);diag['rank']=rank;data['diagnostics'].append(diag)
   print(f"seed={seed} method={method} N={rank} max={diag['all']['max']:.3e} typical={diag['typical']['max']:.3e} boundary={diag['boundary']['max']:.3e}",flush=True)
  if rank<15:
   # Audit values must not subsidize training timings or selection.
   rom.cache.clear();rom.solve_records=[]
   mu,rec=select(rom,method,seed,rank,[r['mu'] for r in data['selections']]);rec['rank']=rank;rec['inner_stats']=stats(rom)
   t=time.perf_counter();cpu=time.process_time();f,info=m.solve(*mu)
   if not info['converged']:raise RuntimeError('FOM convergence failure')
   rec['fom']=info;rec['fom_cpu_seconds']=time.process_time()-cpu;rec['fom_wall_seconds']=time.perf_counter()-t
   data['training_cpu_seconds']+=rec['cpu_seconds']+rec['fom_cpu_seconds'];data['training_wall_seconds']+=rec['wall_seconds']+rec['fom_wall_seconds']
   data['selections'].append(rec);snaps.append(f)
  else:data['complete']=True;data['final_inner_stats']=stats(rom)
  dump(path,data);np.savez_compressed(snapfile,snapshots=np.stack(snaps))
 return dict(seed=seed,method=method,complete=True)

def run(workers):
 tasks=[(seed,method) for seed in SEEDS for method in METHODS]+[(901,'original_adam')]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
  fs={ex.submit(job,task):task for task in tasks}
  for f in concurrent.futures.as_completed(fs):print('DONE',f.result(),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','precision','run']);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
 if args.action=='init':init(args.workers)
 elif args.action=='precision':precision()
 else:run(args.workers)
