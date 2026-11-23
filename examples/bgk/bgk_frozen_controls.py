"""Paired pilot/extra-sampling/Adam controls on two identical-within-pair spaces."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import concurrent.futures,json,time
from pathlib import Path
import numpy as np
from examples.bgk.bgk_decisive import OUT,ROOT,SEEDS,LO,SPAN,CORNERS,points,dump,select,evaluate,stats
from examples.bgk.bgk_strict import StrictROM
from examples.bgk.boltzmann_bgk_screen import BGKModel,Config

def prepare():
 old=json.loads((ROOT/'mechanism_output/trajectory.json').read_text())
 dump(OUT/'frozen_protocol.json',dict(rank=7,spaces=['pool_history','adam_history'],seeds=SEEDS,
  rebuilding='original first seven snapshot parameters, FOM tolerance 1e-11',
  methods=['pilot14','fresh_corners50','hybrid50','fresh_corners_cpu_matched'],
  paired='Within each frozen space and seed, same four corners plus ten random pilots',
  enrichment='Evaluate both fresh_corners50 and hybrid50 one-step enrichments on all 384 common test points',
  cpu_matched='Random trace selected only through the last completed evaluation whose process CPU time does not exceed hybrid CPU budget; at most one discarded overshoot solve.'))
 m=BGKModel(Config(fom_tol=1e-11,inner_maxiter=160));par,groups=points();truth=np.load(OUT/'truth.npy',mmap_mode='r')
 for method in ['pool','adam']:
  mus=[[0.,5.5]]+[x['mu'] for x in old[method]['selections'][:6]]
  snaps=[]
  for mu in mus:
   f,info=m.solve(*mu)
   if not info['converged']:raise RuntimeError('FOM convergence')
   snaps.append(f)
  np.savez_compressed(OUT/f'frozen_{method}_basis.npz',snapshots=np.stack(snaps))
  rom=StrictROM(m,snaps);diag=evaluate(rom,par,truth,groups)
  dump(OUT/f'frozen_{method}_baseline.json',diag)
  print('frozen baseline',method,diag['all']['max'],flush=True)

def run_one(task):
 path,seed=task;outfile=OUT/f'frozen_{path}_{seed}.json'
 if outfile.exists():return str(outfile)
 m=BGKModel(Config(fom_tol=1e-11,inner_maxiter=160));snap=list(np.load(OUT/f'frozen_{path}_basis.npz')['snapshots'])
 rom=StrictROM(m,snap);mu,hybrid=select(rom,'hybrid',seed,7,[])
 hybrid['inner_stats']=stats(rom)
 rom.cache.clear();rom.solve_records=[]
 pool_mu,pool=select(rom,'fresh_corners',seed,7,[])
 # Audit for matched CPU use: independent replay from the same pilots.
 rom.cache.clear();rom.solve_records=[]
 rng=np.random.default_rng(np.random.SeedSequence([seed,7,7721]));z=np.vstack([CORNERS,rng.random((1000,2))])
 budget=hybrid['cpu_seconds'];cpu=time.process_time();best=-np.inf;best_mu=None;used=0;lastcpu=0.
 for point in z:
  loc=LO+SPAN*point;ans=rom.solve_inner(*loc);elapsed=time.process_time()-cpu
  if elapsed>budget:break
  used+=1;lastcpu=elapsed
  if ans['value']>best:best=ans['value'];best_mu=loc.tolist()
 matched=dict(value=best,mu=best_mu,evaluations=used,cpu_seconds=lastcpu,budget=budget)
 par,groups=points();truth=np.load(OUT/'truth.npy',mmap_mode='r');enrich={}
 for method,loc in [('fresh_corners',pool_mu),('hybrid',mu)]:
  f,info=m.solve(*loc)
  if not info['converged']:raise RuntimeError('FOM convergence')
  defect=f.ravel()-rom.u@(rom.u.T@f.ravel())
  novelty=float(np.linalg.norm(defect)/np.linalg.norm(f))
  enriched=StrictROM(m,snap+[f]);enrich[method]=evaluate(enriched,par,truth,groups)
  enrich[method]['relative_projection_novelty']=novelty;enrich[method]['inner_stats']=stats(enriched)
 result=dict(space=path,seed=seed,rank=7,pool=pool,hybrid=hybrid,cpu_matched=matched,enrichment=enrich)
 dump(outfile,result)
 print('frozen',path,seed,'search gain',hybrid['value']/pool['value'],
       'post error ratio',enrich['hybrid']['all']['max']/enrich['fresh_corners']['all']['max'],flush=True)
 return dict(space=path,seed=seed)

if __name__=='__main__':
 prepare()
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  for result in ex.map(run_one,[(p,s) for p in ['pool','adam'] for s in SEEDS]):print('DONE',result,flush=True)
