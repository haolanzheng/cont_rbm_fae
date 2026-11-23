"""Screen of collective rational forcing with the unchanged GD rule."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time,itertools
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .collective import CollectiveDiffusion

OUT=Path(__file__).resolve().parents[3]/'runs/cube_candidate_screen_20261104'
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def main():
  rows=[];curves=[]
  save('collective_settings.json',dict(dimensions=[16,32,64],rho=[.78,.9],delta=[.05,.2,.5],seeds=[1000,1001,1002,1003],alpha=.6,T=80,N='d/2',normalizer='Frobenius norm of lifted residual Gram H_n',test='16384 Sobol interior; all vertices at d16, random vertices and up to three coordinate flips from all-ones at d32/64',methods=['gd','pool256','pool4096','pool10000'],width='not asserted',device='CPU',threads=1))
  with threadpool_limits(limits=1):
    for d in (16,32,64):
      interior=2*qmc.Sobol(d,scramble=True,seed=202611056+d).random_base2(14)-1
      rng=np.random.default_rng(202611056+d)
      if d==16:
        k=np.arange(2**d);vertices=2*((k[:,None]>>np.arange(d))&1)-1
      else:
        combos=[x for count in range(4) for x in itertools.combinations(range(d),count)]
        near=np.ones((len(combos),d))
        for i,c in enumerate(combos):near[i,list(c)]=-1.
        vertices=np.vstack([rng.choice([-1.,1.],size=(32768,d)),near])
      test=np.vstack([interior,vertices])
      for rho in (.78,.9):
       for delta in (.05,.2,.5):
        model=CollectiveDiffusion(d,rho,delta);S=model.solution(test)
        norm0=np.linalg.norm(S,axis=1).max()
        for seed in (1000,1001,1002,1003):
         for method in ('gd','pool256','pool4096','pool10000'):
          rng=np.random.default_rng(seed+1000003);Q=np.empty((d,0));elapsed=0.;selected=[]
          start=time.perf_counter()
          if method!='gd':
            pool=rng.uniform(-1,1,(int(method[4:]),d));sol=model.solution(pool);vals=np.sum(sol*sol,axis=1)
          elapsed+=time.perf_counter()-start
          errs=np.sum(S*S,axis=1)
          curves.append(dict(d=d,rho=rho,delta=delta,seed=seed,method=method,N=0,E=float(np.sqrt(errs.max()))))
          for n in range(d//2):
            start=time.perf_counter()
            if method=='gd':z,_=model.search(Q,rng.uniform(-1,1,d))
            else:z=pool[int(np.argmax(vals))].copy()
            Q=model.append(Q,z);q=Q[:,-1];selected.append(z)
            if method!='gd':vals=np.maximum(vals-(sol@q)**2,0.)
            elapsed+=time.perf_counter()-start
            errs=np.maximum(errs-(S@q)**2,0.)
            curves.append(dict(d=d,rho=rho,delta=delta,seed=seed,method=method,N=n+1,E=float(np.sqrt(errs.max()))))
          E=float(np.linalg.norm(S-(S@Q)@Q.T,axis=1).max());curves[-1]['E']=E
          row=dict(d=d,rho=rho,delta=delta,seed=seed,method=method,N=d//2,E=E,normalized_E=E/norm0,seconds=elapsed,orthogonality=float(np.linalg.norm(Q.T@Q-np.eye(d//2))))
          rows.append(row);print(row,flush=True)
          np.savez_compressed(OUT/f'collective_delta{delta}_r{rho}_d{d}_{method}_s{seed}.npz',Q=Q,selected_parameters=selected)
        save('collective_results.json',rows);save('collective_curves.json',curves)
  print('Collective screen completed.',flush=True)

if __name__=='__main__':main()
