"""Small full-RB screen of exact, uniformly coercive layered diffusion."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .layered import LayeredDiffusion

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/cube_candidate_screen_20261104'

def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def normalizer(model,Q):
    # At z=shift, u0 belongs to every trained space. The exact residual
    # Jacobian is -(I-QQ^T) times columns A_i u0 / a0.
    D=np.zeros((2*model.d,model.d))
    for i in range(model.d):
        D[2*i,i]=model.delta/(model.a0*np.sqrt(2*model.d))
        D[2*i+1,i]=-D[2*i,i]
    D-=Q@(Q.T@D)
    return max(float(np.linalg.norm(D.T@D,'fro')),1e-14)

def gd(fixed,z,ell,T=80,alpha=.6):
    best=z.copy();bestvalue=-np.inf
    for t in range(T+1):
        value,g=fixed.objective(z,gradient=True)
        if value>bestvalue:bestvalue=value;best=z.copy()
        if t<T:
            g=g/ell
            z=np.clip(z+alpha*g/np.sqrt(1+g@g),-1,1)
    return best,float(bestvalue)

def main():
  OUT.mkdir(parents=True,exist_ok=True);rows=[];curves=[]
  save('layered_settings.json',dict(dimensions=[8,16],delta=[.4,.55],shift=.75,a0=1,seeds=list(range(1000,1004)),T=80,alpha=.6,N='d',initial='real center snapshot z_i=.75',normalizer='Frobenius norm of exact residual-Jacobian Gram at center z_i=.75; fixed within each search',test='4096 scrambled Sobol interior + all vertices for d8 /4096 uniform vertices for d16 + both all-same vertices and center',methods=['gd','pool256','pool4096','pool10000'],device='CPU',threads=1,error='actual Galerkin maximum reference energy error on shared test; parameter-dependent energy also recorded',width='unknown; no width ratio asserted'))
  with threadpool_limits(limits=1):
    for d in (8,16):
      interior=2*qmc.Sobol(d,scramble=True,seed=202611052+d).random_base2(12)-1
      rng=np.random.default_rng(202611053+d)
      if d==8:
        k=np.arange(2**d);vertices=2*((k[:,None]>>np.arange(d))&1)-1
      else:vertices=rng.choice([-1.,1.],size=(4096,d))
      test=np.vstack([interior,vertices,-np.ones((1,d)),np.ones((1,d)),np.full((1,d),.75)])
      np.savez_compressed(OUT/f'layered_test_d{d}.npz',parameters=test)
      for delta in (.4,.55):
        model=LayeredDiffusion(d,delta=delta)
        for seed in range(1000,1004):
          for method in ('gd','pool256','pool4096','pool10000'):
            rng=np.random.default_rng(seed+1000003);Q=model.initial_basis();elapsed=0.;selected=[]
            if method!='gd':
              start=time.perf_counter();pool=rng.uniform(-1,1,(int(method[4:]),d));elapsed+=time.perf_counter()-start
            for n in range(1,d+1):
              start=time.perf_counter();fixed=model.prepare(Q);ell=normalizer(model,Q);elapsed+=time.perf_counter()-start
              error=float(fixed.errors(test).max())
              curves.append(dict(d=d,delta=delta,seed=seed,method=method,N=n,E=error))
              if n==d:break
              start=time.perf_counter()
              if method=='gd':z,value=gd(fixed,rng.uniform(-1,1,d),ell)
              else:
                values=fixed.objective(pool);idx=int(np.argmax(values));z=pool[idx].copy();value=float(values[idx])
              Qnew=model.append_basis(Q,z)
              if Qnew.shape[1]!=n+1:raise AssertionError('Dependent selected snapshot')
              Q=Qnew;elapsed+=time.perf_counter()-start
              selected.append(dict(z=z.tolist(),J=value,normalizer=ell))
            er=float(fixed.errors(test,norm='energy').max())
            row=dict(d=d,delta=delta,seed=seed,method=method,N=d,E=error,energy_error=er,seconds=elapsed,coefficient_min=model.coefficient_min,coefficient_max=model.coefficient_max,orthogonality=float(np.linalg.norm(Q.T@Q-np.eye(d),ord=2)))
            rows.append(row);print(row,flush=True)
            np.savez_compressed(OUT/f'layered_delta{delta}_d{d}_{method}_s{seed}.npz',Q=Q,selected_parameters=np.array([v['z'] for v in selected]))
          save('layered_results.json',rows);save('layered_curves.json',curves)
    print('Layered screen completed.',flush=True)

if __name__=='__main__':main()
