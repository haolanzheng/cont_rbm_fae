"""Initial RB screen of the Hadamard-mixed linear diffusion candidate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
from examples.baseline.cube.test_set import make_vertex_test_set
from .rotated import RotatedDiffusion,verify

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/cube_candidate_screen_20261104'

def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    rows=[];curves=[]
    save('rotated_validation.json',verify())
    save('rotated_settings.json',dict(dimensions=[16,32,64],rho=[.6,.78,.9],seeds=list(range(1000,1004)),alpha=.6,T=80,device='CPU',threads=1,test='all antipodal vertices at d16; at larger d65536 fixed uniform vertices plus all modal Hadamard vertices',methods=['gd','pool256','pool4096','pool10000','vertex4096'],N='d/2',width='not asserted; rho**N is the exact error of the coordinate-mode reference space'))
    with threadpool_limits(limits=1):
      for d in (16,32,64):
        test,exact=make_vertex_test_set(d,65536,seed=202611051)
        for rho in (.6,.78,.9):
          model=RotatedDiffusion(d,rho)
          Z=np.vstack([test,model.modal_vertices()]);S=model.solution(Z)
          initerr=np.sum(S*S,axis=1)
          for seed in range(1000,1004):
            for method in ('gd','pool256','pool4096','pool10000','vertex4096'):
              rng=np.random.default_rng(seed+1000003);Q=np.empty((d,0));errs=initerr.copy();elapsed=0.;selected=[]
              start=time.perf_counter()
              if method!='gd':
                M=int(method.replace('pool','').replace('vertex',''))
                pool=(rng.choice([-1.,1.],size=(M,d)) if method.startswith('vertex') else rng.uniform(-1,1,(M,d)))
                solutions=model.solution(pool);vals=np.sum(solutions*solutions,axis=1)
              elapsed+=time.perf_counter()-start
              curves.append(dict(d=d,rho=rho,seed=seed,method=method,N=0,E=float(np.sqrt(errs.max()))))
              for n in range(d//2):
                start=time.perf_counter()
                if method=='gd':
                  z,stats=model.search(Q,rng.uniform(-1,1,d))
                else:z=pool[int(np.argmax(vals))].copy()
                Qnew,novelty=model.append(Q,z)
                if len(Qnew.T)==len(Q.T):raise AssertionError('Dependent selected snapshot')
                Q=Qnew;q=Q[:,-1]
                if method!='gd':vals=np.maximum(vals-(solutions@q)**2,0.)
                elapsed+=time.perf_counter()-start
                selected.append(z)
                errs=np.maximum(errs-(S@q)**2,0.)
                curves.append(dict(d=d,rho=rho,seed=seed,method=method,N=n+1,E=float(np.sqrt(errs.max()))))
              # Stable direct check avoids cancellation in very small errors.
              residual=S-(S@Q)@Q.T
              e=float(np.max(np.linalg.norm(residual,axis=1)))
              curves[-1]['E']=e
              defect=float(np.linalg.norm(Q.T@Q-np.eye(d//2),ord=2))
              row=dict(d=d,rho=rho,seed=seed,method=method,N=d//2,E=e,reference_error=rho**(d//2),seconds=elapsed,orthogonality=defect,test_exhaustive=exact)
              rows.append(row)
              np.savez_compressed(OUT/f'rotated_r{rho}_d{d}_{method}_s{seed}.npz',Q=Q,selected_parameters=selected)
              print(row,flush=True)
          save('rotated_results.json',rows);save('rotated_curves.json',curves)
    print('Rotated candidate screen completed.',flush=True)

if __name__=='__main__':main()
