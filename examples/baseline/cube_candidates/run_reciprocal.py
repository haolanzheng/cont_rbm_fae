"""Full RB comparison for odd reciprocal forcing; one CPU BLAS thread."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .reciprocal import ReciprocalDiffusion

OUT=Path(__file__).resolve().parents[3]/'runs/cube_candidate_screen_20261104'
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')

def main():
    rows=[];curves=[]
    settings=dict(dimensions=[16,32,64],rho=[.78,.9],kappa=[.5,.8,.95],seeds=[1000,1001,1002,1003],alpha=.6,T=80,N='d/2',normalizer='Frobenius norm of modal residual Gram H_n',test='16384 scrambled Sobol interior + 32768 uniform vertices + positive/negative Hadamard corners',methods=['gd','pool256','pool4096','pool10000'],device='CPU',threads=1,width='not asserted; exact modal greedy E_N=sqrt(d)*rho**N/(1-kappa)')
    save('reciprocal_settings.json',settings)
    model=ReciprocalDiffusion(16,.78,.8);rng=np.random.default_rng(52);z=rng.uniform(-1,1,16)
    Q=np.empty((16,0));Q=model.append(Q,rng.uniform(-1,1,16));H=model.matrix(Q)
    val,g=model.objective(H,z,True);eps=1e-6
    fd=np.array([(model.objective(H,z+eps*np.eye(16)[i])-model.objective(H,z-eps*np.eye(16)[i]))/(2*eps) for i in range(16)])
    save('reciprocal_validation.json',dict(gradient_relative_error=float(np.linalg.norm(g-fd)/np.linalg.norm(fd)),denominator_lower_bound=1-model.kappa))
    with threadpool_limits(limits=1):
      for d in settings['dimensions']:
        rng=np.random.default_rng(202611054+d)
        test=np.vstack([2*qmc.Sobol(d,scramble=True,seed=202611054+d).random_base2(14)-1,rng.choice([-1.,1.],size=(32768,d))])
        for rho in settings['rho']:
          for kappa in settings['kappa']:
            model=ReciprocalDiffusion(d,rho,kappa)
            Z=np.vstack([test,model.corners(),-model.corners()]);S=model.solution(Z)
            norm0=np.linalg.norm(S,axis=1).max()
            for seed in settings['seeds']:
              for method in settings['methods']:
                rng=np.random.default_rng(seed+1000003);Q=np.empty((d,0));elapsed=0.;selected=[]
                start=time.perf_counter()
                if method!='gd':
                    pool=rng.uniform(-1,1,(int(method[4:]),d));sol=model.solution(pool);vals=np.sum(sol*sol,axis=1)
                elapsed+=time.perf_counter()-start
                errs=np.sum(S*S,axis=1)
                curves.append(dict(d=d,rho=rho,kappa=kappa,seed=seed,method=method,N=0,E=float(np.sqrt(errs.max()))))
                for n in range(d//2):
                    start=time.perf_counter()
                    if method=='gd':z,_=model.search(Q,rng.uniform(-1,1,d))
                    else:z=pool[int(np.argmax(vals))].copy()
                    Q=model.append(Q,z);q=Q[:,-1];selected.append(z)
                    if method!='gd':vals=np.maximum(vals-(sol@q)**2,0.)
                    elapsed+=time.perf_counter()-start
                    errs=np.maximum(errs-(S@q)**2,0.)
                    curves.append(dict(d=d,rho=rho,kappa=kappa,seed=seed,method=method,N=n+1,E=float(np.sqrt(errs.max()))))
                E=float(np.linalg.norm(S-(S@Q)@Q.T,axis=1).max());curves[-1]['E']=E
                row=dict(d=d,rho=rho,kappa=kappa,seed=seed,method=method,N=d//2,E=E,normalized_E=E/norm0,seconds=elapsed,orthogonality=float(np.linalg.norm(Q.T@Q-np.eye(d//2))),modal_greedy_reference=float(np.sqrt(d)*rho**(d//2)/(1-kappa)))
                rows.append(row);print(row,flush=True)
                np.savez_compressed(OUT/f'reciprocal_k{kappa}_r{rho}_d{d}_{method}_s{seed}.npz',Q=Q,selected_parameters=selected)
            save('reciprocal_results.json',rows);save('reciprocal_curves.json',curves)
    print('Reciprocal screen completed.',flush=True)

if __name__=='__main__':main()
