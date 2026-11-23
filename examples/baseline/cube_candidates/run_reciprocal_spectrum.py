"""Check whether a resolved dominant spectral subspace changes full-RB results."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json,time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .reciprocal import ReciprocalDiffusion
from .run_reciprocal import OUT,save

def main():
  rows=[];curves=[]
  settings=dict(dimensions=[16,32],spectra=['two_level_.1','two_level_.01','groups4_.5'],kappa=[0.,.8,.95],seeds=[1000,1001,1002,1003],alpha=.6,T=80,N='d/2',test='16384 Sobol interior +32768 uniform vertices +all signed Hadamard rows',methods=['gd','pool256','pool4096','pool10000'],purpose='spectrum screening, no width identity asserted')
  save('reciprocal_spectrum_settings.json',settings)
  with threadpool_limits(limits=1):
   for d in settings['dimensions']:
    rng=np.random.default_rng(202611054+d)
    test=np.vstack([2*qmc.Sobol(d,scramble=True,seed=202611054+d).random_base2(14)-1,rng.choice([-1.,1.],size=(32768,d))])
    for spectrum in settings['spectra']:
     for kappa in settings['kappa']:
      model=ReciprocalDiffusion(d,.78,kappa)
      if spectrum.startswith('two_level'):
       model.weights=np.r_[np.ones(d//2),np.full(d//2,float(spectrum.split('_')[-1]))]
      else:model.weights=.5**(np.arange(d)//4)
      S=model.solution(np.vstack([test,model.corners(),-model.corners()]));norm0=np.linalg.norm(S,axis=1).max()
      for seed in settings['seeds']:
       for method in settings['methods']:
        rng=np.random.default_rng(seed+1000003);Q=np.empty((d,0));selected=[];elapsed=0.;errs=np.sum(S*S,axis=1)
        start=time.perf_counter()
        if method!='gd':
         pool=rng.uniform(-1,1,(int(method[4:]),d));sol=model.solution(pool);vals=np.sum(sol*sol,axis=1)
        elapsed+=time.perf_counter()-start
        for n in range(d//2):
         start=time.perf_counter()
         z=model.search(Q,rng.uniform(-1,1,d))[0] if method=='gd' else pool[int(np.argmax(vals))].copy()
         Q=model.append(Q,z);q=Q[:,-1];selected.append(z)
         if method!='gd':vals=np.maximum(vals-(sol@q)**2,0.)
         elapsed+=time.perf_counter()-start
         errs=np.maximum(errs-(S@q)**2,0.)
         curves.append(dict(d=d,spectrum=spectrum,kappa=kappa,seed=seed,method=method,N=n+1,E=float(np.sqrt(errs.max()))))
        E=float(np.linalg.norm(S-(S@Q)@Q.T,axis=1).max());curves[-1]['E']=E
        row=dict(d=d,spectrum=spectrum,kappa=kappa,seed=seed,method=method,N=d//2,E=E,normalized_E=E/norm0,seconds=elapsed,orthogonality=float(np.linalg.norm(Q.T@Q-np.eye(d//2))))
        rows.append(row);print(row,flush=True)
        np.savez_compressed(OUT/f'reciprocal_spectrum_{spectrum}_k{kappa}_d{d}_{method}_s{seed}.npz',Q=Q,selected_parameters=selected,weights=model.weights)
      save('reciprocal_spectrum_results.json',rows);save('reciprocal_spectrum_curves.json',curves)
  print('Reciprocal spectrum screen completed.',flush=True)
if __name__=='__main__':main()
