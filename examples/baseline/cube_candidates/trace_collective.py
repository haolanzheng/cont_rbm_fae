"""First-round trajectory diagnostics for the common-amplitude candidate.

4096 genuine independent uniform initial points, not a search pool. Each
trajectory performs the same normalized, clipped GD. These are empirical
probabilities, not deterministic theoretical lower bounds. Pool cap bounds
are analytic. Existing full-RB experiments are not rerun or overwritten.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parents[3]/'runs/cube_candidate_screen_20261104'

def main():
    rows=[];settings=dict(d=[16,32,64],rho=.9,delta=.01,alpha=.6,gamma=.9,independent_initial_points=4096,seed=202611060,checkpoints=[0,10,20,40,80,160,320],device='CPU float64, one BLAS thread',returned='best objective among all visited points',interval='pointwise 95% Clopper-Pearson; not simultaneous confidence',error='relative distance from w=B1 to the span of the returned snapshot')
    (OUT/'collective_trajectory_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    with threadpool_limits(limits=1):
      for d in settings['d']:
        h=.9**(2*np.arange(d))/d;S=h.sum();ell=np.linalg.norm(h);delta=.01;gamma=.9
        rng=np.random.default_rng(settings['seed']+d);Z=rng.uniform(-1,1,(4096,d));initial=Z.copy()
        best=Z.copy();bestJ=np.zeros(4096);nonmonotone=0;prev=None
        for t in range(321):
          a=1+delta-Z.mean(axis=1);numerator=(Z*Z)@h;J=numerator/a**2
          improve=J>bestJ;best[improve]=Z[improve];bestJ[improve]=J[improve]
          if prev is not None:nonmonotone+=int(np.sum(J<prev-1e-11*np.maximum(1,prev)))
          if t in settings['checkpoints']:
            reached=bestJ>=gamma**2*S/delta**2;k=int(reached.sum());m=len(Z)
            ci=[0. if k==0 else float(beta.ppf(.025,k,m-k+1)),1. if k==m else float(beta.ppf(.975,k+1,m-k))]
            x=best*np.sqrt(h);w=np.sqrt(h);normx2=np.sum(x*x,axis=1)
            coeff=(x@w)/normx2;residual=w[None,:]-coeff[:,None]*x
            defect=np.linalg.norm(residual,axis=1)/np.sqrt(S)
            eps=delta*(1/gamma-1);s=d*eps/2
            logcap=d*np.log10(s)-math.lgamma(d+1)/np.log(10)
            row=dict(d=d,T=t,q_empirical=k/m,hits=k,samples=m,q_ci95=ci,exact_corner_fraction=float(np.mean(np.all(best==1.,axis=1))),direction_within_1percent=float(np.mean(defect<=.01)),direction_defect_quantiles=np.quantile(defect,[.1,.5,.9,.99]).tolist(),mean_deficit_quantiles=np.quantile(1-best.mean(axis=1),[.1,.5,.9,.99]).tolist(),sqrt_relative_objective_quantiles=np.quantile(np.sqrt(bestJ*delta**2/S),[.1,.5,.9,.99]).tolist(),log10_pool_single_upper=float(logcap),log10_pool_10000_upper=float(logcap+4),log10_pool_matched_upper=float(logcap+np.log10(t+1)),observed_objective_decreases=nonmonotone)
            rows.append(row);print(row,flush=True)
            np.savez_compressed(OUT/f'collective_trajectory_d{d}_T{t}.npz',initial=initial,returned=best,J=bestJ,direction_defect=defect)
          if t<320:
            grad=2*Z*h/a[:,None]**2+2*(J/(d*a))[:,None]
            Z=np.clip(Z+.6*grad/np.sqrt(ell**2+np.sum(grad*grad,axis=1,keepdims=True)),-1,1)
          prev=J
        (OUT/'collective_trajectory_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('First-round trajectory diagnostics completed.',flush=True)

if __name__=='__main__':main()
