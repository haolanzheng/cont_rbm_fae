"""Evaluate analytic mechanism identities and first-round probability bounds."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,math
import numpy as np
from .collective import CollectiveDiffusion

OUT=Path(__file__).resolve().parents[3]/'runs/cube_candidate_screen_20261104'

def main():
    existing=json.loads((OUT/'collective_refine_results.json').read_text())
    rows=[];summary=[];prob=[]
    for x in existing:
        if x['rho']!=.9 or x['delta']!=.01:continue
        d=x['d'];m=CollectiveDiffusion(d,.9,.01)
        Q=np.load(OUT/f"collective_refine_delta0.01_r0.9_d{d}_{x['method']}_s{x['seed']}.npz")['Q']
        A=np.diag(m.weights)-Q@(Q.T*m.weights)
        b=float(np.linalg.norm(A@np.ones(d)));s=float(np.linalg.norm(A,2))
        upper=max(b/.01,(b+2*s)/(.01+2/d))
        rows.append(dict(d=d,method=x['method'],seed=x['seed'],N=x['N'],E=x['E_augmented'],corner_error=b/.01,residual_operator_norm=s,analytic_global_upper=upper,corner_dominance_sufficient=bool(b>=d*.01*s)))
    for d in (16,32,64):
      for method in ('gd','pool256','pool4096','pool10000'):
        r=[x for x in rows if x['d']==d and x['method']==method]
        summary.append(dict(d=d,method=method,median_E=float(np.median([x['E'] for x in r])),median_upper=float(np.median([x['analytic_global_upper'] for x in r])),maximum_upper=max(x['analytic_global_upper'] for x in r),minimum_E=min(x['E'] for x in r),median_corner_error=float(np.median([x['corner_error'] for x in r]))))
      h=.9**(2*np.arange(d))/d;p=h/h.sum();gamma=.9;delta=.01
      splus=d*delta*(1/gamma-1)
      assert splus<=2
      logU=d*np.log(splus/2)-math.lgamma(d+1)
      logL=logU-np.sum(np.log1p(d*delta*p/gamma))
      for M in (1,81,10000):
        poolL=-np.expm1(M*np.log1p(-np.exp(logL)))
        poolU=-np.expm1(M*np.log1p(-np.exp(logU)))
        prob.append(dict(d=d,M=M,log10_lower=float(np.log10(poolL)),log10_upper=float(np.log10(poolU)),lower_over_upper=float(poolL/poolU)))
    # Independent numerical check of the exact identity, for random V and
    # projected steps of widely differing positive sizes. No objective Hessian
    # or small-step approximation is used in this identity.
    rng=np.random.default_rng(202611061);errs=[];lower_violation=[]
    for d in (4,16,32):
      m=CollectiveDiffusion(d,.9,.01)
      for _ in range(100):
        Q,_=np.linalg.qr(rng.normal(size=(d,d//2)))
        A=np.diag(m.weights)-Q@(Q.T*m.weights);H=A.T@A
        z=rng.uniform(-1,1,d);J,g=m.objective(H,z,True)
        eta=10**rng.uniform(-4,2);zp=np.clip(z+eta*g,-1,1);hstep=zp-z
        a=1.01-z.mean();ap=1.01-zp.mean();r=A@z/a;rp=A@zp/ap
        lhs=float(rp@rp-r@r);rhs=float(a/ap*(g@hstep)+np.sum((rp-r)**2))
        errs.append(abs(lhs-rhs)/max(1,abs(lhs),abs(rhs)))
        lower_violation.append(max(0.,float(a/(ap*eta)*(hstep@hstep)-lhs))/max(1,abs(lhs)))
    validation=dict(identity_max_scaled_error=max(errs),growth_bound_max_scaled_violation=max(lower_violation),cases=len(errs))
    for name,data in [('collective_mechanism_per_seed.json',rows),('collective_mechanism_summary.json',summary),('collective_pool_bounds.json',prob),('collective_identity_validation.json',validation)]:
      (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(validation))
    for d in (16,32,64):
      gd=next(x for x in summary if x['d']==d and x['method']=='gd')
      pool=next(x for x in summary if x['d']==d and x['method']=='pool10000')
      print(dict(d=d,gd=gd,pool=pool))

if __name__=='__main__':main()
