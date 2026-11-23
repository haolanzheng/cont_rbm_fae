#!/usr/bin/env python3
"""Learning-rate, iteration-count, spectrum, and base-flow audits."""
import json, math
from pathlib import Path
import numpy as np
import scipy.linalg as la

from examples.flow_elasticity.rigorous_candidate_bench import (build_perforated_elasticity,
    elastic_matrices, _k_orthonormal_append, sphere_adam, sphere_power)
from examples.flow_elasticity.step_flow_candidate_bench import (build_step_model, indicator_matrix,
    add_worst_time_snapshot)


def advance_elastic(model,n=7):
    V=np.zeros((model.K.shape[0],0))
    for _ in range(n):
        Q,_,_=elastic_matrices(model,V);ew,EV=la.eigh(Q)
        V,_=_k_orthonormal_append(V,model.U@EV[:,-1],model.K)
    return elastic_matrices(model,V)[0]


def advance_step(model,n=7):
    V=np.zeros((model.M.shape[0],0))
    for _ in range(n):
        Q,_=indicator_matrix(model,V);ew,EV=la.eigh(Q)
        V,_,_=add_worst_time_snapshot(model,V,EV[:,-1])
    return indicator_matrix(model,V)[0]


def optimizer_audit(Q,lrs,selected_lr,seed=777,nstart=100):
    lam=float(la.eigvalsh(Q)[-1]);rng=np.random.default_rng(seed)
    starts=rng.normal(size=(nstart,Q.shape[0]))
    def summ(a):
        a=np.asarray(a);return {'min':float(a.min()),'q05':float(np.quantile(a,.05)),
                                'median':float(np.median(a)),'q95':float(np.quantile(a,.95)),
                                'max':float(a.max())}
    lrout={}
    for lr in lrs:
        vals=[]
        for z in starts:
            _,v,_=sphere_adam(Q,z,steps=100,lr=lr)
            vals.append(math.sqrt(max(v,0)/max(lam,1e-300)))
        lrout[str(lr)]=summ(vals)
    stepout={}
    for T in [1,2,5,10,20,50,100]:
        va=[];vp=[]
        for z in starts:
            _,v,_=sphere_adam(Q,z,steps=T,lr=selected_lr);va.append(math.sqrt(v/lam))
            _,v,_=sphere_power(Q,z,steps=T);vp.append(math.sqrt(v/lam))
        stepout[str(T)]={'adam':summ(va),'normalized_gradient':summ(vp)}
    ev=la.eigvalsh(Q)[::-1];
    return {'eigenvalue_ratios':[float(x/ev[0]) for x in ev],
            'lr_sweep_T100':lrout,'step_sweep':stepout}


def base_diagnostics(model):
    # Probe downstream lower half.  Vector-probe ordering is component-major.
    xx=np.linspace(.08,3.2,50);yy=np.full_like(xx,-.45);pts=np.vstack((xx,yy))
    vals=np.asarray(model.ub.probes(pts)@model.base_u).reshape(2,-1)
    ux=vals[0]
    return {'probe_y':-0.45,'min_downstream_ux':float(ux.min()),
            'x_at_min_ux':float(xx[np.argmin(ux)]),
            'negative_ux_fraction':float(np.mean(ux<0)),
            'sample_ux':[float(x) for x in ux]}


def main():
    em=build_perforated_elasticity(48,12,20);eq=advance_elastic(em)
    sm=build_step_model(28,10,d=16,Re=80.,dt=.08,nt=35);sq=advance_step(sm)
    result={'elastic_round8':optimizer_audit(eq,[.005,.01,.02,.04,.08,.12,.2],.08),
            'step_round8':optimizer_audit(sq,[.005,.01,.02,.04,.06,.08,.12],.06),
            'step_base':base_diagnostics(sm)}
    p=Path('candidate_bench_results/optimizer_audit.json');p.write_text(json.dumps(result,indent=2));print(p)

if __name__=='__main__':main()
