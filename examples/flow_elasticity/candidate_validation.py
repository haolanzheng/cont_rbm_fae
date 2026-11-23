#!/usr/bin/env python3
"""Validation audits for the two continuous-greedy PDE candidates."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import scipy.linalg as la

from examples.flow_elasticity.rigorous_candidate_bench import (build_perforated_elasticity,
    elastic_matrices, _k_orthonormal_append, sphere_adam)
from examples.flow_elasticity.step_flow_candidate_bench import (build_step_model, indicator_matrix,
    add_worst_time_snapshot, run_greedy)


def sphere_samples(rng,n,d,batch=20000):
    for i in range(0,n,batch):
        z=rng.normal(size=(min(batch,n-i),d));z/=np.linalg.norm(z,axis=1,keepdims=True)
        yield z


def pq_for_Q(Q,lr,seed=12345,nmc=200000,nstart=200,steps=100):
    rng=np.random.default_rng(seed); lam=float(la.eigvalsh(Q)[-1]); d=Q.shape[0]
    counts={.9:0,.95:0,.99:0}; total=0
    for Z in sphere_samples(rng,nmc,d):
        vals=np.einsum('bi,ij,bj->b',Z,Q,Z)/max(lam,1e-300)
        total+=len(Z)
        for g in counts:counts[g]+=int(np.sum(vals>=g*g))
    p={str(g):counts[g]/total for g in counts}
    ratios=[]
    for _ in range(nstart):
        _,v,_=sphere_adam(Q,rng.normal(size=d),steps=steps,lr=lr)
        ratios.append(math.sqrt(max(v,0)/max(lam,1e-300)))
    ratios=np.asarray(ratios)
    q={str(g):float(np.mean(ratios>=g)) for g in counts}
    pool101={str(g):1-(1-p[str(g)])**101 for g in counts}
    return {'p':p,'q_adam_T100':q,'pool101_success_from_p':pool101,
            'adam_ratio_quantiles':{str(a):float(np.quantile(ratios,a)) for a in [0,.05,.5,.95,1]}}


def audit_elastic():
    model=build_perforated_elasticity(48,12,20);V=np.zeros((model.K.shape[0],0));out=[]
    for n in range(12):
        Q,_,_=elastic_matrices(model,V)
        if n in [0,3,7,11]:
            out.append({'before_round':n+1,**pq_for_Q(Q,.08,seed=800+n)})
        ew,EV=la.eigh(Q);V,_=_k_orthonormal_append(V,model.U@EV[:,-1],model.K)
    return {'metadata':model.metadata,'round_audits':out}


def audit_step():
    model=build_step_model(28,10,d=16,Re=80.,dt=.08,nt=35)
    V=np.zeros((model.M.shape[0],0));out=[]
    for n in range(10):
        Q,_=indicator_matrix(model,V)
        if n in [0,3,7,9]:
            out.append({'before_round':n+1,**pq_for_Q(Q,.06,seed=900+n)})
        ew,EV=la.eigh(Q);V,_,_=add_worst_time_snapshot(model,V,EV[:,-1])
    return {'metadata':model.metadata,'round_audits':out}


def temporal_refinement():
    # Same final time 2.8, half the implicit-Euler step.
    model=build_step_model(40,14,d=16,Re=80.,dt=.04,nt=70)
    runs=[]
    for method in ['adam','power','pool','pool4096','oracle']:
        seeds=range(6) if method!='oracle' else [0]
        for s in seeds:runs.append(run_greedy(model,method,s,N=10,steps=100,lr=.06,pool_M=101))
    return {'metadata':model.metadata,'runs':runs}


def main():
    t=time.time()
    result={'elastic_pq':audit_elastic(),'step_pq':audit_step(),
            'step_time_refinement':temporal_refinement(),
            'wall_time_s':time.time()-t}
    p=Path('candidate_bench_results/validation.json');p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(result,indent=2));print(p)


if __name__=='__main__':main()
