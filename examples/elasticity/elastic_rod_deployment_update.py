#!/usr/bin/env python3
"""Online-update test on the complete random-elasticity rod benchmark.

The source rank-10 basis is kept.  Deployment moves the traction from the
right boundary to the midpoint.  Each search method receives the same number
of target-condition FOM enrichments and is audited at the same final rank.
"""
from __future__ import annotations

import argparse,json,time
from pathlib import Path
import numpy as np

from contgreedy.sampling.host import adam_ascent
from examples.elasticity.elastic_rod_complete_test import PaperRod,audit,parameter_summary
from examples.elasticity.random_rod_greedy import initial_basis


def source_basis(problem,rows):
    v=initial_basis(problem)
    for row in rows:v=problem.enrich(v,problem.full_trajectory(np.asarray(row['theta'])),1)
    return v


def adapt(problem,v0,method,count,seed,steps=20,lr=.96):
    rng=np.random.default_rng(seed);v=v0.copy();selected=[];search=0.;fom=0.
    fixed=None
    if method=='fixed100':fixed=problem.shell_samples(rng,100)
    if method=='fixed10k':fixed=problem.shell_samples(rng,10000)
    for it in range(count):
        z0=None;tic=time.perf_counter()
        if method.startswith('continuous'):
            z0=problem.shell_samples(rng,1)[0]
            fg=lambda z:problem.sphere_indicator_grad(z,v)
            z,val=adam_ascent(fg,z0,problem.project,steps=steps,lr=lr)
        else:
            cand=problem.shell_samples(rng,100) if method=='resampled100' else fixed
            vals=problem.indicator_batch(cand,v);j=int(np.argmax(vals));z,val=cand[j],vals[j]
        round_search=time.perf_counter()-tic;search+=round_search
        tic=time.perf_counter();u=problem.full_trajectory(z)
        true_before=problem.true_error(z,v,u);v=problem.enrich(v,u,1);fom+=time.perf_counter()-tic
        selected.append({'round':it+1,'indicator_squared':float(val),
          'start_indicator_squared':None if z0 is None else float(problem.indicator_batch(z0,v0)[0]),
          'true_error_before_enrichment':true_before,'search_seconds':round_search,
          'theta':z.tolist(),**parameter_summary(problem,z)})
    return {'basis':v,'selected':selected,'search_seconds':search,
      'fom_enrich_seconds':fom,'training_seconds':search+fom}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,default=Path('elastic_rod_complete_result.json'))
    ap.add_argument('--audit',type=int,default=2000);ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--output',type=Path,default=Path('elastic_rod_deployment_update.json'))
    a=ap.parse_args();old=json.loads(a.baseline.read_text())
    source=PaperRod();rows=old['runs']['continuous_adam100']['selected'];v0=source_basis(source,rows)
    target=PaperRod();target.load[:]=0.;target.load[14]=1./target.h
    all_results={}
    for additions in [2,4]:
        trained={'no_update':{'basis':v0,'selected':[],'search_seconds':0.,
                              'fom_enrich_seconds':0.,'training_seconds':0.}}
        specs=[('fixed100',0,.12),('resampled100',0,.12),('fixed10k',0,.12),
               ('continuous_T10',10,.96),('continuous_T20',20,.96)]
        for name,steps,lr in specs:
            trained[name]=adapt(target,v0,name,additions,a.seed,steps,lr)
            print('add',additions,name,'rank',trained[name]['basis'].shape[1],
                  'seconds',trained[name]['training_seconds'],flush=True)
        result,counts=audit(target,trained,a.audit,a.audit,seed=20260917+additions)
        for name,row in result.items():
            row['search_seconds']=trained[name]['search_seconds']
            row['fom_enrich_seconds']=trained[name]['fom_enrich_seconds']
        all_results[str(additions)]={'audit_counts':counts,'results':result,
          'selected':{n:r['selected'] for n,r in trained.items()}}
    obj={'protocol':{'source_model':'complete 30D random-elasticity rod, right-end traction',
      'source_rank':10,'deployment_change':'same rod/prior/integrator; point traction moved to node 15 (x=0.5)',
      'added_basis_counts':[2,4],'equal_final_rank':True,'audit_gaussian':a.audit,
      'audit_shell':a.audit,'continuous_starts_per_update':1,
      'continuous_steps':[10,20]},'updates':all_results}
    a.output.write_text(json.dumps(obj,indent=2));print(json.dumps(obj['updates'],indent=2))


if __name__=='__main__':main()
