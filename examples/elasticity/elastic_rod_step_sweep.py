#!/usr/bin/env python3
"""Validate the speed/accuracy benefit of short continuous greedy searches.

All methods finish at the same rank.  Learning rates are selected using only
the first-round residual indicator at one common start; FOM test errors never
participate in tuning or selection.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

from contgreedy.sampling.host import adam_ascent
from examples.elasticity.elastic_rod_complete_test import PaperRod, audit, parameter_summary
from examples.elasticity.random_rod_greedy import initial_basis


def reconstruct(problem, rows):
    v=initial_basis(problem); tic=time.perf_counter()
    for row in rows:
        v=problem.enrich(v,problem.full_trajectory(np.asarray(row['theta'])),1)
    return v,time.perf_counter()-tic


def calibrate(problem,steps,rates,seed):
    v=initial_basis(problem)
    z0=problem.shell_samples(np.random.default_rng(seed),1)[0]
    out={}
    for lr in rates:
        fg=lambda z:problem.sphere_indicator_grad(z,v)
        _,val=adam_ascent(fg,z0,problem.project,steps=steps,lr=lr)
        out[str(lr)]=float(val)
    best=float(max(out,key=out.get))
    return best,out


def train_continuous(problem,rounds,steps,lr,seed):
    rng=np.random.default_rng(seed);v=initial_basis(problem)
    search=0.;fom=0.;history=[]
    for it in range(rounds):
        z0=problem.shell_samples(rng,1)[0]
        start_value=float(problem.indicator_batch(z0,v)[0])
        tic=time.perf_counter()
        fg=lambda z:problem.sphere_indicator_grad(z,v)
        z,val=adam_ascent(fg,z0,problem.project,steps=steps,lr=lr)
        round_search=time.perf_counter()-tic;search+=round_search
        tic=time.perf_counter();u=problem.full_trajectory(z)
        selected_error=problem.true_error(z,v,u)
        v=problem.enrich(v,u,1);fom+=time.perf_counter()-tic
        history.append({'round':it+1,'indicator_squared':float(val),
                        'start_indicator_squared':start_value,
                        'true_error_before_enrichment':selected_error,
                        'search_seconds':round_search,'theta':z.tolist(),
                        **parameter_summary(problem,z)})
    return {'basis':v,'search_seconds':search,'fom_enrich_seconds':fom,
            'total_seconds':search+fom,'history':history}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rounds',type=int,default=8)
    ap.add_argument('--seed',type=int,default=11);ap.add_argument('--audit',type=int,default=2000)
    ap.add_argument('--baseline',type=Path,default=Path('elastic_rod_complete_result.json'))
    ap.add_argument('--output',type=Path,default=Path('elastic_rod_step_sweep.json'))
    a=ap.parse_args();problem=PaperRod();old=json.loads(a.baseline.read_text())
    methods={};
    for name in ['fixed_random_100','resampled_random_100','fixed_sobol_100','fixed_random_10000']:
        rows=old['runs'][name]['selected'];v,rebuild=reconstruct(problem,rows)
        methods[name]={'basis':v,'search_seconds':sum(float(r['search_seconds']) for r in rows),
                       'fom_enrich_seconds':rebuild,'total_seconds':old['runs'][name]['training_seconds'],
                       'training_seconds':old['runs'][name]['training_seconds'],'selected':rows,
                       'source':'reconstructed from the published complete-run selections'}

    steps_list=[1,3,5,10,20,50,100]
    rates=[.03,.06,.12,.24,.48,.96]
    calibration={}
    for steps in steps_list:
        lr,table=calibrate(problem,steps,rates,a.seed)
        calibration[str(steps)]={'chosen_lr':lr,'indicator_squared':table}
        print('calibration',steps,lr,max(table.values()),flush=True)
        run=train_continuous(problem,a.rounds,steps,lr,a.seed)
        run['training_seconds']=run['total_seconds'];run['selected']=run.pop('history')
        methods[f'continuous_T{steps}_tuned']=run
        print('trained tuned',steps,run['total_seconds'],flush=True)
        # The fixed-lr track changes only T relative to the original complete bench.
        if steps != 100 or abs(lr-.12)>1e-14:
            fixed=train_continuous(problem,a.rounds,steps,.12,a.seed)
            fixed['training_seconds']=fixed['total_seconds'];fixed['selected']=fixed.pop('history')
            methods[f'continuous_T{steps}_lr012']=fixed
            print('trained fixed-lr',steps,fixed['total_seconds'],flush=True)

    result,audit_counts=audit(problem,methods,a.audit,a.audit,seed=20260916)
    for name,row in result.items():
        row['search_seconds']=float(methods[name]['search_seconds'])
        row['fom_enrich_seconds']=float(methods[name]['fom_enrich_seconds'])
        row['total_seconds']=float(methods[name]['total_seconds'])
    obj={'protocol':{'problem':'paper random-elasticity rod','dimension':30,
         'initial_rank':2,'enrichments':a.rounds,'final_rank':2+a.rounds,
         'continuous_start_per_round':1,'steps':steps_list,
         'lr_rule':'max first-round residual indicator at one common start; no FOM tuning',
         'audit_gaussian':a.audit,'audit_shell':a.audit,
         'audit_rule':'exact complete-bench shared selected set plus one Adam-100 attack per final basis'},
         'calibration':calibration,'audit_counts':audit_counts,'results':result}
    a.output.write_text(json.dumps(obj,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
