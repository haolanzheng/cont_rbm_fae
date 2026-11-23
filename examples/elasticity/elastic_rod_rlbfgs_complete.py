#!/usr/bin/env python3
"""Complete-benchmark comparison of sphere R-LBFGS and projected Adam."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np

from examples.elasticity.elastic_rod_complete_test import PaperRod,audit,parameter_summary
from examples.elasticity.elastic_rod_step_sweep import reconstruct,train_continuous
from examples.elasticity.random_rod_greedy import initial_basis
from contgreedy.sampling import rlbfgs_ascent


def train_rlbfgs(problem,rounds,steps,seed,trust_fraction=.25):
    rng=np.random.default_rng(seed);v=initial_basis(problem);selected=[]
    search=0.;fom=0.;gevals=0;vevals=0
    for it in range(rounds):
        z0=problem.shell_samples(rng,1)[0]
        fg=lambda z:problem.indicator_grad(z,v)
        fo=lambda z:float(problem.indicator_batch(z,v)[0])
        start=fo(z0);tic=time.perf_counter()
        z,val,info=rlbfgs_ascent(fg,fo,z0,problem.radius,steps,
          initial_trust_fraction=trust_fraction)
        rs=time.perf_counter()-tic;search+=rs
        gevals+=info['gradient_evaluations'];vevals+=info['value_evaluations']
        tic=time.perf_counter();u=problem.full_trajectory(z)
        err=problem.true_error(z,v,u);v=problem.enrich(v,u,1);fom+=time.perf_counter()-tic
        selected.append({'round':it+1,'indicator_squared':float(val),
          'start_indicator_squared':float(start),'true_error_before_enrichment':err,
          'search_seconds':rs,'theta':z.tolist(),**parameter_summary(problem,z),
          'gradient_evaluations':info['gradient_evaluations'],
          'value_evaluations':info['value_evaluations'],
          'accepted_steps':sum(bool(q['accepted']) for q in info['trace'])})
        print('rlbfgs',steps,'round',it+1,'indicator',val,'seconds',rs,flush=True)
    return {'basis':v,'selected':selected,'search_seconds':search,
      'fom_enrich_seconds':fom,'training_seconds':search+fom,
      'gradient_evaluations':gevals,'value_evaluations':vevals}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,default=Path('elastic_rod_complete_result.json'))
    ap.add_argument('--audit',type=int,default=2000);ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--output',type=Path,default=Path('elastic_rod_rlbfgs_complete.json'))
    a=ap.parse_args();p=PaperRod();old=json.loads(a.baseline.read_text());trained={}
    for name in ['resampled_random_100','fixed_random_10000','continuous_adam100']:
        rows=old['runs'][name]['selected'];v,rebuild=reconstruct(p,rows)
        trained[name]={'basis':v,'selected':rows,
          'search_seconds':sum(float(q['search_seconds']) for q in rows),
          'fom_enrich_seconds':rebuild,'training_seconds':old['runs'][name]['training_seconds']}
    for steps in [10,20]:
        run=train_continuous(p,8,steps,.96,a.seed);run['selected']=run.pop('history')
        run['training_seconds']=run['total_seconds'];trained[f'adam_T{steps}']=run
    for steps in [5,10,20]:trained[f'rlbfgs_T{steps}']=train_rlbfgs(p,8,steps,a.seed)
    result,counts=audit(p,trained,a.audit,a.audit,seed=20260918)
    for name,row in result.items():
        row['search_seconds']=trained[name]['search_seconds']
        row['fom_enrich_seconds']=trained[name]['fom_enrich_seconds']
        if 'gradient_evaluations' in trained[name]:
            row['gradient_evaluations']=trained[name]['gradient_evaluations']
            row['value_evaluations']=trained[name]['value_evaluations']
    obj={'protocol':{'benchmark':'same complete 30D random-elasticity rod',
      'initial_rank':2,'enrichments':8,'final_rank':10,'audit_gaussian':a.audit,
      'audit_shell':a.audit,'single_start':True,
      'constraint_update':'sphere exponential map; exact parallel transport; no clipping/projection',
      'rlbfgs_objective':'log(residual indicator squared)','rlbfgs_memory':7,
      'initial_trust_fraction':.25,'adam_lr':.96},'audit_counts':counts,
      'results':result,'selected':{n:r['selected'] for n,r in trained.items()}}
    a.output.write_text(json.dumps(obj,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
