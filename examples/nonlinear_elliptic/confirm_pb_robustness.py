"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    #!/usr/bin/env python3
    """Ten-seed confirmation for the competitive PB methods."""
    import json,time
    from pathlib import Path
    import numpy as np
    from examples.nonlinear_elliptic.nonlinear_pb_magnet_continuous_greedy import (PoissonBoltzmann,train,
                                                         test_design,precompute_truth)
    
    OUT=Path('nonlinear_pde_results_pb_confirmation')
    OUT.mkdir(exist_ok=True)
    model=PoissonBoltzmann(grid=36)
    Z,design=test_design(model,seed=20260919)
    truths,truth_seconds=precompute_truth(model,Z)
    methods=[('continuous_lr0.3','continuous',.3,100),
             ('pool_uniform_100','pool_uniform',None,100),
             ('pool_boundary_100','pool_boundary',None,100)]
    rows=[]
    for seed in [11,22,33,44,55,66,77,88,99,110]:
        for name,kind,lr,pool in methods:
            print(name,seed,flush=True)
            tr=train(model,kind,12,seed,lr=.1 if lr is None else lr,pool_size=pool)
            errors=np.array([model.truth_error(z,tr.V,u) for z,u in zip(Z,truths)])
            rows.append({'method':name,'seed':seed,'max_relative_error':float(errors.max()),
                         'mean_relative_error':float(errors.mean()),
                         'q99_relative_error':float(np.quantile(errors,.99)),
                         'training_seconds':tr.seconds,
                         'objective_calls':sum(r['calls'] for r in tr.records),
                         'selected_z':[r['z'] for r in tr.records]})
            (OUT/'pb_10seed_confirmation.json').write_text(json.dumps(
                {'test_design':design,'truth_seconds':truth_seconds,'rows':rows},indent=2))
    summary={}
    for name,_,_,_ in methods:
        rr=[r for r in rows if r['method']==name]
        a=np.array([r['max_relative_error'] for r in rr])
        summary[name]={'median':float(np.median(a)),'mean':float(a.mean()),
                       'min':float(a.min()),'max':float(a.max()),
                       'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75))}
    cont={r['seed']:r['max_relative_error'] for r in rows if r['method']=='continuous_lr0.3'}
    for opponent in ['pool_uniform_100','pool_boundary_100']:
        other={r['seed']:r['max_relative_error'] for r in rows if r['method']==opponent}
        ratio=np.array([cont[s]/other[s] for s in sorted(cont)])
        summary['continuous_vs_'+opponent]={'continuous_win_count':int(np.sum(ratio<1)),
            'paired_runs':len(ratio),'median_error_ratio':float(np.median(ratio)),
            'ratios':ratio.tolist()}
    data={'case':'PB ten-seed fixed-budget confirmation','grid':36,'rank':12,
          'adam_steps':100,'single_start':True,'test_design':design,
          'truth_used_in_selection':False,'truth_seconds':truth_seconds,
          'rows':rows,'summary':summary}
    (OUT/'pb_10seed_confirmation.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
