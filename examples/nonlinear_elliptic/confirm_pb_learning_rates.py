"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    #!/usr/bin/env python3
    """Complete ten-seed PB learning-rate audit and merge with confirmation data."""
    import json
    from pathlib import Path
    import numpy as np
    from examples.nonlinear_elliptic.nonlinear_pb_magnet_continuous_greedy import (PoissonBoltzmann,train,
                                                         test_design,precompute_truth)
    
    OUT=Path('nonlinear_pde_results_pb_confirmation')
    path=OUT/'pb_10seed_confirmation.json'
    base=json.loads(path.read_text())
    model=PoissonBoltzmann(grid=36)
    Z,design=test_design(model,seed=20260919)
    truths,truth_seconds=precompute_truth(model,Z)
    rows=base['rows']
    done={(r['method'],r['seed']) for r in rows}
    for lr in [.03,.1]:
        name=f'continuous_lr{lr:g}'
        for seed in [11,22,33,44,55,66,77,88,99,110]:
            if (name,seed) in done:continue
            print(name,seed,flush=True)
            tr=train(model,'continuous',12,seed,lr=lr)
            errors=np.array([model.truth_error(z,tr.V,u) for z,u in zip(Z,truths)])
            rows.append({'method':name,'seed':seed,'max_relative_error':float(errors.max()),
                         'mean_relative_error':float(errors.mean()),
                         'q99_relative_error':float(np.quantile(errors,.99)),
                         'training_seconds':tr.seconds,
                         'objective_calls':sum(r['calls'] for r in tr.records),
                         'selected_z':[r['z'] for r in tr.records]})
    summary={}
    for name in sorted({r['method'] for r in rows}):
        a=np.array([r['max_relative_error'] for r in rows if r['method']==name])
        summary[name]={'runs':len(a),'median':float(np.median(a)),'mean':float(a.mean()),
                       'min':float(a.min()),'max':float(a.max()),
                       'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75))}
    for lr in [.03,.1,.3]:
        name=f'continuous_lr{lr:g}'
        cont={r['seed']:r['max_relative_error'] for r in rows if r['method']==name}
        for opponent in ['pool_uniform_100','pool_boundary_100']:
            other={r['seed']:r['max_relative_error'] for r in rows if r['method']==opponent}
            ratio=np.array([cont[s]/other[s] for s in sorted(cont)])
            summary[name+'_vs_'+opponent]={'continuous_win_count':int(np.sum(ratio<1)),
                 'paired_runs':len(ratio),'median_error_ratio':float(np.median(ratio))}
    base['rows']=rows;base['summary']=summary;base['additional_truth_seconds']=truth_seconds
    path.write_text(json.dumps(base,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
