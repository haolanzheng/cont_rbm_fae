"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    #!/usr/bin/env python3
    """Ten-seed confirmation for the saturated-magnet fixed-budget methods."""
    import json
    from pathlib import Path
    import numpy as np
    from examples.nonlinear_elliptic.nonlinear_pb_magnet_continuous_greedy import (SaturatingMagnet,train,
                                                         test_design,precompute_truth)
    
    OUT=Path('nonlinear_pde_results_magnet_saturated_v1')
    source=json.loads((OUT/'saturating_magnetostatic_circuit.json').read_text())
    names=['continuous_lr0.03','pool_uniform_100','pool_boundary_100']
    rows=[{k:r[k] for k in ['method','seed','max_relative_error','mean_relative_error',
                             'q99_relative_error','training_seconds','objective_calls']}
          for r in source['rows'] if r['method'] in names and r['seed'] in [11,22,33]]
    model=SaturatingMagnet(grid=28)
    Z,design=test_design(model,seed=20260919)
    truths,truth_seconds=precompute_truth(model,Z)
    configs=[('continuous_lr0.03','continuous',.03),
             ('pool_uniform_100','pool_uniform',None),
             ('pool_boundary_100','pool_boundary',None)]
    for seed in [44,55,66,77,88,99,110]:
        for name,kind,lr in configs:
            print(name,seed,flush=True)
            tr=train(model,kind,10,seed,lr=.1 if lr is None else lr,pool_size=100)
            errors=np.array([model.truth_error(z,tr.V,u) for z,u in zip(Z,truths)])
            rows.append({'method':name,'seed':seed,'max_relative_error':float(errors.max()),
                         'mean_relative_error':float(errors.mean()),
                         'q99_relative_error':float(np.quantile(errors,.99)),
                         'training_seconds':tr.seconds,
                         'objective_calls':sum(r['calls'] for r in tr.records)})
    summary={}
    for name in names:
        a=np.array([r['max_relative_error'] for r in rows if r['method']==name])
        summary[name]={'runs':len(a),'median':float(np.median(a)),'mean':float(a.mean()),
                       'min':float(a.min()),'max':float(a.max()),
                       'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75))}
    cont={r['seed']:r['max_relative_error'] for r in rows if r['method']==names[0]}
    for opponent in names[1:]:
        other={r['seed']:r['max_relative_error'] for r in rows if r['method']==opponent}
        ratio=np.array([cont[s]/other[s] for s in sorted(cont)])
        summary['continuous_vs_'+opponent]={'continuous_win_count':int(np.sum(ratio<1)),
            'paired_runs':len(ratio),'median_error_ratio':float(np.median(ratio)),
            'ratios':ratio.tolist()}
    data={'case':'saturated magnet ten-seed fixed-budget confirmation','grid':28,'rank':10,
          'bsat':model.bsat,'adam_steps':100,'single_start':True,'test_design':design,
          'truth_used_in_selection':False,'truth_seconds':truth_seconds,
          'physics_audit':source['physics_audit'],'rows':rows,'summary':summary}
    (OUT/'magnet_10seed_confirmation.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
