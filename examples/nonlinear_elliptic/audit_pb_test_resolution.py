"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    #!/usr/bin/env python3
    """Check whether the 25x25 PB deployment grid misses larger errors."""
    import json
    from pathlib import Path
    import numpy as np
    from examples.nonlinear_elliptic.nonlinear_pb_magnet_continuous_greedy import PoissonBoltzmann,train,precompute_truth
    
    OUT=Path('nonlinear_pde_results_pb_confirmation')
    model=PoissonBoltzmann(grid=36)
    a=np.linspace(-1,1,41)
    Z=np.stack(np.meshgrid(a,a,indexing='ij'),axis=-1).reshape(-1,2)
    truths,seconds=precompute_truth(model,Z)
    rows=[]
    configs=[('continuous_lr0.3','continuous',.3),
             ('pool_uniform_100','pool_uniform',.1),
             ('pool_boundary_100','pool_boundary',.1)]
    for seed in [11,55,99]:
        for name,kind,lr in configs:
            print(name,seed,flush=True)
            tr=train(model,kind,12,seed,lr=lr,pool_size=100)
            e=np.array([model.truth_error(z,tr.V,u) for z,u in zip(Z,truths)])
            rows.append({'method':name,'seed':seed,'max_relative_error':float(e.max()),
                         'mean_relative_error':float(e.mean()),'worst_z':Z[e.argmax()].tolist()})
    data={'case':'PB deployment-test resolution audit','grid':36,'rank':12,
          'parameter_tensor_grid':'41x41','test_size':len(Z),'truth_seconds':seconds,
          'truth_used_in_selection':False,'rows':rows}
    (OUT/'pb_test_resolution_41x41.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
