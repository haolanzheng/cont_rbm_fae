"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Targeted checks: ordering invariance, deployment versus projection, truth floor."""
    import os
    os.environ['OPENBLAS_NUM_THREADS']='1'
    import json
    import numpy as np
    from examples.bgk.bgk_temporal import OUT,context,model
    from examples.bgk.bgk_decisive import dump
    from examples.bgk.bgk_strict import StrictROM
    from examples.bgk.boltzmann_bgk_screen import BGKModel,Config,relative_l2
    
    par,truth,_=context();m=model();finer=BGKModel(Config(fom_tol=1e-13,inner_maxiter=320))
    names=['old_pool','old_adam','new_pool','new_adam','old_adam_swapped']
    data={k:json.loads((OUT/f'{k}.json').read_text()) for k in names}
    quality=dict(runs={},order_invariance={},truth_floor=[],inner_solver=[])
    for k,x in data.items():
     E=np.array([d['errors'] for d in x['diagnostics']]);P=np.array([d['projection_errors'] for d in x['diagnostics']])
     quality['runs'][k]=dict(complete=x['complete'],last_rank=x['diagnostics'][-1]['rank'],
      solver_failures=sum(d['inner_stats']['failures'] for d in x['diagnostics']),
      max_projection_minus_deployment=float(np.max(P-E)),min_singular_ratio=min(d['singular_ratio'] for d in x['diagnostics']))
     assert x['complete'] and quality['runs'][k]['solver_failures']==0
     assert np.max(P-E)<1e-9
    a=np.array(data['old_adam']['diagnostics'][-1]['errors']);b=np.array(data['old_adam_swapped']['diagnostics'][-1]['errors'])
    quality['order_invariance']=dict(max_absolute_error_difference=float(np.max(np.abs(a-b))),
     maximum_error_ratio=float(max(b)/max(a)),max_pointwise_relative_error_difference=float(np.max(np.abs(a-b)/np.maximum(a,1e-12))))
    assert abs(max(b)/max(a)-1)<.005
    for name in ['new_adam','new_pool']:
     x=data[name];n=x['diagnostics'][-1]['rank'];ss=np.load(OUT/f'{name}.npz')['snapshots'][:n]
     rom=StrictROM(m,list(ss),solver='scipy',tol=1e-13)
     worst=np.argsort(x['diagnostics'][-1]['errors'])[-3:]
     for i in worst:
      f,info=finer.solve(*par[i]);assert info['converged']
      gap=relative_l2(m,f,truth[i]);old=x['diagnostics'][-1]['errors'][i]
      ans=rom.solve_inner(*par[i]);field=rom.field(ans['y']);err=relative_l2(m,field,f)
      quality['truth_floor'].append(dict(name=name,rank=n,index=int(i),truth_difference=gap,old_error=old,
          strict_solver_and_truth_error=err,relative_error_change=abs(err/old-1)))
     dump(OUT/'BGK_TEMPORAL_QUALITY.json',quality)
    print(json.dumps(quality,indent=2))


if __name__ == "__main__":
    main()
