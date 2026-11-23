"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Check complete checkpoints and independently resolve reported worst points."""
    import os
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    import json
    import numpy as np
    from examples.bgk.bgk_decisive import OUT,SEEDS,METHODS,points,dump
    from examples.bgk.bgk_strict import StrictROM
    from examples.bgk.boltzmann_bgk_screen import BGKModel,Config,relative_l2
    
    m=BGKModel(Config(fom_tol=1e-11,inner_maxiter=200));checks=[];repairs=[]
    tasks=[(s,method) for s in SEEDS for method in METHODS]+[(901,'original_adam')]
    for seed,method in tasks:
     p=OUT/f'{method}_{seed}.json';data=json.loads(p.read_text());assert data['complete']
     snap=OUT/f'{method}_{seed}.npz'
     try:
      ss=np.load(snap)['snapshots'];assert ss.shape==(15,100,32,32);assert np.isfinite(ss).all()
     except (OSError,ValueError,EOFError,AssertionError):
      # Selections and FOM settings are authoritative; reconstruct any interrupted
      # non-atomic binary checkpoint without rerunning or changing selections.
      mus=[[0.,5.5]]+[r['mu'] for r in data['selections']];ss=[]
      for mu in mus:
       f,info=m.solve(*mu);assert info['converged'];ss.append(f)
      ss=np.stack(ss);np.savez_compressed(snap,snapshots=ss);repairs.append(snap.name)
     singular=np.linalg.svd(ss.reshape(15,-1),compute_uv=False)
     checks.append(dict(seed=seed,method=method,smallest_relative_singular_value=float(singular[-1]/singular[0]),
       selection_inner_failures=sum(x['inner_stats']['failures'] for x in data['selections']),
       final_inner_failures=data['final_inner_stats']['failures']))
     assert singular[-1]/singular[0]>1e-12
     print('checked',seed,method,flush=True)
    par,groups=points();truth=np.load(OUT/'truth.npy',mmap_mode='r');solver=[]
    for method in METHODS+['original_adam']:
     d=json.loads((OUT/f'{method}_901.json').read_text());ss=list(np.load(OUT/f'{method}_901.npz')['snapshots'])
     mu=np.asarray(d['diagnostics'][-1]['all']['argmax_mu']);index=int(np.argmin(np.linalg.norm(par-mu,axis=1)))
     a=StrictROM(m,ss);b=StrictROM(m,ss,solver='scipy',tol=1e-13)
     x=a.solve_inner(*mu);z=b.solve_inner(*mu)
     delta=relative_l2(m,a.field(x['y']),b.field(z['y']));err=relative_l2(m,a.field(x['y']),truth[index])
     row=dict(method=method,mu=mu.tolist(),independent_solver_state_difference=delta,
       reported_error=d['diagnostics'][-1]['all']['max'],recomputed_error=err,
       value_relative_difference=abs(x['value']/z['value']-1))
     solver.append(row);assert delta<1e-9;assert abs(err-row['reported_error'])<1e-10
    dump(OUT/'BGK_DECISIVE_QUALITY.json',dict(checkpoints=checks,reconstructed_checkpoint_files=repairs,
      independent_final_solver_checks=solver,all_passed=True))
    print('all checks passed',flush=True)


if __name__ == "__main__":
    main()
