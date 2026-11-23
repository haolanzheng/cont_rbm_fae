"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Hold original selected parameters fixed; isolate snapshot truth precision."""
    import os
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    import json
    import numpy as np
    from examples.bgk.bgk_decisive import ROOT,OUT,points,dump,evaluate,stats
    from examples.bgk.bgk_strict import StrictROM
    from examples.bgk.boltzmann_bgk_screen import BGKModel,Config,relative_l2
    
    old=json.loads((ROOT/'mechanism_output/trajectory.json').read_text())
    packed=np.load(ROOT/'mechanism_output/trajectory_snapshots.npz')
    m=BGKModel(Config(fom_tol=1e-11,inner_maxiter=160));par,groups=points();truth=np.load(OUT/'truth.npy',mmap_mode='r')
    answer={}
    for method in ['pool','adam']:
     mus=[[0.,5.5]]+[x['mu'] for x in old[method]['selections']]
     snapshots=[];deltas=[]
     for i,mu in enumerate(mus):
      f,info=m.solve(*mu)
      if not info['converged']:raise RuntimeError('FOM did not converge')
      snapshots.append(f);deltas.append(relative_l2(m,packed[method][i],f))
     answer[method]={'snapshot_relative_changes':deltas}
     for fidelity,ss in [('original',list(packed[method])),('tight',snapshots)]:
      rom=StrictROM(m,ss);diag=evaluate(rom,par,truth,groups);diag['inner_stats']=stats(rom)
      answer[method][fidelity]=diag
      print('precision replay',method,fidelity,diag['all']['max'],flush=True)
     # Independent high-accuracy truth verification at the five hardest points.
     indices=np.argsort(answer[method]['tight']['errors'])[-5:];finer=BGKModel(Config(fom_tol=1e-13))
     checks=[]
     for i in indices:
      f,info=finer.solve(*par[i]);checks.append(dict(mu=par[i].tolist(),converged=info['converged'],
        relative_truth_change=relative_l2(m,truth[i],f)))
     answer[method]['truth_1e13_checks']=checks
     dump(OUT/'snapshot_precision_replay.json',answer)


if __name__ == "__main__":
    main()
