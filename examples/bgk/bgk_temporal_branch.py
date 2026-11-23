"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Advance the first later interior snapshot by one round at the same prefix."""
    import os
    os.environ['OPENBLAS_NUM_THREADS']='1'
    import json
    import numpy as np
    from examples.bgk.bgk_temporal import OUT,context,model
    from examples.bgk.bgk_decisive import evaluate,dump
    from examples.bgk.bgk_strict import StrictROM
    
    m=model();par,truth,groups=context()
    sn=np.load(OUT/'old_adam.npz');ss=sn['snapshots'];mus=sn['mus']
    # Original enrichment at N=8 uses snapshot index 8 on the boundary.
    # Its first later interior snapshot has index 9. Both alternatives start
    # from exactly the same first eight snapshots and have total rank nine.
    path=OUT/'BGK_TEMPORAL_BRANCH.json'
    answer=json.loads(path.read_text()) if path.exists() else dict(prefix_rank=8,result_rank=9,scope='One-step ordering intervention; no optimizer retuning',branches={})
    answer['extra_branch_note']='Index 12 added as an exploratory control after branches 8 and 9: it is a later point with low difficulty under both fixed proxies, unlike the first interior point. Chosen by independent labels, not its branch outcome.'
    for i in [8,9,12]:
     if str(i) in answer['branches']:continue
     rom=StrictROM(m,list(ss[:8])+[ss[i]])
     d=evaluate(rom,par,truth,groups);d['added_mu']=mus[i].tolist();d['snapshot_index']=i
     answer['branches'][str(i)]=d
     print('branch',i,mus[i],d['typical'],flush=True)
     dump(OUT/'BGK_TEMPORAL_BRANCH.json',answer)


if __name__ == "__main__":
    main()
