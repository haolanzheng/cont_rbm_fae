"""Separate inner-search improvement from the change in the greedy spaces."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import copy,json
import numpy as np
from .step_budget import OUT,BASE,PREV,cases
from .model import ThermalBlock,RB
from .run import gd,train,dump
from .correlation_prior import sample,train as prior_train

def main():
    pbase=ThermalBlock(2,7,28);records=[];reproductions=[]
    references=json.loads((OUT/'references.json').read_text())
    for case in cases():
        if case['group']=='correlation':continue
        p=copy.copy(pbase);p.coefficient_center=np.full(14,case['center']);p.coefficient_jacobian=case['jacobian']
        if case['name']=='contrast_10':path=BASE/'basis_d14_gd_s1000.npz'
        else:path=PREV/f"basis_{case['name']}_gd_s1000.npz"
        V=np.load(path)['V'];rb=RB(p,16)
        for v in V[:,:16].T:rb.add(v)
        rng=np.random.default_rng(537)
        Z=sample(rng,8,case['L']) if case['group']=='prior' else rng.uniform(-1,1,(8,14))
        for start,z in enumerate(Z):
            vals=[]
            for T in [80,160,320,640]:
                zz,stat=gd(rb,z.copy(),steps=T);vals.append(dict(T=T,**stat))
            logvals=np.array([v['log_residual'] for v in vals]);assert np.all(np.diff(logvals)>=-1e-12)
            records.append(dict(case=case['name'],start=start,values=vals,residual_norm_ratio_640_to_80=float(np.exp(.5*(logvals[-1]-logvals[0])))))
        print('audited',case['name'],flush=True)
        if case['name'] in ['contrast_10','prior_0.15']:
            if case['group']=='prior':rr,_,rounds=prior_train(p,'gd',1000,case['L'],steps=80)
            else:rr,_,rounds=train(p,'gd',1000,24,steps=80)
            old=next(r for r in references if r['case']==case['name'] and r['method']=='gd' and r['seed']==1000)
            err=max(np.max(abs(np.array(a['z'])-np.array(b['z']))) for a,b in zip(rounds,old['rounds']))
            assert len(rounds)==len(old['rounds']) and err==0
            reproductions.append(dict(case=case['name'],selected_parameter_max_difference=float(err),rounds=len(rounds)))
    dump(OUT/'fixed_space_audit.json',dict(N=16,basis_seed=1000,starts=8,start_seed=537,records=records,reproduction=reproductions,interpretation='Residual norm improvement on one unchanged reduced space, not final solution error.'))
    print('AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
