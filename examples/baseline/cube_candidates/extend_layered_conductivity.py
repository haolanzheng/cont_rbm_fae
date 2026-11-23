"""Continue saved layered-conductivity greedy spaces; evaluate every dimension.
CPU float64, one BLAS thread. Test sets and existing prefixes remain unchanged.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/layered_conductivity_mpl')
from pathlib import Path
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
from .layered_conductivity import LayeredConductivity
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'runs/layered_conductivity_20261110'
OUT=ROOT/'runs/layered_conductivity_full_20261111'

def curves(model,q,test):
    absolute=np.zeros(model.d);relative=absolute.copy()
    for begin in range(0,len(test),256):
        z=test[begin:begin+256];a=model.coefficients(z);u=model.solution(z)
        A=np.einsum('bi,ij,ik->bjk',a,q,q,optimize=True)
        L=np.linalg.cholesky(A);b=-(a*model.lift)@q
        y=np.linalg.solve(L,b[...,None])[...,0]
        basis=np.linalg.solve(L,np.broadcast_to(q.T,(len(z),*q.T.shape)))
        corrections=np.cumsum(basis*y[...,None],axis=1)
        approx=np.concatenate((np.broadcast_to(model.lift,(len(z),1,model.d)),model.lift+corrections),axis=1)
        error=np.linalg.norm(approx-u[:,None,:],axis=2)
        absolute=np.maximum(absolute,error.max(axis=0))
        relative=np.maximum(relative,(error/np.linalg.norm(u,axis=1)[:,None]).max(axis=0))
        if begin==0:
            for n in (1,model.d//2,model.d):
                direct,_=model.reduced(q[:,:n-1],z)
                assert np.max(np.abs(direct-approx[:,n-1]))<2e-11
    return absolute,relative

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    settings=json.loads((OLD/'settings.json').read_text())
    settings.update(N='all dimensions 1 through d; N counts lift plus N-1 correction vectors',tests='identical previously saved common test sets; no new test-point selection',relative_error='pointwise H1 seminorm error divided by H1 seminorm of exact solution')
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    rows=[]
    for d,c in settings['cases']:
        model=LayeredConductivity(d,c);tag=f'd{d}_c{c}'
        test=np.load(OLD/f'test_{tag}.npz')['parameters']
        for method in settings['methods']:
            for seed in settings['seeds']:
                dest=OUT/f'{tag}_{method}_s{seed}.npz'
                rng=np.random.default_rng(seed+1000003)
                old=np.load(OLD/dest.name);q=old['Q'].copy()
                selected=list(old['selected_parameters']);history=list(old['selected_squared_indicators'])
                if method.startswith('pool'):pool=rng.uniform(0,1,(int(method[4:]),d))
                else:
                    for _ in selected:rng.uniform(0,1,d)
                begin=time.perf_counter()
                while q.shape[1]<d-1:
                    if method.startswith('gd'):z,value=model.search(q,rng.uniform(0,1,d),steps=int(method[2:]))
                    else:
                        values=model.values(q,pool);idx=int(values.argmax());z=pool[idx].copy();value=values[idx]
                    q=model.append(q,z);selected.append(z);history.append(float(value))
                seconds=time.perf_counter()-begin
                assert np.linalg.norm(q.T@q-np.eye(d-1))<1e-11
                absolute,relative=curves(model,q,test)
                assert absolute[-1]<1e-10
                np.savez_compressed(dest,Q=q,selected_parameters=selected,selected_squared_indicators=history,E_test=absolute,E_relative=relative)
                row=dict(d=d,contrast=c,method=method,seed=seed,N=list(range(1,d+1)),E_test=absolute.tolist(),E_relative=relative.tolist(),additional_training_seconds=seconds)
                rows.append(row)
                (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(f'{tag} {method} seed={seed}: E(d/2)={absolute[d//2-1]:.4g}, E(d-1)={absolute[-2]:.4g}, E(d)={absolute[-1]:.3g}',flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
