"""Regression checks for cached H@z against original updates and saved runs.

Validate the best returned iterate, input preservation, all saved GD greedy
prefixes, and their common-test error curves. No training data are overwritten.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
from run import model,ascend,append,residual_gram
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/piecewise_forcing_2d_20261112'
SUMMARY=BASE/'slides_summary'

def original_ascend(H,z,steps):
    z=z.copy();ell=max(float(np.linalg.norm(H)),1e-30)
    best=z.copy();score=float(z@H@z)
    for _ in range(steps):
        g=2*H@z/ell;z=np.clip(z+.6*g/np.sqrt(1+g@g),-1,1)
        value=float(z@H@z)
        if value>score:score,best=value,z.copy()
    return best

def main(output):
    if output.exists():raise FileExistsError(output)
    rng=np.random.default_rng(5616);isolated=[]
    for d in [4,16,64]:
        A=rng.normal(size=(d,d));V,_=np.linalg.qr(A)
        for name,H in [('zero',np.zeros((d,d))),('identity',np.eye(d)),('dense',A.T@A),
                       ('ill_conditioned', (V*np.geomspace(1,1e-12,d))@V.T)]:
            for steps in [0,1,20,80]:
                z=rng.uniform(-1,1,d);original=z.copy()
                new=ascend(H,z,steps);ref=original_ascend(H,z,steps)
                np.testing.assert_array_equal(z,original)
                np.testing.assert_allclose(new,ref,rtol=1e-10,atol=1e-10)
                assert np.max(np.abs(new))<=1
                isolated.append(dict(d=d,matrix=name,steps=steps,max_parameter_difference=float(np.max(np.abs(new-ref)))))
    errors=json.loads((SUMMARY/'shared_test_errors.json').read_text());paths=[]
    for d in [4,16]+[m*m for m in range(5,17)]:
        B,G,_=model(round(np.sqrt(d)))
        if d==4:saved=json.loads((SUMMARY/'d4.json').read_text())
        else:saved=np.load(BASE/('dimension_study' if d==16 else 'square_dimension_study')/f'd{d}_snapshots.npz')
        test=np.load(SUMMARY/f'd{d}_test.npz')['parameters'];U=test@B.T;norm2=np.sum(U*U,axis=1)
        for seed in range(1000,1004):
            ref=np.array(next(r['parameters'] for r in saved['curves'] if r['method']=='gd80' and r['seed']==seed)) if d==4 else saved[f'z_s{seed}_gd80']
            Q=np.zeros((d,0));init=np.random.default_rng(seed+10000);selected=[]
            for n in range(len(ref)):
                H=residual_gram(B,G,Q);z=ascend(H,init.uniform(-1,1,d),80)
                Q,_=append(B,Q,z);selected.append(z.copy())
            np.testing.assert_allclose(selected,ref,atol=1e-10,rtol=1e-10)
            curve=np.r_[np.sqrt(norm2.max()),np.sqrt(np.maximum(0,norm2[:,None]-np.cumsum((U@Q)**2,axis=1)).max(axis=0))]
            expected=next(r['energy_error'] for r in errors[str(d)]['curves'] if r['seed']==seed and r['method']=='gd80')
            np.testing.assert_allclose(curve,expected,atol=1e-12,rtol=1e-10)
            paths.append(dict(d=d,seed=seed,N=len(ref),max_parameter_difference=float(np.max(np.abs(np.array(selected)-ref))),
                              max_error_difference=float(np.max(np.abs(curve-expected)))))
        print(f'd={d}: all four GD paths and error curves verified',flush=True)
    result=dict(isolated_updates=isolated,full_greedy_paths=paths,
                max_saved_parameter_difference=max(r['max_parameter_difference'] for r in paths),
                max_saved_error_difference=max(r['max_error_difference'] for r in paths))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('max_')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
