"""Single-thread CPU screening of the natural layered-conductivity cube.

Run: .venv_cpu/bin/python -m examples.baseline.cube_candidates.run_layered_conductivity
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/layered_conductivity_mpl')
from pathlib import Path
import json,time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .layered_conductivity import LayeredConductivity
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'runs/layered_conductivity_20261110'
METHODS=('gd80','gd320','pool256','pool4096','pool10000')
SEEDS=tuple(range(1000,1004))
CASES=[(d,c) for c in (10,100) for d in (8,16,32)]+[(64,10)]

def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def checks(model):
    rng=np.random.default_rng(61);q=np.empty((model.d,0))
    for _ in range(min(3,model.d-2)):q=model.append(q,rng.uniform(0,1,model.d))
    z=rng.uniform(.15,.85,model.d);record={}
    for kind in ('residual','error'):
        value,g=model.objective(q,z,True,kind)
        eps=1e-6;num=np.array([(model.objective(q,z+eps*e,kind=kind)-model.objective(q,z-eps*e,kind=kind))/(2*eps) for e in np.eye(model.d)])
        rel=float(np.linalg.norm(g-num)/max(np.linalg.norm(g),1e-15));assert rel<2e-6,(kind,rel)
        record[kind+'_gradient_relative_error']=rel
    a=model.coefficients(z);u=model.solution(z);t,_=model.reduced(q,z)
    record['exact_flux_variation']=float(np.ptp(a*u));assert record['exact_flux_variation']<1e-11
    record['galerkin_residual']=float(np.linalg.norm(q.T@(a*t)));assert record['galerkin_residual']<1e-10
    assert abs(u.sum()/np.sqrt(model.d)-1)<1e-13
    # Complete homogeneous correction space must reproduce the exact solution.
    qfull=np.linalg.qr(np.eye(model.d)[:,:-1]-np.eye(model.d)[:,-1:])[0]
    full,_=model.reduced(qfull,z);record['full_space_error']=float(np.linalg.norm(full-u));assert record['full_space_error']<1e-11
    return record

def tests(d):
    interior=qmc.Sobol(d,scramble=True,seed=202611100+d).random_base2(14)
    if d<=16:
        indices=np.arange(2**d,dtype=np.uint64)
        vertices=((indices[:,None]>>np.arange(d,dtype=np.uint64))&1).astype(float)
    else:vertices=(qmc.Sobol(d,scramble=True,seed=202611200+d).random_base2(14)>=.5).astype(float)
    patterns=np.vstack((np.eye(d),1-np.eye(d),np.zeros((1,d)),np.ones((1,d)),np.full((1,d),.5)))
    return np.vstack((interior,vertices,patterns)),dict(interior=len(interior),vertices=len(vertices),patterns=len(patterns))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    save('settings.json',dict(cases=CASES,methods=METHODS,seeds=SEEDS,alpha=.6,
        parameter='z_i in [0,1], a_i=1+(contrast-1)z_i on d equal cells',
        PDE="-(a u')'=0; u(0)=0, u(1)=1; exact solution by integrated resistances",
        reduced_model='lift x plus N-1 homogeneous correction snapshots; actual parameter-dependent Galerkin equations',
        N='d/2 total affine-space generators, including lift',
        objective='squared dual norm of residual with respect to fixed reference H1_0 norm',
        normalizer='Frobenius norm of local quadratic objective Gram matrix at cube center, fixed within each greedy search',
        error='absolute reference H1 seminorm (energy of reference coefficient 1)',
        initialization='one uniform start per GD round; identical per-round random draws for GD80 and GD320',
        pool='fixed iid uniform pool reused within greedy, nested across sizes',
        tests='same Sobol interior and vertex set for all methods; augmented by union of 8-start T160 direct error maximization on all final spaces',
        timing='pool preparation, search and enrichment; excludes validation',device='CPU float64; one BLAS thread',
        width='No exact width asserted'))
    allrows=[];allchecks=[]
    for d,contrast in CASES:
        model=LayeredConductivity(d,contrast);N=d//2;tag=f'd{d}_c{contrast}'
        check=checks(model);allchecks.append(dict(d=d,contrast=contrast,**check));save('checks.json',allchecks)
        test,counts=tests(d);starts=qmc.Sobol(d,scramble=True,seed=202611300+d).random_base2(3)
        local=[];extras=[]
        for seed in SEEDS:
            for method in METHODS:
                rng=np.random.default_rng(seed+1000003);q=np.empty((d,0));selected=[];history=[]
                begin=time.perf_counter()
                if method.startswith('pool'):pool=rng.uniform(0,1,(int(method[4:]),d))
                for n in range(N-1):
                    if method.startswith('gd'):z,value=model.search(q,rng.uniform(0,1,d),steps=int(method[2:]))
                    else:
                        vals=model.values(q,pool);idx=int(vals.argmax());z=pool[idx].copy();value=vals[idx]
                    selected.append(z);history.append(float(value));q=model.append(q,z)
                elapsed=time.perf_counter()-begin
                row=dict(d=d,contrast=contrast,N=N,method=method,seed=seed,seconds=elapsed,
                    orthogonality=float(np.linalg.norm(q.T@q-np.eye(N-1))))
                assert row['orthogonality']<1e-12
                np.savez_compressed(OUT/f'{tag}_{method}_s{seed}.npz',Q=q,selected_parameters=selected,selected_squared_indicators=history)
                validation,_=model.search(q,starts,steps=160,kind='error');extras.append(validation)
                local.append((row,q))
                print(f'{tag} {method} seed={seed}: training {elapsed:.3f}s',flush=True)
        common=np.vstack((test,*extras));np.savez_compressed(OUT/f'test_{tag}.npz',parameters=common,base_count=len(test))
        for row,q in local:
            err=model.values(q,common,kind='error');row['E_test']=float(np.sqrt(err.max()))
            row['E_interior']=float(np.sqrt(err[:counts['interior']].max()));row['E_base']=float(np.sqrt(err[:len(test)].max()))
            row['E_mean_square_root']=float(np.sqrt(np.mean(err[:counts['interior']])))
            row['worst_test_index']=int(err.argmax());row['test_count']=len(common)
            # Same fixed validation set for every recorded dimension.
            ns=sorted(set([1, max(2,N//2),N]));curve=[]
            for n in ns:
                curve.append(dict(N=n,error=float(np.sqrt(model.values(q[:,:n-1],common,kind='error').max()))))
            row['test_curve']=curve;allrows.append(row)
        save('results.json',allrows)
        print(f'{tag}: validated on {len(common)} common points',flush=True)
    summary=[]
    for d,c in CASES:
        for method in METHODS:
            rr=[r for r in allrows if (r['d'],r['contrast'],r['method'])==(d,c,method)]
            summary.append(dict(d=d,contrast=c,N=rr[0]['N'],method=method,
                E_test=float(np.median([r['E_test'] for r in rr])),E_min=min(r['E_test'] for r in rr),E_max=max(r['E_test'] for r in rr),
                E_interior=float(np.median([r['E_interior'] for r in rr])),seconds=float(np.median([r['seconds'] for r in rr]))))
    save('summary.json',summary);print('COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
