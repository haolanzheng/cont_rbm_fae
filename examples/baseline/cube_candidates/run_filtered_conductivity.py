"""Reproducible CPU screening of filtered-material diffusion."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/filtered_conductivity_mpl')
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .filtered_conductivity import FilteredConductivity
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'runs/filtered_conductivity_20261111'
METHODS=['gd80','gd320','pool256','pool4096','pool10000']

def tests(d):
    interior=qmc.Sobol(d,scramble=True,seed=811+d).random_base2(12)
    vertices=(qmc.Sobol(d,scramble=True,seed=911+d).random_base2(12)>.5).astype(float)
    # Exhaust all binary configurations of eight macroscopic regions.
    blocks=((np.arange(256)[:,None]>>np.arange(8))&1).astype(float)[:,np.minimum(7,np.arange(d)*8//d)]
    intervals=[]
    for i in range(d):
        for j in range(i+1,d+1):
            z=np.ones(d);z[i:j]=0;intervals.extend([z,1-z])
    x=(np.arange(d)+.5)/d
    waves=np.array([(np.cos(2*np.pi*k*x+phase)>0).astype(float) for k in range(1,9) for phase in np.arange(8)*np.pi/4])
    return np.vstack((interior,vertices,blocks,intervals,waves)),len(interior)

def curves(model,q,test):
    m=q.shape[1];absolute=np.zeros(m+1);relative=absolute.copy()
    for start in range(0,len(test),256):
        z=test[start:start+256];a=model.coefficients(z);u=model.solution(z)
        _,A=model.reduced(q,z);L=np.linalg.cholesky(A);b=-(a*model.lift)@q
        y=np.linalg.solve(L,b[...,None])[...,0]
        basis=np.linalg.solve(L,np.broadcast_to(q.T,(len(z),*q.T.shape)))
        corr=np.cumsum(basis*y[...,None],axis=1)
        approx=np.concatenate((np.broadcast_to(model.lift,(len(z),1,model.d)),model.lift+corr),axis=1)
        err=np.linalg.norm(approx-u[:,None,:],axis=2)
        absolute=np.maximum(absolute,err.max(axis=0));relative=np.maximum(relative,(err/np.linalg.norm(u,axis=1)[:,None]).max(axis=0))
        if start==0:
            for n in [0,m//2,m]:
                direct,_=model.reduced(q[:,:n],z)
                assert np.max(abs(direct-approx[:,n]))<1e-10
    return absolute,relative

def check(model):
    rng=np.random.default_rng(61);d=model.parameter_dimension;q=np.empty((model.d,0))
    for _ in range(3):q=model.append(q,rng.uniform(0,1,d))
    z=rng.uniform(.2,.8,d);checks={}
    for kind in ['residual','error','relative']:
        value,g=model.objective(q,z,True,kind);eps=1e-6
        gn=np.array([(model.objective(q,z+eps*e,kind=kind)-model.objective(q,z-eps*e,kind=kind))/(2*eps) for e in np.eye(d)])
        rel=float(np.linalg.norm(g-gn)/np.linalg.norm(g));assert rel<2e-5,(kind,rel)
        checks[kind+'_gradient_relative_error']=rel
    q0=np.empty((model.d,0));b=z@model.K.T
    assert abs(model.objective(q0,z)-model.span**2*np.var(b))<1e-10
    a=model.coefficients(z);u=model.solution(z);t,_=model.reduced(q,z)
    checks['flux_variation']=float(np.ptp(a*u));checks['Galerkin_residual']=float(np.linalg.norm(q.T@(a*t)))
    assert checks['flux_variation']<1e-10 and checks['Galerkin_residual']<1e-10
    return checks

def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[(d,c,ell) for d in args.d for c in args.contrast for ell in args.length]
    settings=dict(cases=cases,methods=METHODS,seeds=args.seeds,N=args.N,quadrature=args.quadrature,alpha=.6,device='CPU float64 one BLAS thread',objective='reference H1 dual residual squared',normalizer='local quadratic Gram Frobenius norm at center, fixed per search',tests='4096 Sobol interior, 4096 binary vertices, all 8-block vertices, all contiguous binary intervals and low-frequency patterns; union with relative-error ascent on all methods at intermediate and final ranks',N_definition='lift plus N-1 homogeneous correction snapshots')
    (OUT/f'settings_{args.label}.json').write_text(json.dumps(settings,indent=2)+'\n')
    rows=[]
    for d,c,ell in cases:
        tag=f'd{d}_c{c:g}_ell{ell:g}';model=FilteredConductivity(d,c,ell,args.quadrature)
        (OUT/f'{tag}_checks.json').write_text(json.dumps(check(model),indent=2)+'\n')
        base,ni=tests(d);starts=np.vstack((qmc.Sobol(d,scramble=True,seed=333).random_base2(3),base[-8:]))
        local=[];extras=[]
        for method in METHODS:
            for seed in args.seeds:
                dest=OUT/f'{tag}_{method}_s{seed}.npz';rng=np.random.default_rng(seed+1000003)
                q=np.empty((model.d,0));selected=[];hist=[];timings=[]
                begin=time.perf_counter()
                if method.startswith('pool'):pool=rng.uniform(0,1,(int(method[4:]),d))
                if dest.exists():
                    old=np.load(dest);q=old['Q'];selected=list(old['selected']);hist=list(old['indicators']);timings=list(old['timings'])
                    if method.startswith('gd'):
                        for _ in selected:rng.uniform(0,1,d)
                    offset=timings[-1] if timings else 0
                else:offset=0
                while q.shape[1]<args.N-1:
                    if method.startswith('gd'):z,value=model.search(q,rng.uniform(0,1,d),steps=int(method[2:]))
                    else:
                        vals=model.values(q,pool,chunk=512);idx=int(vals.argmax());z=pool[idx];value=vals[idx]
                    try:q=model.append(q,z)
                    except ValueError:break
                    selected.append(z);hist.append(float(value));timings.append(offset+time.perf_counter()-begin)
                np.savez_compressed(dest,Q=q,selected=selected,indicators=hist,timings=timings)
                assert np.linalg.norm(q.T@q-np.eye(q.shape[1]))<1e-10
                for n in sorted(set([min(7,q.shape[1]),min(15,q.shape[1]),q.shape[1]])):
                    found,_=model.search(q[:,:n],starts,steps=160,kind='relative');extras.append(found)
                local.append((method,seed,q,timings))
                print(tag,method,seed,'trained N',q.shape[1]+1,'seconds',round(timings[-1],3),flush=True)
        common=np.vstack((base,*extras));np.savez_compressed(OUT/f'{tag}_test.npz',parameters=common,interior_count=ni,base_count=len(base))
        for method,seed,q,timings in local:
            ab,rel=curves(model,q,common)
            row=dict(d=d,contrast=c,length=ell,method=method,seed=seed,N=list(range(1,len(rel)+1)),E_test=ab.tolist(),E_relative=rel.tolist(),timings=[0]+timings,test_count=len(common))
            rows.append(row)
            print(tag,method,seed,'relative N8,N16,Nfinal',*[round(rel[min(n,len(rel))-1],6) for n in [8,16,len(rel)]],flush=True)
        (OUT/f'results_{args.label}.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--d',type=int,nargs='+',default=[32]);p.add_argument('--contrast',type=float,nargs='+',default=[10,30]);p.add_argument('--length',type=float,nargs='+',default=[.1,.2]);p.add_argument('--N',type=int,default=24);p.add_argument('--quadrature',type=int,default=256);p.add_argument('--seeds',type=int,nargs='+',default=[1000,1001]);p.add_argument('--label',default='screen');args=p.parse_args()
    with threadpool_limits(limits=1):run(args)
