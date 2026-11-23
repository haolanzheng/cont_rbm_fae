"""Modest physical/source variants of the piecewise-source cube baseline.

-uxx-kappa*uyy+reaction*u=f_z, homogeneous Dirichlet boundary.
Sources are cell indicators or tensor products of interior nodal hats.
Both objectives equal the corresponding PDE energy error squared.
No designed spectral weights or source normalization are introduced.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json,csv
from pathlib import Path
import numpy as np
from equal_time_pool import train
from run import extrema,model as old_model

BASE=Path(__file__).resolve().parents[4]/'runs/piecewise_forcing_2d_20261112'
OUT=BASE/'natural_variants'
SEEDS=[1000,1001,1002,1003]
VARIANTS=[('original','block',1.,0.),('anisotropic4','block',4.,0.),
          ('anisotropic16','block',16.,0.),('bilinear_source','hat',1.,0.),
          ('reaction100','block',1.,100.)]


def model(m,basis='block',kappa=1.,reaction=0.,cutoff=1024):
    k=np.arange(1,cutoff+1,dtype=float);r=np.arange(1,m+1,dtype=float)
    if basis=='block':
        Q=np.sin(np.pi*((np.arange(m)+.5)/m)[:,None]*r)
        Q/=np.linalg.norm(Q,axis=0)
        residue=np.arange(1,cutoff+1)%(2*m)
        alias=np.minimum(residue,2*m-residue)
        amplitude=2*np.sqrt(2)*np.sin(np.pi*k/(2*m))/(np.pi*k)
        weights=amplitude**2*np.where(alias==m,m,m/2)
        valid=alias>0;full_mass=np.full(m,1/m)
    else:
        h=1/(m+1)
        Q=np.sqrt(2/(m+1))*np.sin(np.pi*np.arange(1,m+1)[:,None]*r/(m+1))
        residue=np.arange(1,cutoff+1)%(2*(m+1))
        alias=np.minimum(residue,2*(m+1)-residue)
        amplitude=np.sqrt(2)*h*np.sinc(k*h/2)**2
        weights=amplitude**2*(m+1)/2
        valid=(alias>0)&(alias<=m)
        full_mass=h*(2+np.cos(r*np.pi/(m+1)))/3
    mass=np.zeros((cutoff,m));mass[np.flatnonzero(valid),alias[valid]-1]=weights[valid]
    lam=np.pi**2*(k[:,None]**2+kappa*k[None,:]**2)+reaction
    eig=mass.T@(1/lam)@mass
    captured=mass.sum(axis=0)
    missing=np.maximum(0,full_mass[:,None]*full_mass[None,:]-captured[:,None]*captured[None,:])
    omitted_min=np.pi**2*min((cutoff+1)**2+kappa,1+kappa*(cutoff+1)**2)+reaction
    tail=missing/omitted_min
    U=np.kron(Q,Q);B=np.sqrt(eig.ravel())[:,None]*U.T;G=B.T@B
    return B,G,dict(relative_gram_tail_bound=float(np.max(tail/eig)),cutoff=cutoff,
                    gram_minimum=float(G.min()),orthogonality_defect=float(np.linalg.norm(U.T@U-np.eye(m*m))))


def validate():
    checks=[]
    for basis in ['block','hat']:
        m=4;K=39;k=np.arange(1,K+1,dtype=float)
        for kappa,reaction in [(1.,0.),(4.,0.),(1.,100.)]:
            B,G,_=model(m,basis,kappa,reaction,K)
            if basis=='block':
                x=np.arange(m)/m
                F=np.sqrt(2)*(np.cos(np.pi*k[:,None]*x)-np.cos(np.pi*k[:,None]*(x+1/m)))/(np.pi*k[:,None])
            else:
                h=1/(m+1)
                F=(np.sqrt(2)*h*np.sinc(k*h/2)**2)[:,None]*np.sin(np.pi*k[:,None]*np.arange(1,m+1)*h)
                # Independent quadrature of each piecewise linear hat.
                points,w=np.polynomial.legendre.leggauss(32)
                numerical=np.zeros_like(F)
                for j in range(m):
                    center=(j+1)*h
                    for a,b in [(center-h,center),(center,center+h)]:
                        x=(a+b)/2+(b-a)*points/2
                        values=np.maximum(1-np.abs(x-center)/h,0)
                        numerical[:,j]+=np.sqrt(2)*(np.sin(np.pi*k[:,None]*x)@(w*values))*(b-a)/2
                assert np.max(np.abs(numerical-F))<2e-14
            FF=np.einsum('ki,lj->klij',F,F).reshape(K*K,m*m)
            lam=(np.pi**2*(k[:,None]**2+kappa*k[None,:]**2)+reaction).ravel()
            direct=FF.T@(FF/lam[:,None]);diff=float(np.linalg.norm(G-direct)/np.linalg.norm(G))
            assert diff<1e-12
            checks.append(dict(basis=basis,kappa=kappa,reaction=reaction,gram_relative_difference=diff))
    B,G,_=model(8);Bo,Go,_=old_model(8)
    assert np.max(np.abs(B-Bo))<1e-14 and np.max(np.abs(G-Go))<1e-14
    return checks


def study(m,variant,basis,kappa,reaction):
    d=m*m;N=d//2;matched={64:20000,144:28200}[d];methods=[0,10000,matched,100000]
    B,G,checks=model(m,basis,kappa,reaction)
    B2,_,_=model(m,basis,kappa,reaction,2048)
    checks['doubled_cutoff_relative_eigenvalue_change']=float(np.max(np.abs(np.sum(B2**2,axis=1)/np.sum(B**2,axis=1)-1)))
    assert G.min()>0
    E0=float(np.sqrt(np.ones(d)@G@np.ones(d)))
    spaces={};parameters={};training=[]
    for M in methods:
        for seed in SEEDS:
            Q,Z,t=train(B,G,seed,M)
            spaces[(M,seed)]=Q;parameters[f'z_s{seed}_M{M}']=Z
            training.append(dict(M=M,seed=seed,**t))
    np.savez_compressed(OUT/f'd{d}_{variant}_snapshots.npz',**parameters)
    rng=np.random.default_rng(920000+d)
    tests=[rng.uniform(-1,1,(4096,d)),rng.choice([-1.,1.],(4096,d)),np.ones((1,d)),-np.ones((1,d))]
    if variant=='original':
        tests.append(np.load(BASE/'equal_time_pool'/f'd{d}_test.npz')['parameters'])
    extra=[]
    for (M,seed),Q in spaces.items():
        R=B-Q@(Q.T@B);H=R.T@R
        _,v=np.linalg.eigh(H);signs=np.where(v[:,-8:].T>=0,1.,-1.)
        tests.extend([signs,-signs,extrema(H,rng.uniform(-1,1,(128,d)),steps=240)])
        extra.append(extrema(H,rng.uniform(-1,1,(128,d)),steps=240))
    test0=np.unique(np.vstack(tests),axis=0);test=np.unique(np.vstack([test0]+extra),axis=0)
    np.savez_compressed(OUT/f'd{d}_{variant}_test.npz',parameters=test)
    results=[]
    for (M,seed),Q in spaces.items():
        R=B-Q@(Q.T@B);H=R.T@R
        E=float(np.sqrt(np.max(np.sum((test@R.T)**2,axis=1)))/E0)
        Epre=float(np.sqrt(np.max(np.sum((test0@R.T)**2,axis=1)))/E0)
        results.append(dict(M=M,seed=seed,error_over_E0=E,
            normalized_mean_square_error=float(np.trace(H)/(3*E0**2)),
            second_search_increase=E/Epre-1,
            relative_orthogonality_defect=float(np.linalg.norm(Q.T@Q-np.eye(N)))))
    summary={}
    for M in methods:
        e=[r['error_over_E0'] for r in results if r['M']==M]
        summary[str(M)]=dict(median=float(np.median(e)),minimum=float(min(e)),maximum=float(max(e)))
    gains={str(M):100*(1-summary['0']['median']/summary[str(M)]['median']) for M in methods if M}
    report=dict(settings=dict(d=d,N=N,variant=variant,basis=basis,kappa=kappa,reaction=reaction,
                T=80,alpha=.6,seeds=SEEDS,pool_sizes=methods[1:],initial_maximum=E0,
                matched_size_from_original_only=matched,not_a_new_equal_time_benchmark=True,
                test_count=len(test),finite_test_is_global=False),
                checks=checks,training=training,results=results,summary=summary,gd_gain_percent=gains)
    (OUT/f'd{d}_{variant}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(d=d,variant=variant,errors={M:summary[M]['median'] for M in summary},gd_gain_percent=gains,
        max_test_increase=max(r['second_search_increase'] for r in results),checks=checks)),flush=True)
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'model_validation.json').write_text(json.dumps(validate(),indent=2)+'\n')
    reports=[]
    for m in [8,12]:
        for variant,basis,kappa,reaction in VARIANTS:
            reports.append(study(m,variant,basis,kappa,reaction))
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','variant','gd_error','pool10000_error','pool100000_error','gd_gain_vs_10000_percent','gd_gain_vs_100000_percent'])
        for r in reports:
            w.writerow([r['settings']['d'],r['settings']['variant'],r['summary']['0']['median'],r['summary']['10000']['median'],r['summary']['100000']['median'],r['gd_gain_percent']['10000'],r['gd_gain_percent']['100000']])
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
