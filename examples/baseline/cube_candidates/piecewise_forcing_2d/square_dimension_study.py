"""Square source partitions m=5,...,16, with N=floor(d/2) and GD T=80.

The source, PDE, metric, initialization, and greedy updates are unchanged.
All methods at a given d use the same worst-error test union. Search-based
test maxima are not asserted to be exact global maxima.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/piecewise_forcing_2d_matplotlib')
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import model, ascend, append, residual_gram, extrema

BASE=Path(__file__).resolve().parents[4]/'runs/piecewise_forcing_2d_20261112'
OUT=BASE/'square_dimension_study'
SEEDS=[1000,1001,1002,1003]
METHODS=['gd80','pool256','pool4096','pool10000']


def study(m):
    tick=time.perf_counter()
    d=m*m;N=d//2
    B,G,checks=model(m)
    B2,_,_=model(m,2048)
    checks['double_cutoff_relative_eigenvalue_change']=float(np.max(np.abs(np.sum(B2**2,axis=1)/np.sum(B**2,axis=1)-1)))
    E0=float(np.sqrt(np.ones(d)@G@np.ones(d)))
    assert G.min()>0
    previous=np.load(BASE/f'd{d}_snapshots.npz') if (BASE/f'd{d}_snapshots.npz').exists() else None
    spaces,parameters,training={},{},[]
    for seed in SEEDS:
        pool=np.random.default_rng(seed).uniform(-1,1,(10000,d))
        for method in METHODS:
            Q=np.zeros((d,0));selected=[];rounds=[]
            init=np.random.default_rng(seed+10000)
            prefix=None
            if previous is not None:
                matching=[k for k in previous.files if k.startswith(f'z_s{seed}_{method}_n')]
                if matching:
                    key=max(matching,key=lambda x:int(x.rsplit('n',1)[1]))
                    prefix=previous[key][:N]
            if method.startswith('pool'):
                candidates=pool[:int(method[4:])]
                U_pool=candidates@B.T;scores=np.sum(U_pool**2,axis=1)
            for n in range(N):
                H=residual_gram(B,G,Q)
                start=init.uniform(-1,1,d) if method=='gd80' else None
                if prefix is not None and n<len(prefix):
                    z=prefix[n].copy()
                    if method.startswith('pool'):
                        assert np.max(np.abs(z-candidates[np.argmax(scores)]))<1e-12
                elif method=='gd80':
                    z=ascend(H,start,80)
                else:
                    z=candidates[np.argmax(scores)].copy()
                J=float(z@H@z);assert J>1e-20
                Q,q=append(B,Q,z)
                selected.append(z)
                rounds.append(dict(N=n+1,J_selected=J))
                if method.startswith('pool'):
                    scores-=(U_pool@q)**2
            if prefix is not None:
                assert np.array_equal(np.asarray(selected)[:len(prefix)],prefix)
            spaces[(seed,method)]=Q
            parameters[f'z_s{seed}_{method}']=np.asarray(selected)
            training.append(dict(seed=seed,method=method,reused_prefix_length=0 if prefix is None else len(prefix),rounds=rounds))
        print(f'd={d}, N={N}: seed {seed} trained',flush=True)
    np.savez_compressed(OUT/f'd{d}_snapshots.npz',**parameters)

    rng=np.random.default_rng(412000+d)
    tests=[rng.uniform(-1,1,(4096,d)),rng.choice([-1.,1.],(4096,d)),np.ones((1,d)),-np.ones((1,d))]
    # Separate search batches permit a check of test-set sensitivity.
    extra_tests=[];upper_bounds={}
    for (seed,method),Q in spaces.items():
        R=B-Q@(Q.T@B);H=R.T@R
        eig,U=np.linalg.eigh(H)
        upper_bounds[(seed,method)]=float(np.sqrt(d*max(0.,eig[-1]))/E0)
        signs=np.where(U[:,-8:].T>=0,1.,-1.)
        tests.extend([signs,-signs,extrema(H,rng.uniform(-1,1,(128,d)),steps=240)])
        extra_tests.append(extrema(H,rng.uniform(-1,1,(128,d)),steps=240))
    base_test=np.unique(np.vstack(tests),axis=0)
    extra=np.unique(np.vstack(extra_tests),axis=0)
    # The final shared set contains both independent search batches.
    test=np.unique(np.vstack([base_test,extra]),axis=0)
    np.savez_compressed(OUT/f'd{d}_test.npz',parameters=test)
    U=test@B.T;U_base=base_test@B.T
    unorm=np.sum(U**2,axis=1)
    curves=[];results=[]
    for (seed,method),Q in spaces.items():
        coeff=U@Q
        squared=np.maximum(0.,unorm[:,None]-np.cumsum(coeff**2,axis=1))
        curve=np.r_[1.,np.sqrt(squared.max(axis=0))/E0]
        assert np.max(np.diff(curve))<=1e-11
        R=B-Q@(Q.T@B)
        direct=np.sqrt(np.sum((test@R.T)**2,axis=1).max())/E0
        assert abs(direct-curve[-1])<1e-11
        base_max=np.sqrt(np.sum((U_base-(U_base@Q)@Q.T)**2,axis=1).max())/E0
        curves.append(dict(seed=seed,method=method,error_over_E0=curve.tolist()))
        results.append(dict(seed=seed,method=method,N=N,error_over_E0=float(direct),
                            energy_error=float(direct*E0),global_upper_over_E0=upper_bounds[(seed,method)],
                            extra_search_relative_increase=float(direct/base_max-1),
                            orthogonality_defect=float(np.linalg.norm(Q.T@Q-np.eye(N)))))
    summary={}
    for method in METHODS:
        values=[r['error_over_E0'] for r in results if r['method']==method]
        summary[method]=dict(median=float(np.median(values)),minimum=float(min(values)),maximum=float(max(values)))
    comparisons={}
    for method in METHODS[1:]:
        paired=[]
        for seed in SEEDS:
            gd=next(r['error_over_E0'] for r in results if r['method']=='gd80' and r['seed']==seed)
            other=next(r['error_over_E0'] for r in results if r['method']==method and r['seed']==seed)
            paired.append(dict(seed=seed,relative_improvement=1-gd/other))
        comparisons[method]=dict(improvement_of_medians=1-summary['gd80']['median']/summary[method]['median'],
                                 wins=sum(x['relative_improvement']>0 for x in paired),paired=paired)
    report=dict(settings=dict(d=d,m=m,N=N,alpha=.6,T=80,seeds=SEEDS,methods=METHODS,
                source='Unnormalized blockwise constant source on m by m equal cells',
                parameter_domain='[-1,1]^d',initial_global_maximum=E0,
                test_count=len(test),base_test_count=len(base_test),
                test='Common union over all methods and seeds: 4096 uniform interior points, 4096 vertices, constant signs, leading-eigenvector signs, and two independent 128-start vertex searches per final space.',
                finite_test_is_global=False,device='CPU, float64, one BLAS thread'),
                checks=checks,training=training,curves=curves,results=results,summary=summary,
                comparisons=comparisons,elapsed_seconds=time.perf_counter()-tick)
    (OUT/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(d=d,N=N,median_errors={m:summary[m]['median'] for m in METHODS},
               gain_vs_pool10000=comparisons['pool10000']['improvement_of_medians'],
               paired_wins=comparisons['pool10000']['wins'],
               max_extra_test_increase=max(x['extra_search_relative_increase'] for x in results),
               seconds=report['elapsed_seconds'])),flush=True)
    return report


def outputs(reports):
    colors=['#0072B2','#555555','#009E73','#D55E00']
    styles=['-',':','-.','--'];markers=['o','s','^','D']
    labels=['GD, T=80','Pool 256','Pool 4096','Pool 10000']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dimensions=[r['settings']['d'] for r in reports]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for method,color,style,marker,label in zip(METHODS,colors,styles,markers,labels):
        values=[r['summary'][method]['median'] for r in reports]
        axes[0].plot(dimensions,values,color=color,ls=style,marker=marker,ms=4,lw=2,label=label)
    for method,color,style,marker,label in zip(METHODS[1:],colors[1:],styles[1:],markers[1:],labels[1:]):
        values=[100*r['comparisons'][method]['improvement_of_medians'] for r in reports]
        axes[1].plot(dimensions,values,color=color,ls=style,marker=marker,ms=4,lw=2,label=label)
    axes[0].set_ylabel(r'Maximum test energy error $E_N/E_0$')
    axes[0].set_title(r'$N=\lfloor d/2\rfloor$');axes[0].set_yscale('log')
    axes[1].set_ylabel('Error reduction relative to pool (%)')
    axes[1].set_title(r'$100(1-E_N^{\mathrm{GD}}/E_N^{\mathrm{pool}})$')
    axes[1].axhline(0,color='#999999',lw=.8)
    for ax in axes:
        ax.set_xlabel('Parameter dimension d');ax.set_xticks([25,64,121,196,256]);ax.grid(alpha=.17,which='both');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/'error_vs_dimension.png',dpi=180);fig.savefig(OUT/'error_vs_dimension.pdf');plt.close(fig)
    with (OUT/'final_errors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','N']+METHODS+['relative_improvement_vs_pool10000','paired_wins_vs_pool10000'])
        for r in reports:
            w.writerow([r['settings']['d'],r['settings']['N']]+[r['summary'][m]['median'] for m in METHODS]+[
                r['comparisons']['pool10000']['improvement_of_medians'],r['comparisons']['pool10000']['wins']])
    (OUT/'summary.json').write_text(json.dumps([{k:r[k] for k in ['settings','checks','summary','comparisons']} for r in reports],indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[study(m) for m in range(5,17)]
    outputs(reports)
    print('COMPLETE',flush=True)
