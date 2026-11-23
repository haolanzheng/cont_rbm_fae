"""Pool-size and equal-wall-time comparison for the square-source baseline.

Complete training from V0, CPU float64, one BLAS thread. Includes sampling,
method-specific precomputation, search, score updates and enrichment. Excludes
shared spectral-model construction, evaluation, validation, plotting and I/O.
The matched pool is calibrated using time only, never using test errors.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/cg_equal_time_pool_matplotlib')
import csv
import json
import platform
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from run import model,ascend,append,residual_gram,extrema

BASE=Path(__file__).resolve().parents[2]/'runs/piecewise_forcing_2d_20261112'
OUT=BASE/'equal_time_pool'
SEEDS=[1000,1001,1002,1003]
SIZES=[256,4096,10000,25000,50000,100000]
REPEATS=5


def train(B,G,seed,M,*,steps=80,max_n=None):
    cpu0=process_time();tick=perf_counter();d=len(B);N=d//2 if max_n is None else max_n
    Q=np.zeros((d,0));selected=[]
    rng=np.random.default_rng(seed if M else seed+10000)
    if M:
        pool=rng.uniform(-1,1,(M,d));U=pool@B.T;scores=np.sum(U**2,axis=1)
    setup=perf_counter()-tick
    for n in range(N):
        z=pool[np.argmax(scores)] if M else ascend(residual_gram(B,G,Q),rng.uniform(-1,1,d),steps)
        Q,q=append(B,Q,z)
        selected.append(z.copy())
        if M and n<N-1:scores-=(U@q)**2
    measured=dict(wall_seconds=perf_counter()-tick,cpu_seconds=process_time()-cpu0,setup_seconds=setup)
    return Q,np.asarray(selected),measured


def calibrate(B,G):
    train(B,G,1000,0)
    gd=float(np.median([train(B,G,seed,0)[2]['wall_seconds'] for seed in SEEDS]))
    M=25000;records=[]
    # Aim slightly below budget, allowing moderate timing noise.
    for probe in range(4):
        train(B,G,1000,M)
        value=float(np.median([train(B,G,seed,M)[2]['wall_seconds'] for seed in SEEDS]))
        records.append(dict(M=M,wall_seconds=value))
        if .90<=value/gd<=1.0 or probe==3:break
        M=max(100,int(round(M*.95*gd/value/100))*100)
    return M,dict(gd_wall_seconds=gd,probes=records)


def study(m):
    d=m*m;N=d//2;B,G,checks=model(m)
    E0=float(np.sqrt(np.ones(d)@G@np.ones(d)))
    matched,calibration=calibrate(B,G)
    methods=[0]+sorted(set(SIZES+[matched]))
    print(json.dumps(dict(d=d,matched_M=matched,calibration=calibration)),flush=True)
    spaces={};parameters={};timings=[];validation=[]
    old=np.load(BASE/'square_dimension_study'/f'd{d}_snapshots.npz')
    for M in methods:
        for seed in SEEDS:
            Q,Z,_=train(B,G,seed,M)
            spaces[(M,seed)]=Q;parameters[f'z_s{seed}_'+('gd80' if M==0 else f'pool{M}')]=Z
            if M in [0,256,4096,10000]:
                name='gd80' if M==0 else f'pool{M}'
                diff=float(np.max(np.abs(Z-old[f'z_s{seed}_{name}'])))
                assert diff<1e-12,(d,M,seed,diff)
                validation.append(dict(M=M,seed=seed,maximum_parameter_difference=diff))
    jobs=[(M,seed) for M in methods for seed in SEEDS]
    order=np.random.default_rng(814000+d)
    for repeat in range(REPEATS):
        for k in order.permutation(len(jobs)):
            M,seed=jobs[k]
            _,_,measured=train(B,G,seed,M)
            timings.append(dict(M=M,seed=seed,repeat=repeat,**measured))
        print(f'd={d}: timing repeat {repeat+1}/{REPEATS}',flush=True)
    np.savez_compressed(OUT/f'd{d}_snapshots.npz',**parameters)
    # Preserve every previous test point and add searches from every new space.
    previous=np.load(BASE/'square_dimension_study'/f'd{d}_test.npz')['parameters']
    tests=[previous];extra=[]
    rng=np.random.default_rng(815000+d)
    for (M,seed),Q in spaces.items():
        R=B-Q@(Q.T@B);H=R.T@R
        _,v=np.linalg.eigh(H)
        signs=np.where(v[:,-8:].T>=0,1.,-1.)
        tests.extend([signs,-signs,extrema(H,rng.uniform(-1,1,(128,d)),steps=240)])
        extra.append(extrema(H,rng.uniform(-1,1,(128,d)),steps=240))
    test_base=np.unique(np.vstack(tests),axis=0)
    test=np.unique(np.vstack([test_base]+extra),axis=0)
    np.savez_compressed(OUT/f'd{d}_test.npz',parameters=test)
    results=[]
    for (M,seed),Q in spaces.items():
        R=B-Q@(Q.T@B);H=R.T@R
        error=float(np.sqrt(np.max(np.sum((test@R.T)**2,axis=1)))/E0)
        old_error=float(np.sqrt(np.max(np.sum((previous@R.T)**2,axis=1)))/E0)
        base_error=float(np.sqrt(np.max(np.sum((test_base@R.T)**2,axis=1)))/E0)
        results.append(dict(M=M,seed=seed,error_over_E0=error,
            old_test_relative_increase=error/old_error-1,
            second_search_relative_increase=error/base_error-1,
            mean_squared_error=float(np.trace(H)/3),
            orthogonality_defect=float(np.linalg.norm(Q.T@Q-np.eye(N)))))
    summary=[]
    for M in methods:
        t=[r for r in timings if r['M']==M];e=[r for r in results if r['M']==M]
        summary.append(dict(M=M,error_median=float(np.median([r['error_over_E0'] for r in e])),
            error_minimum=min(r['error_over_E0'] for r in e),error_maximum=max(r['error_over_E0'] for r in e),
            wall_median=float(np.median([r['wall_seconds'] for r in t])),
            wall_q25=float(np.quantile([r['wall_seconds'] for r in t],.25)),
            wall_q75=float(np.quantile([r['wall_seconds'] for r in t],.75)),
            cpu_median=float(np.median([r['cpu_seconds'] for r in t])),
            setup_median=float(np.median([r['setup_seconds'] for r in t]))))
    gd=next(r for r in summary if r['M']==0)
    affordable=[r for r in summary if r['M']>0 and r['wall_median']<=gd['wall_median']]
    comparison=max(affordable,key=lambda r:r['M']) if affordable else None
    paired=[]
    if comparison:
        for seed in SEEDS:
            eg=next(r['error_over_E0'] for r in results if r['M']==0 and r['seed']==seed)
            ep=next(r['error_over_E0'] for r in results if r['M']==comparison['M'] and r['seed']==seed)
            paired.append(dict(seed=seed,gd_error=eg,pool_error=ep,gd_relative_improvement=1-eg/ep))
    report=dict(settings=dict(d=d,N=N,m=m,T=80,alpha=.6,seeds=SEEDS,repeats=REPEATS,
                device='CPU, float64, single BLAS thread',sizes=methods,calibrated_M=matched,
                compared_M=None if comparison is None else comparison['M'],
                initial_global_maximum=E0,test_count=len(test),old_test_count=len(previous),
                test='Previous complete common test set, plus leading-eigenvector signs and two independent 128-start vertex searches per new space; common union for all methods.',
                finite_test_is_global=False),checks=checks,calibration=calibration,
                timings=timings,results=results,summary=summary,paired=paired,validation=validation)
    (OUT/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(d=d,summary=summary,compared_M=None if comparison is None else comparison['M'],
        max_test_increase=max(r['second_search_relative_increase'] for r in results))),flush=True)
    return report


def outputs(reports):
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['d','N','M','error_median','wall_ms','cpu_ms','time_over_gd','error_over_gd'])
        for r in reports:
            gd=next(s for s in r['summary'] if s['M']==0)
            for s in r['summary']:
                writer.writerow([r['settings']['d'],r['settings']['N'],s['M'],s['error_median'],1000*s['wall_median'],1000*s['cpu_median'],s['wall_median']/gd['wall_median'],s['error_median']/gd['error_median']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.5))
    for ax,r in zip(axes.flat,reports):
        gd=next(s for s in r['summary'] if s['M']==0);pool=sorted([s for s in r['summary'] if s['M']>0],key=lambda s:s['M'])
        x=[s['wall_median']/gd['wall_median'] for s in pool];y=[s['error_median']/gd['error_median'] for s in pool]
        ax.plot(x,y,'o-',color='#D55E00',ms=5,label='Random pool')
        ax.scatter([1],[1],marker='*',s=130,color='#0072B2',label='GD, T=80',zorder=5)
        for s,xx,yy in zip(pool,x,y):
            if s['M'] in [10000,100000,r['settings']['compared_M']]:
                ax.annotate(f"{s['M']:,}",(xx,yy),xytext=(4,6),textcoords='offset points',fontsize=9)
        ax.axhline(1,color='#999999',lw=.8);ax.axvline(1,color='#999999',lw=.8,ls='--')
        ax.set_xscale('log');ax.set_title(f"d={r['settings']['d']}, N={r['settings']['N']}")
        ax.set_xlabel('Training time / GD training time');ax.set_ylabel('Maximum test error / GD error')
        ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'error_vs_time.png',dpi=180);fig.savefig(OUT/'error_vs_time.pdf');plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(dict(environment=dict(platform=platform.platform(),numpy=np.__version__,repeats=REPEATS),
        timing='Sequential randomized full retraining, sampling/precomputation/search/enrichment included; shared model construction and testing excluded.',
        reports=[{k:r[k] for k in ['settings','summary','paired','calibration']} for r in reports]),indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[study(m) for m in [5,8,12,16]]
    outputs(reports)
    print('COMPLETE',flush=True)
