"""Projected Adam versus GD80 on the unchanged piecewise-source baseline.

Learning-rate sweep is exploratory, not a held-out tuning benchmark. All rates
are reported. Both optimizers return the best feasible iterate, including the
initial point. Adam moments restart at every greedy round; no weight decay.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/piecewise_adam_mpl')
import argparse,json,csv,hashlib,platform
from pathlib import Path
from time import perf_counter
import numpy as np
from run import model,ascend,append,residual_gram

BASE=Path(__file__).resolve().parents[4]/'runs/piecewise_forcing_2d_20261112/slides_summary'
DIMS=[4,16,25,64,100,144,256];SEEDS=list(range(1000,1004))
RATES=[.01,.03,.1,.3,1.]

def adam(H,z,rate,steps=80):
    z=z.copy();ell=max(float(np.linalg.norm(H)),1e-30)
    m=np.zeros_like(z);v=np.zeros_like(z);b1pow=b2pow=1.
    hz=H@z;best=z.copy();best_value=float(z@hz)
    for _ in range(steps):
        g=2*hz/ell
        m=.9*m+.1*g;v=.999*v+.001*g*g
        b1pow*=.9;b2pow*=.999
        z=np.clip(z+rate*(m/(1-b1pow))/(np.sqrt(v/(1-b2pow))+1e-8),-1,1)
        hz=H@z;value=float(z@hz)
        if value>best_value:best_value=value;best=z.copy()
    return best

def validate():
    rng=np.random.default_rng(72);B=rng.normal(size=(5,5));H=B.T@B;initial=rng.uniform(-.3,.3,5)
    g=2*H@initial/np.linalg.norm(H)
    expected=np.clip(initial+.03*g/(abs(g)+1e-8),-1,1)
    np.testing.assert_allclose(adam(H,initial,.03,1),expected,atol=1e-14)
    # Independent scalar implementation checks moments, bias correction,
    # projection, and best-iterate selection over multiple steps.
    z=initial.copy();m=np.zeros(5);v=np.zeros(5);best=z.copy();score=z@H@z
    for t in range(1,21):
        g=2*H@z/np.linalg.norm(H)
        for i in range(5):
            m[i]=.9*m[i]+.1*g[i];v[i]=.999*v[i]+.001*g[i]**2
            z[i]=max(-1,min(1,z[i]+.03*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)))
        if z@H@z>score:best=z.copy();score=z@H@z
    np.testing.assert_allclose(adam(H,initial,.03,20),best,atol=2e-13)
    return dict(first_step=True,scalar_reference=True)

def train(B,G,seed,rate,N):
    tick=perf_counter();d=len(B);Q=np.zeros((d,0));rng=np.random.default_rng(seed+10000);params=[]
    for _ in range(N):
        H=residual_gram(B,G,Q);z=rng.uniform(-1,1,d)
        z=ascend(H,z,80) if rate is None else adam(H,z,rate)
        Q,_=append(B,Q,z);params.append(z)
    return Q,np.asarray(params),perf_counter()-tick

def name(rate):return 'gd80' if rate is None else f'adam{rate:g}'

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('d*.json')):raise FileExistsError('Use a new output directory.')
    validation=validate();reports=[]
    for d in DIMS:
        B,G,checks=model(round(np.sqrt(d)));test_file=BASE/f'd{d}_test.npz'
        test=np.load(test_file)['parameters'];U=test@B.T
        old=json.loads((BASE/'full_rank_20261116'/f'd{d}.json').read_text())
        ref=np.load(BASE/'full_rank_20261116'/f'd{d}_snapshots.npz')
        jobs=[(rate,seed) for rate in [None]+RATES for seed in SEEDS]
        curves=[];snapshots={};frozen=[]
        for rate,seed in jobs:
            method=name(rate);Q,Z,_=train(B,G,seed,rate,d-1)
            assert np.isfinite(Q).all();orth=float(np.linalg.norm(Q.T@Q-np.eye(d-1)));assert orth<1e-10
            full,_=np.linalg.qr(Q,mode='complete');coef=U@full
            errors=np.sqrt(np.cumsum((coef**2)[:,::-1],axis=1)[:,::-1].max(axis=0))
            for n in [d//2,d-1]:
                direct=np.linalg.norm(test@(B-Q[:,:n]@(Q[:,:n].T@B)).T,axis=1).max()
                np.testing.assert_allclose(errors[n],direct,atol=1e-13,rtol=1e-10)
            if rate is None:
                np.testing.assert_array_equal(Z,ref[f'z_s{seed}_gd80'])
                before=next(r['energy_error'] for r in old['curves'] if r['method']=='gd80' and r['seed']==seed)
                np.testing.assert_allclose(errors,before,rtol=1e-12,atol=1e-14)
            curves.append(dict(method=method,seed=seed,energy_error=errors.tolist(),orthogonality_defect=orth))
            snapshots[f'z_s{seed}_{method}']=Z
        # Same H and same initial point: isolate inner-search quality from
        # the effect of building different spaces during full greedy runs.
        for seed in SEEDS:
            Q=np.zeros((d,0));n=d//2-1
            for z in ref[f'z_s{seed}_gd80'][:n]:Q,_=append(B,Q,z)
            H=residual_gram(B,G,Q);ell=np.linalg.norm(H);rng=np.random.default_rng(seed+10000)
            for _ in range(n+1):initial=rng.uniform(-1,1,d)
            zg=ascend(H,initial,80);jg=float(zg@H@zg/ell)
            for rate in RATES:
                za=adam(H,initial,rate)
                frozen.append(dict(seed=seed,n=n,rate=rate,J_gd=jg,J_adam=float(za@H@za/ell),initial_J=float(initial@H@initial/ell)))
        print(f'd={d}: all full paths, errors and paired fixed-space searches complete',flush=True)
        for rate,seed in jobs:
            _,Z,_=train(B,G,seed,rate,d//2)
            np.testing.assert_array_equal(Z,snapshots[f'z_s{seed}_{name(rate)}'][:d//2])
        order=np.random.default_rng(171100+d);timings=[]
        for repeat in range(5):
            for idx in order.permutation(len(jobs)):
                rate,seed=jobs[idx];_,_,elapsed=train(B,G,seed,rate,d//2)
                timings.append(dict(method=name(rate),seed=seed,repeat=repeat,wall_seconds=elapsed))
        summary=[]
        for rate in [None]+RATES:
            method=name(rate);rows=[r for r in curves if r['method']==method]
            values=np.array([r['energy_error'] for r in rows]);median=np.median(values,axis=0)
            summary.append(dict(method=method,rate=rate,median_curve=median.tolist(),
                E_half=float(median[d//2]),E_last=float(median[-1]),E_half_min=float(values[:,d//2].min()),E_half_max=float(values[:,d//2].max()),
                wall_ms=float(1000*np.median([r['wall_seconds'] for r in timings if r['method']==method])),
                frozen_J_ratio=None if rate is None else float(np.median([r['J_adam']/r['J_gd'] for r in frozen if r['rate']==rate]))))
        report=dict(d=d,checks=checks,curves=curves,frozen=frozen,timings=timings,summary=summary,test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest())
        np.savez_compressed(out/f'd{d}_snapshots.npz',**snapshots)
        (out/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        print(json.dumps(dict(d=d,summary=[{k:v for k,v in s.items() if k!='median_curve'} for s in summary])),flush=True)
    from threadpoolctl import threadpool_info
    libs=threadpool_info();assert all(r['num_threads']==1 for r in libs)
    settings=dict(T=80,GD_alpha=.6,adam_rates=RATES,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,
        moment_reset='Every greedy round',weight_decay=0,selection='Best feasible objective value among initial point and 80 updates',
        dimensions=DIMS,seeds=SEEDS,N_max='d-1',timing_N='floor(d/2)',repeats=5,
        tests='Unchanged saved 4096 uniform interior + 4096 uniform vertices',
        initialization='Same uniform initialization stream at each round; each method builds its own space',
        timing='Full training including matrix construction, initialization, optimization and enrichment; excludes shared model, tests, validation and I/O; randomized order after warmup',
        interpretation='Exploratory learning-rate sweep; test results are reported for every rate, not a held-out evaluation of a tuned optimizer',
        threadpools=libs,platform=platform.platform(),validation=validation)
    (out/'summary.json').write_text(json.dumps(dict(settings=settings,reports=[dict(d=r['d'],summary=r['summary']) for r in reports]),indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','method','E_half','E_last','wall_ms','frozen_J_ratio'])
        for r in reports:
            for s in r['summary']:w.writerow([r['d'],s['method'],s['E_half'],s['E_last'],s['wall_ms'],s['frozen_J_ratio']])
    return reports

def plots(reports,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#0072B2','#999999','#E69F00','#009E73','#D55E00','#CC79A7']
    markers=['o','v','s','^','D','x'];lines=['-',':','--','-.','--',':']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for i,rate in enumerate([None]+RATES):
        xs=[];ys=[]
        for r in reports:
            gd=r['summary'][0]['E_half'];s=next(s for s in r['summary'] if s['method']==name(rate));xs.append(r['d']);ys.append(s['E_half']/gd)
        ax.plot(xs,ys,color=colors[i],marker=markers[i],ls=lines[i],label='GD' if rate is None else f'Adam, lr={rate:g}')
    ax.set_xscale('log',base=2);ax.set_xticks(DIMS,labels=[str(d) for d in DIMS]);ax.set_xlabel('$d$');ax.set_ylabel('Final test error / GD final test error')
    ax.set_title(r'$N=\lfloor d/2\rfloor$, $T=80$');ax.grid(alpha=.18);ax.legend(frameon=False,ncol=3)
    fig.savefig(out/'adam_learning_rates.pdf');fig.savefig('/tmp/adam_learning_rates.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,d in zip(axes.flat,[16,64,144,256]):
        r=next(r for r in reports if r['d']==d)
        for i,rate in enumerate([None]+RATES):
            s=next(s for s in r['summary'] if s['method']==name(rate));y=s['median_curve']
            ax.semilogy(range(d),y,color=colors[i],marker=markers[i],markevery=max(1,d//12),ms=3,ls=lines[i],lw=1.4,label='GD' if rate is None else f'Adam, lr={rate:g}')
        ax.set_title(f'$d={d}$');ax.set_xlabel('$N$');ax.set_ylabel('$E_N$');ax.grid(alpha=.18)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=3,frameon=False)
    fig.savefig(out/'adam_error_vs_N.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--plots-only',action='store_true');args=p.parse_args()
    reports=[json.loads((args.output/f'd{d}.json').read_text()) for d in DIMS] if args.plots_only else main(args.output)
    plots(reports,args.output);print('COMPLETE',flush=True)
