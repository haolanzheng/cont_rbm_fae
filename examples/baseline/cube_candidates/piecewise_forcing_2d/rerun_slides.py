"""Recompute the piecewise-source slides with an explicit GD step budget.

Preserves the saved test sets, seeds, spectral model and figure layouts.
All methods are retrained. Timers exclude tests, validation and shared model
construction. Historical output directories are never overwritten.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/piecewise_rerun_mpl')
import argparse, json, itertools, hashlib, platform, csv
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import model, append, residual_gram
from equal_time_pool import train

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'runs/piecewise_forcing_2d_20261112/slides_summary'
DIMS=[4,16]+[m*m for m in range(5,17)]
TABLE_DIMS=[4,16,25,64,100,144,256]
SEEDS=list(range(1000,1004))
GRIDS=['grid1','grid0.5','grid0.25']
COLORS=['#0072B2','#555555','#009E73','#C74432']
MARKERS=['o','s','^','D']; STYLES=['-',':','-.','--']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':12})

def train_method(B,G,seed,method,steps,max_n=None):
    if not method.startswith('grid'):
        return train(B,G,seed,0 if method=='gd' else int(method[4:]),steps=steps,max_n=max_n)
    tick=perf_counter();d=len(B);N=d//2 if max_n is None else max_n
    h=float(method[4:]);points=np.array(list(itertools.product(np.linspace(-1,1,round(2/h)+1),repeat=d)))
    U=points@B.T;scores=np.sum(U**2,axis=1);Q=np.zeros((d,0));Z=[]
    for n in range(N):
        z=points[np.argmax(scores)];Q,q=append(B,Q,z);Z.append(z.copy())
        if n<N-1:scores-=(U@q)**2
    return Q,np.asarray(Z),dict(wall_seconds=perf_counter()-tick)

def median_curve(report,method):
    return np.median([r['energy_error'] for r in report['curves'] if r['method']==method],axis=0)

def experiments(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('d*.json')):raise FileExistsError('Choose a fresh output directory to preserve logs.')
    old=json.loads((BASE/'shared_test_errors.json').read_text());reports={}
    for d in DIMS:
        B,G,checks=model(round(np.sqrt(d)));N=d//2
        max_n=len(old[str(d)]['curves'][0]['energy_error'])-1
        test_file=BASE/f'd{d}_test.npz';test=np.load(test_file)['parameters']
        U=test@B.T;norm2=np.sum(U**2,axis=1)
        methods=['gd','pool256','pool4096','pool10000']+(['pool40000'] if d in TABLE_DIMS else [])+(GRIDS if d==4 else [])
        jobs=[(method,seed) for method in methods for seed in ([1000] if method.startswith('grid') else SEEDS)]
        curves=[];snapshots={};validation=[]
        for method,seed in jobs:
            Q,Z,_=train_method(B,G,seed,method,args.steps,max_n)
            snapshots[f'z_s{seed}_{method}']=Z
            e2=np.maximum(0,norm2[:,None]-np.cumsum((U@Q)**2,axis=1))
            curve=np.r_[np.sqrt(norm2.max()),np.sqrt(e2.max(axis=0))]
            direct=np.linalg.norm(test@(B-Q[:,:N]@(Q[:,:N].T@B)).T,axis=1).max()
            np.testing.assert_allclose(curve[N],direct,rtol=1e-11,atol=1e-13)
            assert np.all(np.diff(curve)<=1e-12)
            orth=float(np.linalg.norm(Q.T@Q-np.eye(max_n)));assert orth<1e-11
            ref=next((r for r in old[str(d)]['curves'] if r['method']==method and r['seed']==seed),None)
            if ref:np.testing.assert_allclose(curve,ref['energy_error'],rtol=1e-9,atol=1e-12)
            validation.append(dict(method=method,seed=seed,orthogonality_defect=orth,direct_error_difference=float(abs(direct-curve[N])),previous_pool_or_grid_error_matched=ref is not None))
            curves.append(dict(method=method,seed=seed,energy_error=curve.tolist()))
        print(f'd={d}: all training paths and test errors complete',flush=True)
        timings=[]
        if d in TABLE_DIMS:
            for method,seed in jobs:
                _,Z,_=train_method(B,G,seed,method,args.steps)
                np.testing.assert_array_equal(Z,snapshots[f'z_s{seed}_{method}'][:N])
            rng=np.random.default_rng(940000+d)
            for repeat in range(5):
                for k in rng.permutation(len(jobs)):
                    method,seed=jobs[k];_,_,t=train_method(B,G,seed,method,args.steps)
                    timings.append(dict(method=method,seed=seed,repeat=repeat,**t))
        summary=[]
        for method in methods:
            values=[r['energy_error'][N] for r in curves if r['method']==method]
            times=[r['wall_seconds'] for r in timings if r['method']==method]
            summary.append(dict(method=method,error_median=float(np.median(values)),error_min=min(values),error_max=max(values),wall_ms_median=float(1000*np.median(times)) if times else None))
        report=dict(d=d,N=N,max_plot_N=max_n,curves=curves,timings=timings,summary=summary,validation=validation,checks=checks,test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest())
        np.savez_compressed(out/f'd{d}_snapshots.npz',**snapshots)
        (out/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n');reports[d]=report
        print(json.dumps(dict(d=d,summary=summary)),flush=True)
    from threadpoolctl import threadpool_info
    libs=threadpool_info();assert all(x['num_threads']==1 for x in libs)
    settings=dict(T=args.steps,alpha=.6,seeds=SEEDS,dimensions=DIMS,N='floor(d/2)',timing_dimensions=TABLE_DIMS,repeats=5,
        tests='Unchanged 4096 uniform interior + 4096 uniform vertices; same saved test set for all methods, seeds and N.',
        timing='Sampling, search, matrix construction, score maintenance and basis enrichment included; shared model, test evaluation, validation and I/O excluded. Randomized sequential order after warmup.',
        platform=platform.platform(),numpy=np.__version__,threadpools=libs)
    (out/'summary.json').write_text(json.dumps(dict(settings=settings,reports=[{k:r[k] for k in ['d','N','summary']} for r in reports.values()]),indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','N','method','error_median','wall_ms_median'])
        for r in reports.values():
            for s in r['summary']:w.writerow([r['d'],r['N'],s['method'],s['error_median'],s['wall_ms_median']])
    return reports

def figures(reports,args):
    out=args.output;methods=['gd','pool256','pool4096','pool10000']
    labels=[f'GD ($T={args.steps}$)','Pool 256','Pool 4096','Pool 10000']
    fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained')
    for method,label,c,m,ls in zip(methods,labels,COLORS,MARKERS,STYLES):
        ax.plot(DIMS,[median_curve(reports[d],method)[d//2] for d in DIMS],label=label,color=c,marker=m,ls=ls,ms=5,lw=1.8)
    ax.set_xlabel('Parameter dimension $d$');ax.set_ylabel('$E_N$');ax.grid(alpha=.18)
    ax.set_xscale('log',base=2);ax.set_xticks([4,16,32,64,128,256],labels=['4','16','32','64','128','256']);ax.legend(frameon=False,ncol=2)
    fig.savefig(out/'piecewise_source_error_vs_d.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,5.6),layout='constrained')
    for ax,d in zip(axes.flat,[4,16,64,256]):
        for method,label,c,m,ls in zip(methods,labels,COLORS,MARKERS,STYLES):
            values=median_curve(reports[d],method)
            ax.semilogy(range(len(values)),values,label=label,color=c,marker=m,markevery=max(1,len(values)//7),ms=3.5,ls=ls,lw=1.4)
        if d==4:
            for method,ls in zip(GRIDS,['--',':','-.']):
                ax.semilogy(range(4),median_curve(reports[d],method),ls=ls,color='#9A6B26',lw=1.2,label='Grid '+method[4:])
            ax.set_xticks(range(4))
        ax.set_title(f'$d={d}$');ax.set_xlabel('$N$');ax.set_ylabel('$E_N$');ax.grid(alpha=.18)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=4,frameon=False,fontsize=10)
    fig.savefig(out/'piecewise_source_error_vs_N.pdf');plt.close(fig)
    error_rows=[];time_rows=[]
    for d in TABLE_DIMS:
        summary={r['method']:r for r in reports[d]['summary']}
        best=min(summary[m]['error_median'] for m in methods+GRIDS if m in summary)
        errors=[];times=[]
        for method in methods+GRIDS:
            if method not in summary:errors.append('---');times.append('---');continue
            value=summary[method]['error_median']
            errors.append((r'$\mathbf{%.4f}$' if abs(value-best)<1e-12 else '$%.4f$')%value)
            times.append('$%.2f$'%summary[method]['wall_ms_median'])
        error_rows.append(f'{d} & {d//2} & '+' & '.join(errors)+r'\\')
        time_rows.append(f'{d} & {d//2} & '+' & '.join(times)+r'\\')
    (out/'table_rows.tex').write_text('\n'.join(error_rows)+'\n')
    (out/'timing_rows.tex').write_text('\n'.join(time_rows)+'\n')

def histories(args):
    out=args.output;B,G,_=model(8);Z=np.load(out/'d64_snapshots.npz')['z_s1000_gd']
    Q=np.zeros((64,0))
    for z in Z[:8]:Q,_=append(B,Q,z)
    H=residual_gram(B,G,Q);ell=np.linalg.norm(H)
    rng=np.random.default_rng(11000)
    for _ in range(9):z=rng.uniform(-1,1,64)
    history=[z.copy()];hz=H@z
    for _ in range(args.steps):
        g=2*hz/ell;z=np.clip(z+.6*g/np.sqrt(1+g@g),-1,1);hz=H@z;history.append(z.copy())
    history=np.asarray(history);objective=np.einsum('bi,ij,bj->b',history,H,history)/ell
    np.testing.assert_allclose(history[np.argmax(objective)],Z[8],atol=2e-12,rtol=2e-12)
    pool=np.random.default_rng(1000).uniform(-1,1,(10000,64));scores=np.einsum('bi,ij,bj->b',pool,H,pool)/ell
    selected={str(M):dict(parameter=pool[np.argmax(scores[:M])].tolist(),J=float(scores[:M].max())) for M in [256,4096,10000]}
    np.savez_compressed(out/'d64_n8_history.npz',parameters=history,J=objective,H=H)
    (out/'pool_source_comparison.json').write_text(json.dumps(dict(d=64,n=8,T=args.steps,selected=selected),indent=2)+'\n')
    fig=plt.figure(figsize=(10,5.3),layout='constrained');gs=fig.add_gridspec(2,5)
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red',['#0072B2','#FAFAF8','#C53B32']);axes=[]
    for i,t in enumerate(np.linspace(0,args.steps,5,dtype=int)):
        ax=fig.add_subplot(gs[0,i]);axes.append(ax)
        im=ax.imshow(history[t].reshape(8,8).T,origin='lower',extent=[0,1,0,1],cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
        ax.set_title(f'GD: $t={t}$');ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('$x_1$')
        if i==0:ax.set_ylabel('$x_2$')
    for col,M in zip([1,2,3],[256,4096,10000]):
        ax=fig.add_subplot(gs[1,col]);axes.append(ax)
        ax.imshow(np.array(selected[str(M)]['parameter']).reshape(8,8).T,origin='lower',extent=[0,1,0,1],cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
        ax.set_title(f'Pool {M}');ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('$x_1$')
        if col==1:ax.set_ylabel('$x_2$')
    fig.colorbar(im,ax=axes,ticks=[-1,0,1],shrink=.72,label='$f(x;z)$')
    fig.savefig(out/'piecewise_source_gd_history.pdf');plt.close(fig)
    test=np.load(BASE/'d64_test.npz')['parameters'];R=B-Q@(Q.T@B);before=np.sum((test@R.T)**2,axis=1)
    def evaluate(z):
        den=float(z@H@z);Qnew,_=append(B,Q,z)
        direct=np.sum((test@(B-Qnew@(Qnew.T@B)).T)**2,axis=1)
        identity=before-(test@H@z)**2/den
        np.testing.assert_allclose(direct,identity,atol=1e-14,rtol=1e-10)
        return dict(J=den/ell,E9=float(np.sqrt(direct.max())),identity_max_abs_error=float(np.max(abs(identity-direct))))
    gd=[dict(t=t,**evaluate(z)) for t,z in enumerate(history)]
    pools=[dict(M=M,**evaluate(np.array(selected[str(M)]['parameter']))) for M in [256,4096,10000]]
    (out/'enrichment.json').write_text(json.dumps(dict(T=args.steps,d=64,n=8,E8=float(np.sqrt(before.max())),gd=gd,pools=pools),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8.4,4.2),layout='constrained')
    ax.plot(range(args.steps+1),[r['E9'] for r in gd],color=COLORS[0],lw=2,marker='o',markevery=max(1,args.steps//4),ms=5,label='GD candidate')
    for r,c,ls in zip(pools,COLORS[1:],STYLES[1:]):ax.axhline(r['E9'],color=c,ls=ls,lw=1.8,label=f"Pool {r['M']}")
    ax.set_xlabel('GD iteration $t$');ax.set_ylabel('$E_9$');ax.set_xlim(0,args.steps);ax.set_xticks(np.linspace(0,args.steps,5))
    ax.grid(alpha=.18);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.legend(frameon=False,ncol=4,fontsize=10.5,loc='lower center',bbox_to_anchor=(.5,1.02))
    fig.savefig(out/'piecewise_source_enrichment_history.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--figures-only',action='store_true');args=p.parse_args()
    assert args.steps>0
    reports={d:json.loads((args.output/f'd{d}.json').read_text()) for d in DIMS} if args.figures_only else experiments(args)
    figures(reports,args);histories(args);print('COMPLETE',flush=True)
