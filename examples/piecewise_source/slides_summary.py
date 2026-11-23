"""Reproduce slide figures from saved runs; add a d=4 uniform-grid check.

No high-dimensional training is repeated. Source histories replay a saved GD
round and are checked against its selected parameter. All outputs use the
same energy Gram and update as run.py.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_slides_mpl')
from pathlib import Path
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import model, ascend, append, residual_gram

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'runs/piecewise_forcing_2d_20261112'
OUT = BASE/'slides_summary'
METHODS = ['gd80', 'pool256', 'pool4096', 'pool10000']
LABELS = ['GD ($T=80$)', 'Pool 256', 'Pool 4096', 'Pool 10000']
COLORS = ['#0072B2', '#555555', '#009E73', '#C74432']
MARKERS = ['o', 's', '^', 'D']
STYLES = ['-', ':', '-.', '--']
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'axes.labelsize':12})

def low_dimension():
    B,G,checks = model(2)
    test = np.array(list(itertools.product([-1.,1.], repeat=4)))
    E0 = np.sqrt(np.ones(4)@G@np.ones(4))
    rows=[]
    for seed in range(1000,1004):
        pool=np.random.default_rng(seed).uniform(-1,1,(10000,4))
        for method in METHODS+['grid1','grid0.5','grid0.25']:
            if method.startswith('grid') and seed!=1000: continue
            Q=np.zeros((4,0)); init=np.random.default_rng(seed+10000)
            if method.startswith('pool'): candidates=pool[:int(method[4:])]
            if method.startswith('grid'):
                h=float(method[4:]); candidates=np.array(list(itertools.product(np.linspace(-1,1,round(2/h)+1),repeat=4)))
            curve=[1.]; zs=[]
            for n in range(3):
                H=residual_gram(B,G,Q)
                if method=='gd80': z=ascend(H,init.uniform(-1,1,4),80)
                else: z=candidates[np.argmax(np.einsum('bi,ij,bj->b',candidates,H,candidates))]
                Q,_=append(B,Q,z); zs.append(z.tolist())
                R=B-Q@(Q.T@B)
                curve.append(float(np.linalg.norm(test@R.T,axis=1).max()/E0))
            rows.append(dict(seed=seed,method=method,error_over_E0=curve,parameters=zs))
    result=dict(settings=dict(d=4,N=2,T=80,alpha=.6,initial_global_maximum=float(E0),seeds=list(range(1000,1004)),test='All 16 vertices',
                             grid_h=[1,.5,.25],device='CPU, float64, one BLAS thread'),checks=checks,curves=rows)
    (OUT/'d4.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def median_curve(data,method):
    return data['settings']['initial_global_maximum'] * np.median([r['error_over_E0'] for r in data['curves'] if r['method']==method],axis=0)

def shared_test_errors(data):
    """Re-evaluate saved bases on unoptimized interior + vertex samples only."""
    reports={}
    for d,dat in sorted(data.items()):
        B,G,checks=model(round(np.sqrt(d)))
        rng=np.random.default_rng(71000+d)
        test=np.vstack([rng.uniform(-1,1,(4096,d)),rng.choice([-1.,1.],(4096,d))])
        U=test@B.T; norm2=np.sum(U*U,axis=1)
        E0=dat['settings']['initial_global_maximum']
        if d!=4:
            folder='dimension_study' if d==16 else 'square_dimension_study'
            saved=np.load(BASE/folder/f'd{d}_snapshots.npz')
        for row in dat['curves']:
            parameters=np.array(row['parameters']) if d==4 else saved[f"z_s{row['seed']}_{row['method']}"]
            Q=np.zeros((d,0))
            for z in parameters: Q,_=append(B,Q,z)
            errors2=np.maximum(0,norm2[:,None]-np.cumsum((U@Q)**2,axis=1))
            row['error_over_E0']=(np.r_[np.sqrt(norm2.max()),np.sqrt(errors2.max(axis=0))]/E0).tolist()
        np.savez_compressed(OUT/f'd{d}_test.npz',parameters=test)
        reports[d]=dict(test_seed=71000+d,interior_count=4096,vertex_count=4096,
                       curves=[dict(seed=r['seed'],method=r['method'],energy_error=(E0*np.array(r['error_over_E0'])).tolist()) for r in dat['curves']])
        print(f'd={d}: shared interior/vertex errors recomputed',flush=True)
    (OUT/'shared_test_errors.json').write_text(json.dumps(reports,indent=2)+'\n')

def style(ax):
    ax.grid(alpha=.18); ax.set_ylabel(r'$E_N$')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    low=low_dimension()
    dims=list(range(5,17)); dims=[m*m for m in dims]
    data={d:json.loads((BASE/'square_dimension_study'/f'd{d}.json').read_text()) for d in dims}
    sixteen=json.loads((BASE/'dimension_study/d16.json').read_text())
    for row in sixteen['curves']:
        row['error_over_E0']=row['error_over_initial_maximum']
    data.update({4:low,16:sixteen})
    dims=[4,16]+dims
    shared_test_errors(data)
    fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained')
    for meth,label,col,mark,ls in zip(METHODS,LABELS,COLORS,MARKERS,STYLES):
        ax.plot(dims,[median_curve(data[d],meth)[d//2] for d in dims],label=label,color=col,marker=mark,ls=ls,ms=5,lw=1.8)
    ax.set_xlabel(r'Parameter dimension $d$'); style(ax)
    ax.set_xscale('log',base=2); ax.set_xticks([4,16,32,64,128,256],labels=['4','16','32','64','128','256']); ax.legend(frameon=False,ncol=2)
    fig.savefig(OUT/'piecewise_source_error_vs_d.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,5.6),layout='constrained')
    for ax,d in zip(axes.ravel(),[4,16,64,256]):
        dat=low if d==4 else data[d]
        for meth,label,col,mark,ls in zip(METHODS,LABELS,COLORS,MARKERS,STYLES):
            values=median_curve(dat,meth); ax.semilogy(range(len(values)),values,label=label,color=col,marker=mark,markevery=max(1,len(values)//7),ms=3.5,ls=ls,lw=1.4)
        if d==4:
            for meth,ls in [('grid1','--'),('grid0.5',':'),('grid0.25','-.')]:
                values=median_curve(low,meth)
                ax.semilogy(range(4),values,ls=ls,color='#9A6B26',lw=1.2,label='Grid '+meth[4:])
            ax.set_xticks(range(4))
        ax.set_title(f'$d={d}$'); ax.set_xlabel('$N$');style(ax)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False,fontsize=10)
    fig.savefig(OUT/'piecewise_source_error_vs_N.pdf');plt.close(fig)
    # Fixed actual GD round: first eight saved snapshots, ninth initialization.
    B,G,checks=model(8)
    saved=np.load(BASE/'square_dimension_study/d64_snapshots.npz')['z_s1000_gd80']
    Q=np.zeros((64,0))
    for z in saved[:8]: Q,_=append(B,Q,z)
    H=residual_gram(B,G,Q); ell=np.linalg.norm(H)
    rng=np.random.default_rng(11000)
    for _ in range(9): z=rng.uniform(-1,1,64)
    history=[z.copy()]
    for _ in range(80):
        g=2*H@z/ell;z=np.clip(z+.6*g/np.sqrt(1+g@g),-1,1);history.append(z.copy())
    history=np.array(history)
    np.testing.assert_allclose(history[-1],saved[8],atol=2e-12,rtol=2e-12)
    objective=np.einsum('bi,ij,bj->b',history,H,history)/ell
    np.savez_compressed(OUT/'d64_n8_history.npz',parameters=history,J=objective,H=H)
    (OUT/'history.json').write_text(json.dumps(dict(d=64,n=8,seed=1000,T=80,alpha=.6,
        basis='First eight saved GD snapshots',J=objective.tolist(),checks=checks),indent=2)+'\n')
    fig=plt.figure(figsize=(10,5.3),layout='constrained')
    gs=fig.add_gridspec(2,5,height_ratios=[1,1])
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red',['#0072B2','#FAFAF8','#C53B32'])
    axes=[]
    for i,t in enumerate([0,20,40,60,80]):
        ax=fig.add_subplot(gs[0,i]); axes.append(ax)
        im=ax.imshow(history[t].reshape(8,8).T,origin='lower',extent=[0,1,0,1],cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
        ax.set_title(f'GD: $t={t}$'); ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('$x_1$')
        if i==0: ax.set_ylabel('$x_2$')
    # Reuse the original seed-1000 training pool, evaluated on the SAME V_8.
    pool=np.random.default_rng(1000).uniform(-1,1,(10000,64))
    pool_J=np.einsum('bi,ij,bj->b',pool,H,pool)/ell
    pool_selected={}
    for column,M in zip([1,2,3],[256,4096,10000]):
        index=int(np.argmax(pool_J[:M])); selected=pool[index]
        pool_selected[str(M)]=dict(index=index,J=float(pool_J[index]),parameter=selected.tolist())
        ax=fig.add_subplot(gs[1,column]);axes.append(ax)
        ax.imshow(selected.reshape(8,8).T,origin='lower',extent=[0,1,0,1],cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
        ax.set_title(f'Pool {M}');ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('$x_1$')
        if column==1: ax.set_ylabel('$x_2$')
    fig.colorbar(im,ax=axes,ticks=[-1,0,1],shrink=.72,label='$f(x;z)$')
    (OUT/'pool_source_comparison.json').write_text(json.dumps(dict(d=64,n=8,pool_seed=1000,
        basis='Same first eight GD snapshots as the trajectory',selected=pool_selected),indent=2)+'\n')
    fig.savefig(OUT/'piecewise_source_gd_history.pdf');plt.close(fig)
    table=[]
    for d in [4,16,25,64,100,144,256]:
        dat=low if d==4 else data[d]; N=d//2
        values=[median_curve(dat,meth)[N] for meth in METHODS]
        values += [median_curve(low,meth)[N] if d==4 else None for meth in ['grid1','grid0.5','grid0.25']]
        best=min(v for v in values if v is not None)
        cells=['---' if v is None else (r'$\mathbf{%.4f}$'%v if abs(v-best)<1e-12 else '$%.4f$'%v) for v in values]
        table.append(f'{d} & {N} & '+' & '.join(cells)+r'\\')
    (OUT/'table_rows.tex').write_text('\n'.join(table)+'\n')
    print('\n'.join(table));print('GD J initial/final:',objective[0],objective[-1])

if __name__=='__main__': main()
