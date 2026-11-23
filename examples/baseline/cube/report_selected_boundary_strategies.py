"""Reevaluate saved cube spaces for three boundary-update slides; no retraining."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/baseline_maximum_plot')
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
from ..geometry_cpu import CoordinateROM, rom_options

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'runs/cube_boundary_strategies_20261102'
MODEL=ROOT/'runs/analytic_hf_a06_t40_20261022'
TEST=ROOT/'runs/cube_slides_a06_t80_20261102'
OUT=ROOT/'runs/cube_boundary_slides_20261109'
FIG=ROOT/'paper/slides_short/figs'
STYLES=[('clip','Clipping','#245E91','-','o'),
        ('active_face','Active face','#BC762C','--','^'),
        ('normal_scaled','Adaptive normal updates','#477A68','-.','s')]


def main():
    OUT.mkdir(exist_ok=True)
    data=json.loads((SOURCE/'greedy.json').read_text())
    selected=[r for r in data if r['method'] in [s[0] for s in STYLES]]
    summary=[]
    for d in [6,8,16,32,48,64]:
        for method,*_ in STYLES:
            rr=[r for r in selected if r['d']==d and r['method']==method]
            assert len(rr)==8
            summary.append(dict(d=d,N=d//2,method=method,
                error=float(np.median([r['test_max_error'] for r in rr])),
                ratio=float(np.median([r['ratio'] for r in rr]))))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    curves=[];max_relative_difference=0.
    for d in [6,32,64]:
        model=np.load(MODEL/f'model_d{d}.npz')
        Z=np.load(TEST/f'test_vertices_d{d}.npz')['parameters']
        G=model['G0']; cross=model['cross'] if 'cross' in model else G
        base=np.sum((Z@G)*Z,axis=1)
        for row in [r for r in selected if r['d']==d]:
            Q=np.load(SOURCE/f"basis_d{d}_{row['method']}_s{row['seed']}.npz")['Q']
            # Q.T G Q=I. For prefix n, c=Q_n.T cross z and
            # ||u-u_n||_A^2=z.T G z + sum_{j<=n}(c_j^2-2 c_j q_j.T G z).
            coefficients=Z@(cross.T@Q); inner=Z@(G@Q)
            squared=base[:,None]+np.cumsum(coefficients**2-2*coefficients*inner,axis=1)
            curve=np.sqrt(np.maximum(0,np.r_[base.max(),squared.max(axis=0)]))
            # Independently check the prefix formula at every n using the
            # original error Gram matrix on a fixed subset of the same tests.
            rom=CoordinateROM(G,model['H0'],**rom_options(model))
            check=Z[:min(257,len(Z))]
            for n in range(1,Q.shape[1]+1):
                rom.Q=Q[:,:n];_,Gn=rom.matrices()
                direct=np.sum((check@Gn)*check,axis=1)
                assert np.allclose(direct,squared[:len(check),n-1],rtol=2e-8,atol=3e-13)
            diff=abs(curve[-1]-row['test_max_error'])/row['test_max_error']
            max_relative_difference=max(max_relative_difference,float(diff))
            assert diff<2e-8
            curves.append(dict(d=d,method=row['method'],seed=row['seed'],
                N=list(range(len(curve))),error=curve.tolist(),test_count=len(Z)))
        print(f'd={d}: three methods, eight seeds, all saved basis prefixes evaluated',flush=True)
    (OUT/'curves.json').write_text(json.dumps(curves,indent=2)+'\n')
    (OUT/'checks.json').write_text(json.dumps(dict(max_final_relative_difference=max_relative_difference,
        source=str(SOURCE),training_rerun=False,curve_aggregation='median of per-seed test maxima',
        test_points='same saved vertices as the boundary-strategy study'),indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8','savefig.facecolor':'#FAFAF8'})
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    for d,ax in zip([6,32,64],axes):
        for method,label,color,ls,marker in STYLES:
            rr=[r for r in curves if r['d']==d and r['method']==method]
            values=np.median([r['error'] for r in rr],axis=0)
            ax.semilogy(range(len(values)),values,label=label,color=color,ls=ls,
                lw=2.3,marker=marker,ms=5,mfc='white',markevery=max(1,(d//2)//4))
        ax.set(title=f'$d={d}$',xlabel='Reduced dimension $N$',
            xticks=([0,1,2,3] if d==6 else list(range(0,d//2+1,d//8))))
        ax.grid(alpha=.16)
        if d==6:
            ax.yaxis.set_major_locator(FixedLocator([1.35,1.40,1.45,1.50,1.55,1.60]))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel('Maximum energy error')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.075,right=.99,top=.88,bottom=.24,wspace=.3)
    for ext in ['pdf','png']:
        fig.savefig(FIG/f'cube_boundary_updates_error_vs_N.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Figure and checks saved.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
