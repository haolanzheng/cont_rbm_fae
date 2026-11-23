"""Exact sphere/ball supremum errors of saved ROMs, versus analytic widths.

For the discrete analytic snapshots S and reduced Galerkin map S M,
G=(I-M)^T G0 (I-M), G0=S^T A S. The maximum squared error over
either the unit sphere or ball is lambda_max(G), not a test-set maximum.
Run from the repository root: PYTHONPATH=src:. .venv_cpu/bin/python -m
examples.baseline.common.plot_meeting_worst_width
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/meeting_worst_width_mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.baseline.geometry_cpu import CoordinateROM,rom_options
from examples.baseline.common.plot_dimension_sweep import STYLES

def main():
    root=Path(__file__).resolve().parents[3]
    old=root/'runs/analytic_hf_a06_t40_20261022'
    sweep=root/'runs/baseline_dimension_sweep_20261102'
    out=root/'runs/meeting_20261113'
    figdir=root/'paper/slides_meeting_20261113/figs'
    saved=json.loads((old/'results.json').read_text())['runs']
    dims=list(range(8,65,4));records=[];summary=[]
    for d in dims:
        src=old if (old/f'model_d{d}.npz').exists() else sweep
        data=np.load(src/f'model_d{d}.npz');G0=data['G0'];N=d//2
        width=.78**N
        discrete_width=float(np.sqrt(np.linalg.eigvalsh(G0)[::-1][N]))
        for domain in ('sphere','ball'):
            for method,*_ in STYLES:
                ratios=[]
                for seed in range(1000,1008):
                    rom=CoordinateROM(G0,data['H0'],**rom_options(data))
                    if src==sweep:
                        rom.Q=np.load(sweep/f'basis_{domain}_d{d}_{method}_s{seed}.npz')['Q']
                    else:
                        key=('exp_rgd40' if domain=='sphere' else 'normalized_pg40') if method=='gd' else 'fixed_'+method
                        run=next(r for r in saved if r['domain']==domain and r['dimension']==d and r['method']==key and r['seed']==seed)
                        for step in run['rounds']:rom.append(step['parameter'])
                    assert rom.Q.shape[1]==N
                    _,G=rom.matrices()
                    error=float(np.sqrt(np.linalg.eigvalsh(G)[-1]))
                    assert error>=discrete_width*(1-1e-9)
                    ratio=error/width;ratios.append(ratio)
                    records.append(dict(domain=domain,d=d,N=N,method=method,seed=seed,worst_error=error,analytic_width=width,discrete_width=discrete_width,ratio=ratio))
                summary.append(dict(domain=domain,d=d,method=method,median_ratio=float(np.median(ratios))))
    (out/'worst_width_ratios.json').write_text(json.dumps(dict(metric='exact discrete parameter-domain supremum / analytic width',runs=records,summary=summary),indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.3))
    for domain,ax in zip(('sphere','ball'),axes):
        for method,label,col,ls,mk in STYLES:
            vals=[next(r['median_ratio'] for r in summary if (r['domain'],r['d'],r['method'])==(domain,d,method)) for d in dims]
            if method=='gd':label+=', $T=40$'
            ax.plot(dims,vals,label=label,color=col,ls=ls,marker=mk,ms=4.5,lw=2,mfc='white')
        ax.axhline(1,color='#777777',lw=.8,ls=':',zorder=0)
        ax.set(xlabel='Parameter dimension $d$',xticks=[8,16,24,32,40,48,56,64],title=domain.capitalize(),ylim=(.96,2.25));ax.grid(alpha=.16)
    axes[0].set_ylabel('$E_N/d_N$')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=10,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,top=.9,bottom=.25,wspace=.22)
    for ext in ('pdf','png'):fig.savefig(figdir/f'test_width_ratio_vs_d.{ext}',dpi=180)
    print('Computed',len(records),'ROM errors; median ratio range:',min(r['median_ratio'] for r in summary),max(r['median_ratio'] for r in summary))
if __name__=='__main__':main()
