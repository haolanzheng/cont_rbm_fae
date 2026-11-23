"""Compact numerical report; no changes to existing paper or slides."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[3]/'runs/cube_candidate_screen_20261104'
def main():
    rows=json.loads((OUT/'collective_refine_results.json').read_text())
    curves=json.loads((OUT/'collective_refine_curves.json').read_text())
    methods=['gd','pool256','pool4096','pool10000']
    labels=['GD, T=80','Pool, M=256','Pool, M=4096','Pool, M=10000']
    colors=['#235D83','#A56B33','#648A7C','#925B6A']
    styles=['-','--','-.',':']
    summary=[]
    fig,axs=plt.subplots(1,3,figsize=(12.4,3.7),constrained_layout=True)
    for ax,d in zip(axs,(16,32,64)):
      for m,label,color,style in zip(methods,labels,colors,styles):
        r=[x for x in rows if (x['d'],x['rho'],x['delta'],x['method'])==(d,.9,.01,m)]
        summary.append(dict(d=d,N=d//2,method=m,median_E=float(np.median([x['E_augmented'] for x in r])),min_E=min(x['E_augmented'] for x in r),max_E=max(x['E_augmented'] for x in r),median_seconds=float(np.median([x['seconds'] for x in r]))))
        c=[x for x in curves if (x['d'],x['rho'],x['delta'],x['method'])==(d,.9,.01,m)]
        Ns=sorted(set(x['N'] for x in c));E=np.array([[next(x['E'] for x in c if x['seed']==s and x['N']==N) for N in Ns] for s in range(1000,1008)])
        ax.semilogy(Ns,np.median(E,axis=0),style,color=color,label=label,lw=2)
        ax.fill_between(Ns,np.quantile(E,.25,axis=0),np.quantile(E,.75,axis=0),color=color,alpha=.10,linewidth=0)
      ax.set_title(f'd = {d}');ax.set_xlabel('Reduced dimension N');ax.grid(alpha=.18)
      ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Maximum energy error on common test set')
    axs[1].legend(fontsize=9,frameon=False)
    fig.suptitle('Common-amplitude diffusion: rho = 0.9, delta = 0.01; median and interquartile range of 8 seeds',fontsize=11)
    fig.savefig(OUT/'collective_rb_error.png',dpi=180);plt.close(fig)
    (OUT/'collective_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'collective_summary.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(OUT/'collective_rb_error.png')

if __name__=='__main__':main()
