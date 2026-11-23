"""Replot analytical dimension bounds from the saved normalized-step audit."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/greedy_plot_config'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
folder=root/"paper/slides_short/figs/baselines_normalized_20261020"
rows=json.loads((root/'runs/normalized_projection_20261020/probability_curves.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8','savefig.facecolor':'#FAFAF8','axes.edgecolor':'#68727A','text.color':'#263039','axes.labelcolor':'#263039','pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
for domain,ax,method in zip(['sphere','ball'],axes,['exp_rgd','normalized_pg']):
 records=sorted([r for r in rows if r['domain']==domain and r['rank']==r['dimension']//2],key=lambda r:r['dimension'])
 dims=[r['dimension'] for r in records];x=np.arange(len(dims))
 ax.plot(x,[r['methods'][method]['snapshots'][-1]['analytic_lower'] for r in records],color='#245E91',lw=2.8,ls='--',marker='s',mfc='white',ms=6,label=r'GD: lower bound for $q_{100}(\tau)$')
 for m,c,ls,mk in [(256,'#BC762C',':','^'),(4096,'#477A68','-.','D'),(10000,'#B44E49','-','v')]:
  ps=np.array([r['direct_analytic_upper'] for r in records]);upper=np.ones_like(ps);mask=ps<1;upper[mask]=-np.expm1(m*np.log1p(-ps[mask]))
  ax.plot(x,upper,color=c,lw=2,ls=ls,marker=mk,mfc='white',ms=6,label=rf'Pool: upper bound for $P_{{{m}}}(\tau)$')
 ax.set_xticks(x,dims);ax.set_title(domain.capitalize());ax.set_xlabel(r'Parameter dimension $d$');ax.set_yscale("log");ax.set_ylim(top=1.5);ax.grid(alpha=.16,which="major")
axes[0].set_ylabel('Probability bound')
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,0),fontsize=11)
fig.subplots_adjust(left=.085,right=.985,top=.91,bottom=.27,wspace=.2)
for ext in ['pdf','png']:fig.savefig(folder/f'probability_bounds_dimension.{ext}',dpi=180,bbox_inches='tight')
