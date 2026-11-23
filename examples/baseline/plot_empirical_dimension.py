"""Plot saved empirical GD and iid-pool frequencies versus dimension."""
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
 records=sorted([r for r in rows if r['domain']==domain and r['rank']==r['dimension']//2],key=lambda r:r['dimension']);dims=[r['dimension'] for r in records];x=np.arange(len(dims))
 gd=[r['methods'][method]['snapshots'][-1] for r in records];pool=[r['iid_pool'][-1] for r in records]
 for vals,c,style,mk,label in [(gd,'#245E91','-','o',r'GD: $\widehat q_{100}(\tau)$'),(pool,'#BC762C','--','^',r'Pool: $\widehat P_{101}(\tau)$')]:
  ax.plot(x,[v['q_hat'] for v in vals],color=c,lw=2.5,ls=style,marker=mk,ms=6,label=label)
  ax.fill_between(x,[v['ci95'][0] for v in vals],[v['ci95'][1] for v in vals],color=c,alpha=.16)
 ax.set_xticks(x,dims);ax.set_title(domain.capitalize());ax.set_xlabel(r'Parameter dimension $d$');ax.set_ylim(-.035,1.035);ax.grid(alpha=.16)
axes[0].set_ylabel(r'Empirical probability of reaching $J\geq\tau$')
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.01));fig.subplots_adjust(left=.085,right=.985,top=.91,bottom=.23,wspace=.2)
for ext in ['pdf','png']:fig.savefig(folder/f'empirical_probability_dimension.{ext}',dpi=180,bbox_inches='tight')
