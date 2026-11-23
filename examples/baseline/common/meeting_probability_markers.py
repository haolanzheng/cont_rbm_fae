"""Add sparse markers to meeting probability curves, reusing saved ordinates."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/analytic_baseline_matplotlib')
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.baseline import plot_analytic_baseline  # retain the existing figure style
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'runs/analytic_hf_a06_t40_20261022'
FIG=ROOT/'paper/slides_meeting_20261113/figs'
for name,record_name,key,xlabel,ylabel in [
 ('step_size_probability_bounds','step_size_bounds.json','steps',r'Step scale $\alpha$',r'Analytical lower bound for $q_{40}$'),
 ('iteration_probability_bounds','iteration_bounds.json','T',r'Number of updates $T$',r'Analytical lower bound for $q_T$')]:
 records=json.loads((SOURCE/record_name).read_text())
 dims=sorted({r['d'] for r in records})
 fig,axes=plt.subplots(1,2,figsize=(11.6,3.9))
 for d,color,style,marker in zip(dims,['#27647B','#B1813D','#527B61','#A55359','#64717D'],['-','--','-.',':',(0,(5,1,1,1))],['o','^','D','s','v']):
  for dom,ax in zip(['sphere','ball'],axes):
   r=next(r for r in records if r['domain']==dom and r['d']==d)
   ax.plot(r[key],r['lower_bound'],color=color,ls=style,lw=2.4,label=f'd = {d}',marker=marker,ms=5,mfc='white',mew=1.1,markevery=max(1,len(r[key])//8))
 for ax,dom in zip(axes,['Sphere','Ball']):
  ax.set_title(dom);ax.set_ylim(0,1.025);ax.set_xlim(left=0);ax.grid(alpha=.17);ax.set_xlabel(xlabel)
  if key=='T':
   ax.set_xlim(0,100);ax.axvline(40,color='#68727A',ls=':',lw=1,alpha=.65)
 axes[0].set_ylabel(ylabel)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.52,-.05))
 fig.subplots_adjust(wspace=.2,bottom=.18)
 for ext in ('pdf','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
