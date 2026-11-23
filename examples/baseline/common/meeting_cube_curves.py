"""GD, fixed random pools and uniform meshes on common test sets."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cube_n_matplotlib')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FormatStrFormatter, NullFormatter
root=Path(__file__).resolve().parents[3];out=root/'runs/cube_fixed_clip_N_20261102'
rows=json.loads((out/'comparison_results.json').read_text())
rows += [dict(r,method='gd') for r in json.loads((out/'results.json').read_text())]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10,5.4),sharey=False)
styles=[('gd','GD','#24557A','-'),('pool256','Pool 256','#C17E3C','--'),('pool4096','Pool 4096','#438273','-.'),('pool10000','Pool 10000','#993F4D',':'),('grid1','Grid $h=1$','#555555','--'),('grid0.5','Grid $h=1/2$','#555555','-.'),('grid0.25','Grid $h=1/4$','#555555',':')]
MARKERS={'gd':'o','pool256':'^','pool4096':'D','pool10000':'s','grid1':'v','grid0.5':'P','grid0.25':'X'}
handles=[];labels=[]
for d,ax in zip((6,16,32,64),axs.flat):
 values=[]
 for method,label,color,style in styles:
  if method.startswith('grid') and d!=6:continue
  last=min(max(r['N'] for r in rows if r['d']==d and r['method']==method and r['seed']==seed) for seed in ([1000] if method.startswith('grid') else range(1000,1008)))
  last=min(last,d-1)
  ns=np.arange(last+1);v=[np.median([r['error'] for r in rows if r['d']==d and r['N']==n and r['method']==method]) for n in ns]
  values.extend(v)
  line,=ax.semilogy(ns,v,color=color,ls=style,lw=1.7,label=label,marker=MARKERS[method],markevery=max(1,len(ns)//7),ms=4,mfc='white',mew=1)
  if d==6:handles.append(line);labels.append(label)
 ax.set_title(f'$d={d}$',pad=3);ax.set_xlabel('$N$',labelpad=1);ax.grid(alpha=.18)
 positive=np.asarray(values);positive=positive[positive>0]
 lo,hi=float(positive.min()),float(positive.max())
 padding=max(.035,.08*np.log10(hi/lo))
 ax.set_ylim(lo/10**padding,hi*10**padding)
 if hi/lo<10:
  ticks=MaxNLocator(nbins=5).tick_values(lo,hi)
  ax.yaxis.set_major_locator(FixedLocator(ticks[(ticks>=lo/10**padding)&(ticks<=hi*10**padding)]))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
  ax.yaxis.set_minor_formatter(NullFormatter())
for ax in axs[:,0]:ax.set_ylabel('Test maximum error')
fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9,frameon=False)
fig.tight_layout(pad=.7,h_pad=.65,w_pad=1.5,rect=(0,.12,1,1))
fig.savefig(root/'paper/slides_meeting_20261113/figs/cube_test_error_vs_N_dimensions.pdf')
fig.savefig(root/'runs/meeting_20261113/cube_curves.png',dpi=180)
