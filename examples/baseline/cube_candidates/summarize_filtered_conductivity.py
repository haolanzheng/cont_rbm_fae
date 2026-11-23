import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/filtered_conductivity_mpl')
import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[3]/'runs/filtered_conductivity_20261111'
rows=json.loads((P/'results_enriched.json').read_text())
methods=['gd80','gd320','pool256','pool4096','pool10000']
styles={'gd80':('#1766a1','-','GD, T=80'),'gd320':('#148577','--','GD, T=320'),'pool256':('#888888',':','Pool, M=256'),'pool4096':('#b3782a','-.','Pool, M=4096'),'pool10000':('#b44046','-','Pool, M=10000')}
summary=[]
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,d in zip(axes,[32,64,128]):
 print('DIM',d)
 for m in methods:
  rr=[r for r in rows if r['d']==d and r['method']==m];assert len(rr)==4
  a=np.array([r['E_relative'] for r in rr]);e=np.median(a,axis=0);t=np.median([r['timings'] for r in rr],axis=0)
  color,line,label=styles[m];x=np.arange(1,len(e)+1)
  ax.semilogy(x,e,color=color,linestyle=line,label=label,linewidth=1.7)
  if m in ['gd80','pool10000']:ax.fill_between(x,a.min(axis=0),a.max(axis=0),color=color,alpha=.1,linewidth=0)
  thresholds={}
  for target in [.01,.005,.001]:
   inds=np.flatnonzero(e<=target);n=int(inds[0]) if len(inds) else None
   per_run=[]
   for r in rr:
    ids=np.flatnonzero(np.array(r['E_relative'])<=target)
    per_run.append(int(ids[0]+1) if len(ids) else None)
   thresholds[str(target)]=dict(N=None if n is None else n+1,seconds=None if n is None else float(t[n]),N_per_seed=per_run)
  summary.append(dict(d=d,method=m,N16=float(e[15]),N24=float(e[23]),N32=float(e[31]),seconds32=float(t[31]),thresholds=thresholds))
  print(m,'N16,N24,N32',*[round(float(e[n-1]),7) for n in [16,24,32]],'time32',round(float(t[-1]),4),'1pct',thresholds['0.01'])
 ax.set(xlabel='N (lift + correction vectors)',title=f'd = {d}',ylim=(5e-5,1))
 ax.axhline(.01,color='#555555',linewidth=.7,linestyle=':')
 ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Maximum relative test error')
axes[1].legend(fontsize=8,frameon=False,loc='lower left')
fig.savefig(P/'error_vs_N.png',dpi=190)
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
