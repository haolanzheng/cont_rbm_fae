"""Aggregate common-test thermal-block comparisons; never edits slides."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/thermal_block_cube_20261103'
records=json.loads((OUT/'results.json').read_text());summary=[]
methods=['gd','pool256','pool4096','pool10000','mixed4096','grid1','grid0.5']
labels=['GD','Pool 256','Pool 4096','Pool 10000','Mixed pool 4096','Grid h=1','Grid h=1/2']
colors=['#24557A','#C17E3C','#438273','#993F4D','#776C42','#666666','#999999']
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
for d,ax in zip((4,14),axs):
 for method,label,color in zip(methods,labels,colors):
  a=[r for r in records if r['d']==d and r['method']==method]
  if not a:continue
  for n in range(1,min(r['rank'] for r in a)+1):
   errs=[r['curve'][n-1]['max_error'] for r in a];relative=[r['curve'][n-1]['max_relative_error'] for r in a]
   summary.append(dict(d=d,method=method,N=n,count=len(a),median_max_error=float(np.median(errs)),min_max_error=min(errs),max_max_error=max(errs),median_max_relative_error=float(np.median(relative)),median_training_seconds=float(np.median([r['training_seconds'] for r in a]))))
  b=[r for r in summary if r['d']==d and r['method']==method];ax.semilogy([r['N'] for r in b],[r['median_max_error'] for r in b],label=label,color=color,lw=1.8,ls='-' if method=='gd' else '--')
 ax.set(xlabel='Reduced dimension N',ylabel='Maximum test error in reference energy norm',title=f'd = {d}');ax.grid(alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9);fig.tight_layout(rect=(0,.13,1,1));fig.savefig(OUT/'maximum_error_vs_N.png',dpi=180)
with (OUT/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for d in [4,14]:
 for m in methods:
  a=[r for r in summary if r['d']==d and r['method']==m and r['N']==24]
  if a:print(a[0])
checks={'run_count':len(records),'direct_error_audit_max_abs_difference':max(abs(c['direct']-c['formula']) for r in records for c in r['direct_error_check'])}
(OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
