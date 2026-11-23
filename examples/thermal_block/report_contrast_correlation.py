"""Report all predeclared contrast and matched-variance correlation cases."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/thermal_block_contrast_correlation_20261103'
r=json.loads((OUT/'results.json').read_text());cfg=json.loads((OUT/'config.json').read_text());summary=[]
methods=cfg['methods'];labels=['GD','Pool 256','Pool 4096','Pool 10000','Mixed pool 4096'];colors=['#24557A','#C17E3C','#438273','#993F4D','#776C42']
for c in cfg['cases']:
 for m in methods:
  a=[v for v in r if v['case']==c['name'] and v['method']==m]
  if not a:continue
  for n in cfg['checkpoints']:
   vals=[next(x for x in v['curve'] if x['N']==n) for v in a if any(x['N']==n for x in v['curve'])]
   if not vals:continue
   e=[x['max_error'] for x in vals];er=[x['max_relative_error'] for x in vals]
   summary.append(dict(case=c['name'],method=m,N=n,count=len(vals),max_error_median=float(np.median(e)),max_error_min=min(e),max_error_max=max(e),max_relative_error_median=float(np.median(er)),training_seconds_median=float(np.median([v['training_seconds'] for v in a])),interior_max_median=float(np.median([x['interior_max'] for x in vals])),boundary_max_median=float(np.median([x['boundary_max'] for x in vals]))))
with (OUT/'summary.csv').open('w') as fp:
 w=csv.DictWriter(fp,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(11,7))
for col,group in enumerate(['contrast','correlation']):
 cases=[c for c in cfg['cases'] if c['group']==group];xs=[c['contrast'] if group=='contrast' else c['length'] for c in cases]
 for m,label,color in zip(methods,labels,colors):
  vals=[];ratios=[]
  for c in cases:
   a=next(v for v in summary if v['case']==c['name'] and v['method']==m and v['N']==24)
   g=next(v for v in summary if v['case']==c['name'] and v['method']=='gd' and v['N']==24)
   vals.append(a['max_error_median']);ratios.append(a['max_error_median']/g['max_error_median'])
  axs[0,col].plot(xs,vals,'o-',label=label,color=color)
  if m!='gd':axs[1,col].plot(xs,ratios,'o-',label=label,color=color)
 axs[0,col].set_yscale('log');axs[0,col].set_title('Independent block coefficients' if group=='contrast' else 'Correlated coefficients: fixed mean and variance')
 axs[1,col].axhline(1,color='gray',ls='--',lw=1)
 for row in [0,1]:
  axs[row,col].set_xlabel('Contrast C' if group=='contrast' else 'Correlation length');axs[row,col].grid(alpha=.2)
  if group=='contrast':axs[row,col].set_xscale('log')
axs[0,0].set_ylabel('Maximum test error (N=24)');axs[1,0].set_ylabel('Pool error / GD error')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=9);fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'contrast_correlation.png',dpi=180)
for c in cfg['cases']:
 print(c['name'])
 for x in summary:
  if x['case']==c['name'] and x['N']==24:print(x['method'],x['count'],x['max_error_median'],x['max_relative_error_median'],x['training_seconds_median'])
