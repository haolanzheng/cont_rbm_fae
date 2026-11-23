"""Report the correlation-only control and contrast comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/thermal_block_contrast_correlation_20261103'
r=json.loads((OUT/'prior_results.json').read_text());c=json.loads((OUT/'results.json').read_text());summary=[]
methods=['gd','pool256','pool4096','pool10000','mixed4096'];labels=['GD','Pool 256','Pool 4096','Pool 10000','Mixed pool 4096'];colors=['#24557A','#C17E3C','#438273','#993F4D','#776C42']
for length in [0,.15,.35]:
 for m in methods:
  a=[v for v in r if v['length']==length and v['method']==m];e=[v['curve'][-1]['max_error'] for v in a];er=[v['curve'][-1]['max_relative_error'] for v in a]
  summary.append(dict(length=length,method=m,count=len(a),median_max_error=float(np.median(e)),min_max_error=min(e),max_max_error=max(e),median_max_relative_error=float(np.median(er)),median_training_seconds=float(np.median([v['training_seconds'] for v in a]))))
with (OUT/'prior_summary.csv').open('w') as fp:
 w=csv.DictWriter(fp,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(OUT/'prior_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
for m,label,color in zip(methods,labels,colors):
 es=[]
 for C in [10,30,100,300]:es.append(np.median([v['curve'][-1]['max_error'] for v in c if v['case']==f'contrast_{C}' and v['method']==m]))
 axs[0].plot([10,30,100,300],es,'o-',label=label,color=color,lw=1.8)
 axs[1].plot([0,.15,.35],[v['median_max_error'] for v in summary if v['method']==m],'o-',label=label,color=color,lw=1.8)
axs[0].set(xscale='log',xlabel='Contrast C',title='Independent coefficients in [1,C]')
axs[1].set(xlabel='Sampling correlation length',title='Same coefficient cube [1,100], same marginals')
for ax in axs:ax.set_ylabel('Maximum test error (N=24)');ax.grid(alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=9);fig.tight_layout(rect=(0,.1,1,1));fig.savefig(OUT/'contrast_and_sampling_correlation.png',dpi=180)
print(json.dumps(summary,indent=2))
