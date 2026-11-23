"""Plot the fixed-clipping N study from saved results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cube_n_matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[3]/'runs/cube_fixed_clip_N_20261102'
r=json.loads((p/'results.json').read_text());s=json.loads((p/'summary.json').read_text())
with (p/'summary.csv').open('w') as fp:
 w=csv.DictWriter(fp,fieldnames=list(s[0]));w.writeheader();w.writerows(s)
fig,axs=plt.subplots(2,3,figsize=(11,6.5))
for d,ax in zip((6,8,16,32,48,64),axs.flat):
 a=[v for v in s if v['d']==d];ns=[v['N'] for v in a];errs=[v['error'] for v in a]
 ax.semilogy(ns,errs,'o-',ms=2.5,label='Test maximum error')
 ax.semilogy(ns[:-1],[v['width'] for v in a[:-1]],'--',label='Exact analytic width')
 lo=[];hi=[]
 for n in ns:
  vals=[v['error'] for v in r if v['d']==d and v['N']==n];lo.append(min(vals));hi.append(max(vals))
 ax.fill_between(ns,lo,hi,alpha=.15);ax.set(title=f'd = {d}',xlabel='N');ax.grid(alpha=.2)
fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=2);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(p/'error_trend.png',dpi=180)
