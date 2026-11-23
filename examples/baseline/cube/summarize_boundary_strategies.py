"""Summarize saved boundary comparisons without modifying presentation files."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .boundary_strategies import METHODS
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'runs/cube_boundary_strategies_20261102'
g=json.loads((p/'greedy.json').read_text());f=json.loads((p/'frozen.json').read_text())
dims=sorted({r['d'] for r in g});rows=[]
for m in METHODS:
 for d in dims:
  a=[r for r in g if r['method']==m and r['d']==d];b=[r for r in f if r['method']==m and r['d']==d]
  rows.append(dict(method=m,d=d,median_error_ratio=float(np.median([r['ratio'] for r in a])),median_wall_ms=float(np.median([r['wall_ms'] for r in a])),median_h_actions=float(np.median([r['h_actions'] for r in a])),median_coordinate_scores=float(np.median([r['coordinate_endpoint_scores'] for r in a])),frozen_fraction_lower=float(np.mean([r['quality_interval'][0]>=.9 for r in b])),frozen_fraction_upper=float(np.mean([r['quality_interval'][1]>=.9 for r in b]))))
with (p/'summary.csv').open('w') as fp:
 w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
for m in METHODS:
 a=[r for r in rows if r['method']==m]
 axs[0].plot(dims,[r['median_error_ratio'] for r in a],marker='o',ms=3,label=m)
 axs[1].plot(dims,[r['frozen_fraction_lower'] for r in a],marker='o',ms=3,label=m)
axs[0].set(xlabel='Dimension d',ylabel='Median test maximum / exact width',title='Complete greedy: N = d/2; eight seeds')
axs[1].set(xlabel='Dimension d',ylabel='Fraction meeting the threshold',title='Fixed reduced space: 64 common initial points',ylim=(-.03,1.03))
for ax in axs:ax.grid(alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8)
fig.tight_layout(rect=(0,.16,1,1));fig.savefig(p/'comparison.png',dpi=180);plt.close(fig)
checks={'greedy_records':len(g),'frozen_records':len(f),'max_orthogonality_defect':max(r['orthogonality_defect'] for r in g),'max_feasibility_defect':max(r['stats']['max_feasibility_defect'] for r in f),'minimum_frozen_step_increment':min(r['stats']['min_increment'] for r in f)}
(p/'summary_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(checks)
print('d64 cost',[(r['method'],round(r['median_wall_ms'],2),r['median_h_actions']) for r in rows if r['d']==64])
