"""Summarize all declared budgets; do not select a budget using test errors."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .step_budget import OUT,cases
from .run import dump

def main():
    runs=json.loads((OUT/'results.json').read_text());refs=json.loads((OUT/'references.json').read_text())
    allruns=refs+runs;summary=[]
    for case in cases():
        rr=[r for r in allruns if r['case']==case['name']]
        baseline={r['seed']:r['curve'][-1]['max_error'] for r in rr if r['method']=='gd' and r['T']==80}
        pools={m:{r['seed']:r['curve'][-1]['max_error'] for r in rr if r['method']==m} for m in ['pool4096','pool10000','mixed4096']}
        keys=sorted({(r['method'],r['T']) for r in rr})
        for method,T in keys:
            rows=[r for r in rr if r['method']==method and r['T']==T];err=np.array([r['curve'][-1]['max_error'] for r in rows]);relative=[r['curve'][-1]['max_relative_error'] for r in rows]
            s=dict(case=case['name'],method=method,T=T if method=='gd' else None,seeds=len(rows),median_error=float(np.median(err)),min_error=float(err.min()),max_error=float(err.max()),median_relative_error=float(np.median(relative)),median_training_seconds=float(np.median([r['training_seconds'] for r in rows])))
            if method=='gd':
                s['wins_over_T80']=sum(r['curve'][-1]['max_error']<baseline[r['seed']] for r in rows)
                s['median_paired_error_ratio_to_T80']=float(np.median([r['curve'][-1]['max_error']/baseline[r['seed']] for r in rows]))
                iterations=[a['iterations'] for r in rows for a in r['rounds'] if 'iterations' in a]
                s['median_iterations']=float(np.median(iterations));s['fraction_rounds_hitting_budget']=float(np.mean(np.array(iterations)==T))
                for m,p in pools.items():s['wins_over_'+m]=sum(r['curve'][-1]['max_error']<p[r['seed']] for r in rows)
            summary.append(s)
    dump(OUT/'summary.json',summary)
    fields=list(dict.fromkeys(k for r in summary for k in r))
    with (OUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(summary)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1))
    groups=[('contrast','Coefficient contrast'),('prior','Correlation of initialization / pools'),('correlation','Affine correlated coefficient family')]
    for ax,(group,title) in zip(axes,groups):
        for c in [c for c in cases() if c['group']==group]:
            rows=sorted([r for r in summary if r['case']==c['name'] and r['method']=='gd'],key=lambda r:r['T'])
            label=f"C={c['contrast']}" if group=='contrast' else f"length={c['length']:g}"
            line,=ax.plot([r['T'] for r in rows],[r['median_error'] for r in rows],'-o',label=label)
            pool=next(r for r in summary if r['case']==c['name'] and r['method']=='pool10000')
            ax.axhline(pool['median_error'],color=line.get_color(),ls='--',lw=1,alpha=.7)
        ax.set_xscale('log',base=2);ax.set_xticks([80,160,320,640],labels=['80','160','320','640']);ax.set_yscale('log');ax.set_xlabel('Maximum iterations per greedy search, T');ax.set_title(title,fontsize=10);ax.legend(fontsize=8);ax.grid(alpha=.15)
    axes[0].set_ylabel('Maximum test energy error at N=24')
    fig.text(.5,.015,'8-seed medians. Solid: GD. Dashed: pool of 10,000 points (matching color).',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'error_vs_steps.png',dpi=180);fig.savefig(OUT/'error_vs_steps.pdf');plt.close(fig)
    print(json.dumps([r for r in summary if r['method']=='gd'],indent=2))
if __name__=='__main__':main()
