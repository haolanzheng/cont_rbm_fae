"""Joint error/time summary of squared-residual GD and historical references."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run_squared_residual import OUT,NORMALIZATION
from .step_budget import cases
from .run import dump

def summarize(records,label):
    answer=[]
    for case in cases():
        rr=[r for r in records if r['case']==case['name']]
        for method,T in sorted({(r['method'],r['T'] if r['method']=='gd' else 0) for r in rr}):
            rows=[r for r in rr if r['method']==method and (method!='gd' or r['T']==T)]
            errors=[r['curve'][-1]['max_error'] for r in rows]
            s=dict(case=case['name'],group=case['group'],contrast=case.get('contrast',100 if case['group']=='prior' else None),length=case.get('length',0),objective=label,method=method,T=T if method=='gd' else None,seeds=len(rows),minimum_rank=min(r['rank'] for r in rows),median_error=float(np.median(errors)),min_error=float(np.min(errors)),max_error=float(np.max(errors)),median_relative_error=float(np.median([r['curve'][-1]['max_relative_error'] for r in rows])),median_training_seconds=float(np.median([r['training_seconds'] for r in rows])))
            if label=='squared' and method=='gd':
                final=[r['rounds'][-1] for r in rows]
                s.update(median_last_round_displacement=float(np.median([a['displacement'] for a in final])),median_last_round_initial_gradient=float(np.median([a['initial_gradient_norm'] for a in final])),fraction_rounds_with_decrease=float(np.mean([a['min_increment'] < -1e-12 for r in rows for a in r['rounds']])))
            answer.append(s)
    return answer

def main():
    runs=json.loads((OUT/'results.json').read_text());pools=json.loads((OUT/'pool_references.json').read_text());logs=json.loads((OUT/'log_references.json').read_text())
    summary=summarize(runs,'squared')+summarize(pools,'pool')+summarize(logs,'log')
    dump(OUT/'summary.json',summary)
    fields=list(dict.fromkeys(k for r in summary for k in r))
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(summary)
    table=['Each cell: median maximum test energy error (median complete training seconds). Eight seeds, d=14, N=24.', 'Normalization: '+NORMALIZATION, '']
    for case in cases():
        table.append(case['name'])
        for T in [80,160,320,640]:
            s=next(r for r in summary if r['case']==case['name'] and r['objective']=='squared' and r['T']==T)
            l=next(r for r in summary if r['case']==case['name'] and r['objective']=='log' and r['T']==T)
            table.append(f"  T={T}: squared {s['median_error']:.8g} ({s['median_training_seconds']:.3f}s); historical log+Armijo {l['median_error']:.8g} ({l['median_training_seconds']:.3f}s)")
        for method in ['pool256','pool4096','pool10000','mixed4096']:
            s=next(r for r in summary if r['case']==case['name'] and r['method']==method)
            table.append(f"  {method}: {s['median_error']:.8g} ({s['median_training_seconds']:.3f}s)")
    (OUT/'comparison.txt').write_text('\n'.join(table)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.4))
    for column,(group,title) in enumerate([('contrast','Coefficient contrast'),('prior','Correlated initialization / pools'),('correlation','Affine correlated coefficient family')]):
        for case in [c for c in cases() if c['group']==group]:
            r=sorted([r for r in summary if r['case']==case['name'] and r['objective']=='squared'],key=lambda r:r['T'])
            label=f"C={case['contrast']}" if group=='contrast' else f"length={case['length']:g}"
            color=None
            for row,key in enumerate(['median_error','median_training_seconds']):
                line,=axes[row,column].plot([x['T'] for x in r],[x[key] for x in r],'-o',label=label,color=color)
                color=line.get_color()
                pool=next(x for x in summary if x['case']==case['name'] and x['method']=='pool10000')
                axes[row,column].axhline(pool[key],color=color,ls='--',lw=1,alpha=.7)
        for row in [0,1]:
            ax=axes[row,column];ax.set_xscale('log',base=2);ax.set_xticks([80,160,320,640],labels=['80','160','320','640']);ax.set_yscale('log');ax.grid(alpha=.15);ax.legend(fontsize=8)
        axes[0,column].set_title(title,fontsize=10);axes[1,column].set_xlabel('Iterations per greedy search, T')
    axes[0,0].set_ylabel('Maximum test energy error at N=24');axes[1,0].set_ylabel('Complete training time (seconds)')
    fig.text(.5,.015,f'8-seed medians. Solid: squared-residual GD ({NORMALIZATION} normalization). Dashed: pool of 10,000 points.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'error_time_vs_steps.png',dpi=180);fig.savefig(OUT/'error_time_vs_steps.pdf');plt.close(fig)
    print('\n'.join(table))
if __name__=='__main__':main()
