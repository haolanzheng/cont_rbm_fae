"""Compare objective scales, keeping squared loss and projected updates fixed."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import dump
from .step_budget import cases
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/thermal_block_squared_unscaled_20261103'
FOLDERS={'none':'thermal_block_squared_unscaled_20261103','source':'thermal_block_squared_residual_20261103','gram':'thermal_block_squared_gram_20261103'}

def main():
    rows=[];raw={}
    raw_check=json.loads((OUT/'validation_contrast_10.json').read_text())
    source_check=json.loads((ROOT/'runs'/FOLDERS['source']/'validation_contrast_10.json').read_text())
    source_normalizer=raw_check['objective']/source_check['objective']
    for mode,name in FOLDERS.items():
        data=json.loads((ROOT/'runs'/name/'results.json').read_text());raw[mode]=data
        for case in cases():
            for T in [80,160,320,640]:
                a=[r for r in data if r['case']==case['name'] and r['T']==T]
                assert len(a)==8
                rows.append(dict(case=case['name'],group=case['group'],contrast=case.get('contrast',100 if case['group']=='prior' else None),length=case.get('length',0),normalization=mode,T=T,seeds=len(a),median_error=float(np.median([r['curve'][-1]['max_error'] for r in a])),median_relative_error=float(np.median([r['curve'][-1]['max_relative_error'] for r in a])),median_seconds=float(np.median([r['training_seconds'] for r in a])),median_last_initial_gradient=float(np.median([r['rounds'][-1]['initial_gradient_norm'] for r in a])),median_last_displacement=float(np.median([r['rounds'][-1]['displacement'] for r in a])),median_last_normalizer=float(np.median([r['rounds'][-1].get('normalizer',source_normalizer if mode=='source' else 1.) for r in a]))))
    refs=json.loads((OUT/'pool_references.json').read_text());pools={}
    for c in cases():
        for method in ['pool256','pool4096','pool10000','mixed4096']:
            a=[r for r in refs if r['case']==c['name'] and r['method']==method]
            pools[c['name'],method]=dict(error=float(np.median([r['curve'][-1]['max_error'] for r in a])),seconds=float(np.median([r['training_seconds'] for r in a])))
    dump(OUT/'normalization_comparison.json',rows)
    with (OUT/'normalization_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    table=['All values are eight-seed medians, d=14, N=24. Each entry is error (training seconds).','']
    for c in cases():
        table.append(c['name'])
        for T in [80,160,320,640]:
            entries=[]
            for mode in FOLDERS:
                r=next(x for x in rows if x['case']==c['name'] and x['T']==T and x['normalization']==mode)
                entries.append(f"{mode}: {r['median_error']:.7g} ({r['median_seconds']:.3f})")
            table.append(f"T={T}: "+'; '.join(entries))
        p=pools[c['name'],'pool10000'];table.append(f"pool10000: {p['error']:.7g} ({p['seconds']:.3f})")
    (OUT/'normalization_comparison.txt').write_text('\n'.join(table)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(12,6.8))
    selected=['contrast_10','contrast_30','contrast_100','contrast_300','prior_0.15','correlation_0.35']
    names=['Independent coefficients: C=10','Independent coefficients: C=30','Independent coefficients: C=100','Independent coefficients: C=300','C=100, sampling length=0.15','Affine family, length=0.35']
    labels={'none':'Unscaled squared residual','source':'Fixed source normalization','gram':'Gram-matrix normalization'}
    for ax,c,title in zip(axes.ravel(),selected,names):
        for mode,marker in zip(FOLDERS,['o','s','^']):
            rr=sorted([r for r in rows if r['case']==c and r['normalization']==mode],key=lambda x:x['T'])
            ax.plot([r['T'] for r in rr],[r['median_error'] for r in rr],marker=marker,label=labels[mode],lw=1.4)
        ax.axhline(pools[c,'pool10000']['error'],color='black',ls='--',lw=1.2,label='Pool 10000')
        ax.set_xscale('log',base=2);ax.set_xticks([80,160,320,640],labels=['80','160','320','640']);ax.set_yscale('log');ax.set_title(title,fontsize=10);ax.grid(alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Maximum test energy error')
    for ax in axes[1,:]:ax.set_xlabel('Iterations per search, T')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(OUT/'normalization_comparison.png',dpi=180);fig.savefig(OUT/'normalization_comparison.pdf');plt.close(fig)
    initial={};difference=0.
    for mode,data in raw.items():
        for r in data:
            for a in r['rounds']:
                k=(r['case'],r['seed'],a['n']);z=np.array(a['initial'])
                if k in initial:difference=max(difference,float(np.max(abs(z-initial[k]))))
                else:initial[k]=z
    validations=[json.loads(p.read_text()) for p in OUT.glob('validation_*.json')]
    records=raw['none']
    checks=dict(run_count=len(records),all_rank_24=all(r['rank']==24 for r in records),all_unit_normalizers=all(a['normalizer']==1 for r in records for a in r['rounds']),all_exact_iteration_budgets=all(a['iterations']==r['T'] for r in records for a in r['rounds']),maximum_initial_difference_across_T_and_normalizations=difference,gradient_relative_error_max=max(v['gradient_relative_error'] for v in validations),validation_cases=len(validations))
    assert difference==0 and checks['all_unit_normalizers'] and checks['run_count']==320
    dump(OUT/'checks.json',checks)
    print('\n'.join(table));print(checks)
if __name__=='__main__':main()
