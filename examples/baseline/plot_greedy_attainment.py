"""Threshold attainment in the final selection of the reported greedy histories.
No new sampling or optimization: each method uses its own recorded reduced spaces.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/analytic_baseline_matplotlib')
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(out,figdir):
    source=out/'results.json';data=json.loads(source.read_text());dims=data['config']['dimensions']
    gamma=.9;records=[]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
    styles=[('gd','GD, $T=40$','#245E91','-','o'),
            ('fixed_pool256','Pool $M=256$','#BC762C','--','^'),
            ('fixed_pool4096','Pool $M=4096$','#477A68','-.','D'),
            ('fixed_pool10000','Pool $M=10000$','#B44E49',':','s')]
    for dom,ax in zip(['sphere','ball'],axes):
        for key,label,color,ls,mk in styles:
            method=('exp_rgd40' if dom=='sphere' else 'normalized_pg40') if key=='gd' else key
            frequencies=[]
            for d in dims:
                runs=sorted([r for r in data['runs'] if (r['domain'],r['dimension'],r['method'])==(dom,d,method)],key=lambda r:r['seed'])
                assert len(runs)==8 and all(len(r['rounds'])==d//2 for r in runs)
                per_seed=[{'seed':r['seed'],'hits':int(r['rounds'][-1]['indicator_quality']>=gamma),'indicator_quality':r['rounds'][-1]['indicator_quality']} for r in runs]
                hits=sum(r['hits'] for r in per_seed);total=len(per_seed)
                records.append({'domain':dom,'dimension':d,'method':method,'gamma':gamma,'hits':hits,'runs':total,'final_dimension':d//2,'objective_rank':d//2-1,'frequency':hits/total,'per_seed':per_seed})
                frequencies.append(hits/total)
            ax.plot(dims,np.maximum(frequencies,np.finfo(np.float64).eps),color=color,lw=2.3,ls=ls,marker=mk,ms=7,mfc='white',mew=1.5,label=label)
        ax.set_title(dom.capitalize());ax.set_xlabel(r'Parameter dimension $d$');ax.set_xticks(dims)
        ax.set_yscale('log');ax.set_ylim(np.finfo(np.float64).eps/3,2)
        ax.set_yticks([1,1e-4,1e-8,1e-12,1e-16]);ax.grid(alpha=.16,which='major')
    axes[0].set_ylabel('Empirical attainment frequency')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.01),fontsize=11)
    fig.subplots_adjust(left=.10,right=.985,top=.91,bottom=.25,wspace=.20)
    for ext in ['pdf','png']:fig.savefig(figdir/f'empirical_probability_dimension.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (out/'final_greedy_attainment_frequencies.json').write_text(json.dumps({'source':'results.json','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'criterion':'indicator_quality >= 0.9, equivalently J_n(selected) >= 0.81 max_Z J_n','aggregation':'Final selection producing V_N with N=d/2; indicator evaluated on V_(N-1); eight saved runs per method, method-specific reduced spaces','zero_display_floor':float(np.finfo(np.float64).eps),'records':records},indent=2)+'\n')

if __name__=='__main__':
    from .run_analytic_baseline import OUT,FIG
    from . import plot_analytic_baseline  # shared figure style
    render(OUT,FIG)
