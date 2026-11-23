"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Standalone scientific figure: medians and observed ranges, NOT confidence bands."""
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    data=json.loads(Path('continuous_greedy_results.json').read_text())
    groups={(x['d'],x['method']):x for x in data['groups']}
    methods=['lbfgs5','lbfgs10','lbfgs20','lbfgs100','local']
    fig,axs=plt.subplots(2,3,figsize=(12.6,7.4),constrained_layout=True)
    for j,(d,n) in enumerate([(11,50),(21,80),(33,100)]):
        for row,(metric,label) in enumerate([('max','Maximum held-out relative X-error'),('uniform_mean','Uniform mean relative X-error')]):
            ax=axs[row,j];xx=np.arange(5)
            med=[groups[d,m][metric+'_median'] for m in methods]
            lo=[groups[d,m][metric+'_min'] for m in methods];hi=[groups[d,m][metric+'_max'] for m in methods]
            ax.fill_between(xx,lo,hi,color='#2769b0',alpha=.13,label='Observed range (3 seeds)')
            ax.plot(xx,med,'o-',color='#2769b0',lw=2,label='Single-start L-BFGS')
            for pool,color,style,labelp in [('pool2000','#bd6231','--','Fixed pool, 2000'),('pool4000','#9369a2',':','Fixed pool, 4000'),('resamp2000','#2f8564','-.','Fresh pool, 2000/round')]:
                ax.axhline(groups[d,pool][metric+'_median'],color=color,ls=style,lw=1.6,label=labelp)
            ax.set_xticks(xx,['5','10','20','100','Local'])
            ax.set_xlabel('Iteration cap / local convergence protocol')
            ax.set_ylabel(label if j==0 else '')
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_title(f'd = {d}, N = {n}')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Thermal fin: raw indicator, 10,017 FE unknowns, identical basis counts\nMedians over seeds 11, 22, 33; held-out maximum is not a certified supremum',fontsize=12)
    fig.savefig('continuous_greedy_summary.png',dpi=170)
    fig.savefig('continuous_greedy_summary.pdf')


if __name__ == "__main__":
    main()
