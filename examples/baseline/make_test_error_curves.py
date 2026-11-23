"""Evaluate saved nested reduced spaces on fixed shared test sets (local CPU).

No training or new random test samples are needed. Each prefix of a saved
A-orthonormal basis is the corresponding intermediate greedy space.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/greedy_plot_config')
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_info

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/slides_comparison_20261018'
OUT=ROOT/'runs/test_error_curves_20261020'
FIG=ROOT/'paper/slides/figs/baselines_20261020'
METHODS=['gd','pool256','pool4096','pool10000']
LABELS=['GD, 100 updates','Pool 256','Pool 4096','Pool 10000']
COLORS=['#2A5374','#B1813D','#64717D','#47776A']
STYLES=['-','--','-.',':']
MARKERS=['o','s','^','D']

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    previous=json.loads((SOURCE/'fixed_final_test_errors.json').read_text())
    reference={(r['domain'],r['method'],r['training_seed']):r for r in previous if r['d']==32}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#FAFAF8',
        'axes.facecolor':'#FAFAF8','savefig.facecolor':'#FAFAF8','text.color':'#263039',
        'axes.labelcolor':'#263039','pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.3),sharey=True)
    records=[];summary=[];checks={'max_orthogonality_defect':0.,'max_endpoint_quantile_difference':0.,'max_pointwise_error_increase':0.}
    for domain,ax in zip(['sphere','ball'],axes):
        path=SOURCE/f'fixed_final_spaces_{domain}_d32.npz'
        data=np.load(path);Z=data['test_parameters'];G0=data['G0']
        den=np.einsum('ij,ij->i',Z@G0,Z)
        cross=data['cross'] if 'cross' in data else G0
        assert Z.shape==(16384,32) and np.min(den)>0
        for method,label,color,style,marker in zip(METHODS,LABELS,COLORS,STYLES,MARKERS):
            curves=[]
            for seed in range(1000,1008):
                Q=data[f'Q_{method}_seed{seed}']
                defect=float(np.linalg.norm(Q.T@G0@Q-np.eye(16),2))
                checks['max_orthogonality_defect']=max(checks['max_orthogonality_defect'],defect)
                assert defect<1e-10
                curve=[];last=None
                for N in range(17):
                    q=Q[:,:N];P=np.eye(32)-q@(q.T@cross)
                    residual=Z@P.T
                    numerator=np.einsum('ij,ij->i',residual@G0,residual)
                    assert np.min(numerator)>-1e-12
                    squared=np.maximum(numerator,0)/den
                    assert np.max(squared)<1+1e-9
                    if last is not None:
                        increase=float(np.max(squared-last))
                        checks['max_pointwise_error_increase']=max(checks['max_pointwise_error_increase'],increase)
                        assert increase<1e-9
                    last=squared
                    rms=float(np.sqrt(np.mean(squared)));curve.append(rms)
                    quant=np.quantile(np.sqrt(squared),[.5,.95])
                    records.append({'domain':domain,'d':32,'N':N,'method':method,'training_seed':seed,
                                    'test_count':len(Z),'rms_relative_energy_error':rms,
                                    'median_relative_energy_error':float(quant[0]),'p95_relative_energy_error':float(quant[1])})
                    if N==16:
                        r=reference[(domain,method,seed)]
                        diff=max(abs(quant[0]-r['relative_error_median']),abs(quant[1]-r['relative_error_p95']))
                        checks['max_endpoint_quantile_difference']=max(checks['max_endpoint_quantile_difference'],float(diff))
                        assert diff<1e-10
                curves.append(curve)
            mid=np.median(curves,axis=0)
            summary.append({'domain':domain,'method':method,'N':list(range(17)),'median_over_training_seeds':mid.tolist()})
            ax.semilogy(range(17),mid,color=color,ls=style,lw=2.4 if method=='gd' else 1.9,
                        marker=marker,markevery=[0,4,8,12,16],ms=4.8,label=label)
        ax.set_title('Sphere' if domain=='sphere' else 'Ball')
        ax.set_xlabel('Reduced dimension $N$');ax.set_xlim(0,16);ax.set_xticks([0,4,8,12,16])
        ax.set_ylim(.015,1.15);ax.grid(which='major',alpha=.2);ax.grid(which='minor',axis='y',alpha=.08)
    axes[0].set_ylabel('RMS relative energy error')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.005),ncol=4,frameon=False)
    fig.subplots_adjust(left=.085,right=.985,top=.89,bottom=.24,wspace=.12)
    for suffix in ['pdf','png']:fig.savefig(FIG/f'test_error_vs_N_d32.{suffix}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    meta={'d':32,'N':list(range(17)),'test_count_per_domain':16384,'training_seeds':list(range(1000,1008)),
          'metric':'sqrt(mean_z(||S z-u_N(z)||_A^2 / ||S z||_A^2))',
          'aggregation':'Median over eight training seeds at each N, after computing RMS on the fixed test set for each seed.',
          'test_set':'Unchanged stored test_parameters; shared across all N, seeds and methods within each domain. Separate sphere and ball samples.',
          'device':'local CPU','threadpools':threadpool_info(),'checks':checks,
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/f'fixed_final_spaces_{domain}_d32.npz' for domain in ['sphere','ball']]}}
    for name,obj in [('per_seed_errors',records),('curves',summary),('metadata',meta)]:
        (OUT/f'{name}.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks))
    for r in summary:print(r['domain'],r['method'],'N=16',r['median_over_training_seeds'][-1])

if __name__=='__main__':main()
