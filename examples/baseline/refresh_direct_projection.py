"""Refresh sphere/ball reports with ordinary ball projection, on one local CPU.

Run: PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.refresh_direct_projection
Archived data are read only. New histories, test evaluations and timing samples
are saved separately; unchanged sphere/pool histories are reused with provenance.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/continuous_greedy_mpl')
from pathlib import Path
import json, copy, hashlib, platform, sys
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits,threadpool_info
from .geometry_cpu import CoordinateROM,search
from .run_geometry_cpu import run_greedy,pool_probability
from .run_extended_comparison_cpu import selector_run,batch_probability
from . import make_slides_comparison as compare
from . import make_test_error_curves as curves
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/geometry_cpu_20261012'
EXT=ROOT/'runs/geometry_extended_cpu_20261015'
OUT=ROOT/'runs/direct_projection_20261020'
FIG=OUT/'figures'

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def final_errors(high):
    summary=[];records=[];checks={'max_orthogonality_defect':0.,'max_error_reconstruction_difference':0.}
    for d in [8,16,32,64]:
        data=np.load(OLD/f'fem_d{d}.npz');G0,H0=data['G0'],data['H0'];N=d//2
        for domain in ['sphere','ball']:
            path=ROOT/f'runs/final_test_errors_20261020/test_{domain}_d{d}.npz'
            Z=np.load(path)['test_parameters'];den=np.einsum('ij,ij->i',Z@G0,Z)
            methods=['exp_rgd100' if domain=='sphere' else 'hard_pg100','fixed_pool256','fixed_pool4096','fixed_pool10000']
            for method in methods:
                values=[]
                for r in [r for r in high if r['dimension']==d and r['domain']==domain and r['method']==method]:
                    rom=CoordinateROM(G0,H0)
                    for step in r['rounds'][:N]:rom.append(step['parameter'])
                    P=np.eye(d)-rom.Q@(rom.Q.T@G0);e=Z@P.T
                    ratio=np.maximum(np.einsum('ij,ij->i',e@G0,e),0)/den
                    value=float(np.sqrt(ratio.mean()));values.append(value)
                    _,G=rom.matrices();diff=abs(np.sqrt(max(la.eigvalsh(G)[-1],0))-r['errors'][N])/r['errors'][N]
                    assert diff<1e-7
                    checks['max_error_reconstruction_difference']=max(checks['max_error_reconstruction_difference'],float(diff))
                    checks['max_orthogonality_defect']=max(checks['max_orthogonality_defect'],rom.orthogonality_defect())
                    records.append({'domain':domain,'d':d,'N':N,'method':method,'seed':r['seed'],'rms':value})
                assert len(values)==8
                summary.append({'domain':domain,'d':d,'N':N,'method':method,'median_rms_relative_energy_error':float(np.median(values))})
    dump('final_test_errors.json',summary);dump('final_test_errors_per_seed.json',records)
    return checks

def probability_plot(rows):
    import matplotlib.pyplot as plt
    r=next(r for r in rows if r['domain']=='ball' and r['dimension']==64 and r['rank']==32)
    fig,axes=plt.subplots(1,2,figsize=(11.6,6.8))
    colors=['#263039','#245E91','#BC762C','#B44E49','#477A68']
    x=np.arange(101);gd=r['methods']['hard_pg']['snapshots'];pool=r['iid_pool']
    ax=axes[0]
    ax.plot(x,[v['q_hat'] for v in gd],color=colors[0],lw=2.3,marker='o',markevery=12,ms=4,label=r'GD: empirical $\widehat q_T(\tau)$')
    ax.fill_between(x,[v['ci95'][0] for v in gd],[v['ci95'][1] for v in gd],color=colors[0],alpha=.13)
    ax.plot(x,[v['analytic_lower'] for v in gd],color=colors[1],lw=3,ls='--',marker='s',mfc='white',markevery=12,ms=5,label=r'GD: lower bound for $q_T(\tau)$')
    ax.plot(x,[v['q_hat'] for v in pool],color=colors[2],lw=1.5,marker='^',markevery=12,ms=4,label=r'Pool: empirical $\widehat P_{T+1}(\tau)$')
    ax.fill_between(x,[v['ci95'][0] for v in pool],[v['ci95'][1] for v in pool],color=colors[2],alpha=.13)
    ax.plot(x,[v['analytic_upper'] for v in pool],color=colors[3],lw=2.2,ls=':',marker='D',mfc='white',markevery=12,ms=4,label=r'Pool: upper bound for $P_{T+1}(\tau)$')
    ax.set_title('d=64, n=32');ax.set_xlabel('Gradient updates T');ax.set_ylabel(r'Probability of reaching $J\geq\tau$')
    rr=sorted((r for r in rows if r['domain']=='ball' and r['rank']==r['dimension']//2),key=lambda r:r['dimension'])
    dims=[r['dimension'] for r in rr];x=np.arange(len(dims));ax=axes[1]
    gd=[r['methods']['hard_pg']['snapshots'][-1] for r in rr]
    ax.plot(x,[v['q_hat'] for v in gd],color=colors[0],lw=2.3,marker='o',ms=4,label=r'GD: empirical $\widehat q_{100}(\tau)$')
    ax.fill_between(x,[v['ci95'][0] for v in gd],[v['ci95'][1] for v in gd],color=colors[0],alpha=.13)
    ax.plot(x,[v['analytic_lower'] for v in gd],color=colors[1],lw=3,ls='--',marker='s',mfc='white',ms=5,label=r'GD: lower bound for $q_{100}(\tau)$')
    for m,c,ls,marker in [(256,colors[2],':','^'),(4096,colors[4],'-.','D'),(10000,colors[3],'-.','v')]:
        ax.plot(x,[pool_probability(r['direct_analytic_upper'],m) for r in rr],color=c,lw=2.2,ls=ls,marker=marker,mfc='white',ms=5,label=rf'Pool: upper bound for $P_{{{m}}}(\tau)$')
    ax.set_xticks(x,dims);ax.set_title('T=100, n=d/2');ax.set_xlabel('Parameter dimension d')
    for ax in axes:
        ax.set_ylim(-.04,1.05);ax.grid(alpha=.16);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),frameon=False,fontsize=11)
    fig.subplots_adjust(left=.09,right=.985,top=.92,bottom=.37,wspace=.25)
    for ext in ['pdf','png']:fig.savefig(FIG/f'probability_ball.{ext}',dpi=190)
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    high=json.loads((OLD/'results.json').read_text());low=json.loads((EXT/'low_dimension_results.json').read_text())
    checks={};high['config']['ball_omega']=1.
    # Recompute every ball GD history, retaining the original starts/seeds.
    oldhard={(r['dimension'],r['seed']):r for r in high['runs'] if r['domain']=='ball' and r['method']=='hard_pg100'}
    high['runs']=[r for r in high['runs'] if not(r['domain']=='ball' and r['method'] in ['relaxed_pg100','hard_pg100'])]
    maxdiff=0.
    for d in [8,16,32,64]:
        data=np.load(OLD/f'fem_d{d}.npz')
        for seed in range(1000,1008):
            r=run_greedy(data['G0'],data['H0'],'ball','hard_pg100',seed,high['config'])
            maxdiff=max(maxdiff,float(np.max(np.abs(np.array(r['errors'])-oldhard[d,seed]['errors']))))
            high['runs'].append(r)
        print('Ball greedy refreshed:',d,flush=True)
    checks['max_difference_from_archived_direct_projection_errors']=maxdiff
    for i,r in enumerate(low['runs']):
        if r['domain']=='ball' and r['method']=='gd':
            data=np.load(EXT/f'low_d{r["dimension"]}.npz');prefix='rotated_' if r['orientation']=='rotated' else ''
            new=selector_run(data[prefix+'G0'],data[prefix+'H0'],'ball','gd',r['seed'],100,len(r['rounds']),omega=1.)
            new['orientation']=r['orientation'];low['runs'][i]=new
    dump('results.json',high);dump('low_dimension_results.json',low)
    compare.OUT=OUT;compare.FIG=FIG;compare.ALIASES.pop('relaxed_pg100');compare.ALIASES['hard_pg100']='gd'
    compare.step_bounds(omega=1.);print('Step bounds refreshed',flush=True)
    checks['quantile_checks']=compare.test_final_spaces(low['runs'],high['runs']);print('Common-test quantiles refreshed',flush=True)
    curves.SOURCE=OUT;curves.OUT=OUT/'test_error_curves';curves.FIG=FIG;curves.main()
    checks['rms_checks']=final_errors(high['runs']);print('All final test RMS values refreshed',flush=True)
    # Timing all compared methods in the same session, with randomized order.
    compare.timings(low['runs'],omega=1.)
    original=json.loads((EXT/'probability_curves.json').read_text());probability=[]
    for r in original:
        if r['domain']=='sphere':probability.append(r);continue
        H=np.load(EXT/f'frozen_ball_d{r["dimension"]}_n{r["rank"]}.npz')['H']
        new=batch_probability(H,'ball',r['starts'],100,r['seed'],r['gamma'],OUT,r['rank'],ball_omegas=(1.,))
        assert [x['hits'] for x in new['methods']['hard_pg']['snapshots']]==[x['hits'] for x in r['methods']['hard_pg']['snapshots']]
        probability.append(new);print('Ball probability refreshed:',r['dimension'],r['rank'],flush=True)
    dump('probability_curves.json',probability);probability_plot(probability)
    dump('metadata.json',{'ball_update':'P_B(z + H z / ||H||_F); eta=1/2, no relaxation','sphere_update':'unchanged, alpha=0.2',
        'device':'local CPU','platform':platform.platform(),'python':sys.executable,'threadpools':threadpool_info(),
        'training_seeds':list(range(1000,1008)),'updates':100,'test_count':16384,'probability_repetitions':4096,
        'timings':'All table methods remeasured in one session; 3 repeats per random seed, 5 per grid; candidate generation, matrices, selection and enrichment included.',
        'unchanged_data':'Sphere, exact-indicator, grid and pool histories reused from archived runs; ball GD retrained. Test sets are the same saved independent sets.',
        'sources':[str(OLD.relative_to(ROOT)),str(EXT.relative_to(ROOT))],'checks':checks,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('DONE',OUT,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
