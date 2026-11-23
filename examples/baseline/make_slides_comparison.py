"""Reproduce slide additions: step bounds, frozen-ROM tests, and N=d/2 wall times.
Run with the local CPU environment: PYTHONPATH=src:. .venv_cpu/bin/python -m examples.baseline.make_slides_comparison
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/greedy_plot_config')
from pathlib import Path
import json,time,platform,hashlib,sys
import numpy as np
import scipy.linalg as la
from scipy.special import betaincc
from threadpoolctl import threadpool_limits,threadpool_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .geometry_cpu import CoordinateROM,uniform_parameters,search,pool_search,rom_options
from .run_extended_comparison_cpu import tensor_candidates

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/geometry_cpu_20261012'
EXT=ROOT/'runs/geometry_extended_cpu_20261015'
OUT=ROOT/'runs/slides_comparison_20261018'
FIG=ROOT/'paper/slides/figs/baselines_20261018'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,
 'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8','savefig.facecolor':'#FAFAF8',
 'axes.edgecolor':'#68727A','text.color':'#263039','axes.labelcolor':'#263039',
 'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight',dpi=180)
    plt.close(fig)
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def step_bounds(omega=0.5, ball_step_rule="fixed"):
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.9))
    colors=['#27647B','#B1813D','#527B61','#A55359']; styles=['-','--','-.',':']
    records=[]
    for d,color,style in zip([8,32,64,128],colors,styles):
        for domain,ax in zip(['sphere','ball'],axes):
            H=np.load(EXT/f'frozen_{domain}_d{d}_n{d//2}.npz')['H']
            sigma=la.eigvalsh(H)[-1]/la.norm(H,'fro')
            step=np.linspace(0,.5 if domain=='sphere' or ball_step_rule=='normalized' else 1,301)
            if domain=='sphere':
                c=1+2*step*sigma*(1-.9**2)/np.sqrt(1+sigma**2)*(1-step**2/6-step*.9**2*sigma)
                second=(d-1)/2
            else:
                eta=step/np.sqrt(1+4*.9**2*sigma**2) if ball_step_rule=="normalized" else step
                c=1-omega+omega*(1+2*eta*sigma)/np.sqrt(1+4*.9**2*eta*sigma*(1+eta*sigma))
                second=(d+1)/2
            q=betaincc(.5,second,.9**2*c**(-200))
            ax.plot(step,q,color=color,ls=style,lw=2.4,label=f'd = {d}')
            records.append({'domain':domain,'d':d,'rank':d//2,'sigma':float(sigma),
                            'steps':step.tolist(),'lower_bound':q.tolist()})
    for ax,domain,mark in zip(axes,['Sphere','Ball'],[.2,.2 if ball_step_rule=='normalized' else .5]):
        ax.set_title(domain); ax.set_ylim(0,1.025); ax.set_xlim(left=0)
        ax.grid(alpha=.17); ax.axvline(mark,color='#525D65',lw=1,ls='--',alpha=.7)
        ax.text(mark,.045,'  current scale',rotation=90,fontsize=9,color='#525D65')
    axes[0].set_xlabel(r'Step scale $\alpha$'); axes[1].set_xlabel(r'Step scale $\alpha$' if ball_step_rule=='normalized' else r'Trial step $\eta$' if omega == 1 else rf'Trial step $\eta$ ($\lambda={omega}$)')
    axes[0].set_ylabel(r'Analytical lower bound for $q_{100}$')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.52,-.05))
    fig.subplots_adjust(wspace=.2,bottom=.18)
    save(fig,'step_size_probability_bounds'); dump(OUT/'step_size_bounds.json',records)

METHODS=['gd','pool256','pool4096','pool10000','grid1','grid0.5','grid0.25']
LABEL={'oracle':'Exact indicator','gd':'GD, 100 updates','pool256':'Pool 256',
 'pool4096':'Pool 4096','pool10000':'Pool 10000','grid1':'Grid h = 1',
 'grid0.5':'Grid h = 1/2','grid0.25':'Grid h = 1/4'}
ALIASES={'exact_indicator':'oracle','exp_rgd100':'gd','relaxed_pg100':'gd',
 'fixed_pool256':'pool256','fixed_pool4096':'pool4096','fixed_pool10000':'pool10000'}

def test_final_spaces(low,high,model_loader=None):
    all_stats=[]; max_defect=0.; max_relative_E_mismatch=0.
    for d in [6,32]:
        data=model_loader(d) if model_loader is not None else (np.load(EXT/f'low_d{d}.npz') if d==6 else np.load(OLD/f'fem_d{d}.npz'))
        G0=data['rotated_G0'] if d==6 else data['G0']
        H0=data['rotated_H0'] if d==6 else data['H0']
        options=rom_options(data)
        if d==6:options={k:data['rotated_'+k] for k in options}
        fig,axes=plt.subplots(1,2,figsize=(11.6,4.35))
        methods=['oracle']+METHODS if d==6 else ['oracle']+METHODS[:4]
        for domain,ax in zip(['sphere','ball'],axes):
            test_seed=202610180+100*d+(domain=='ball')
            Z=uniform_parameters(np.random.default_rng(test_seed),16384,d,domain)
            norm2=np.einsum('ij,ij->i',Z@G0,Z)
            assert np.min(norm2)>0
            arrays={'test_parameters':Z,'G0':G0,'H0':H0,**options}
            rows=[r for r in (low if d==6 else high) if r['dimension']==d and r['domain']==domain
                  and (d!=6 or r['orientation']=='rotated')]
            for mi,method in enumerate(methods):
                rr=[r for r in rows if (r['method'] if d==6 else ALIASES.get(r['method']))==method]
                stats=[]
                for r in rr:
                    rom=CoordinateROM(G0,H0,**options)
                    for step in r['rounds'][:d//2]: rom.append(step['parameter'])
                    P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross)
                    residual=Z@P.T
                    error2=np.einsum('ij,ij->i',residual@G0,residual)
                    assert np.min(error2)>-1e-14
                    rel=np.sqrt(np.maximum(error2,0)/norm2)
                    assert rel.max()<1+1e-8
                    _,G=rom.matrices(); E=np.sqrt(max(la.eigvalsh(G)[-1],0))
                    mismatch=abs(E-r['errors'][d//2])/r['errors'][d//2]
                    max_relative_E_mismatch=max(max_relative_E_mismatch,mismatch)
                    assert mismatch<1e-8
                    max_defect=max(max_defect,rom.orthogonality_defect())
                    qs=np.quantile(rel,[.5,.95,1])
                    stats.append(qs)
                    arrays[f'Q_{method}_seed{r["seed"]}']=rom.Q
                    all_stats.append({'d':d,'N':d//2,'domain':domain,'method':method,'training_seed':r['seed'],
                                      'test_seed':int(test_seed),'test_count':len(Z),'relative_error_median':float(qs[0]),
                                      'relative_error_p95':float(qs[1]),'relative_error_test_max':float(qs[2]),
                                      'relative_E_N_reconstruction_difference':float(mismatch)})
                assert len(stats)==(1 if d==6 and (method=='oracle' or method.startswith('grid')) else 8)
                q=np.median(stats,axis=0)
                color='#2A5374' if method=='gd' else '#B1813D' if method.startswith('pool') else '#537B71' if method.startswith('grid') else '#363E44'
                ax.plot(q[:2],[mi,mi],color=color,lw=2,alpha=.7)
                ax.plot(q[0],mi,'o',color=color,ms=6)
                ax.plot(q[1],mi,'|',color=color,ms=13,mew=2)
            ax.set_yticks(range(len(methods)),[LABEL[m] for m in methods]); ax.invert_yaxis()
            ax.grid(axis='x',alpha=.16)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_xlabel('Relative energy error'); ax.set_title(f'{domain.capitalize()}: d = {d}, N = {d//2}')
            np.savez_compressed(OUT/f'fixed_final_spaces_{domain}_d{d}.npz',**arrays)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([],[],marker='o',color='#4D5961',ls=''),Line2D([],[],marker='|',color='#4D5961',ls='',ms=13,mew=2)],
                   ['Test-set median','Test-set 95th percentile'],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,-.01))
        fig.tight_layout(rect=(0,.08,1,1)); save(fig,f'fixed_final_error_d{d}')
    dump(OUT/'fixed_final_test_errors.json',all_stats)
    return {'max_A_orthogonality_defect':max_defect,'max_relative_E_N_reconstruction_difference':max_relative_E_mismatch}

def timed_run(G0,H0,domain,method,seed,omega=0.5,ball_step_rule="fixed",rom_kwargs=None,updates=100,alpha=.2):
    # Timer encloses candidate generation, each matrix formation, selection,
    # and enrichment. Shared FEM/preprocessing and post-hoc audits are excluded.
    start=time.perf_counter()
    d=len(G0); rom=CoordinateROM(G0,H0,**(rom_kwargs or {})); rng=np.random.default_rng(seed)
    pool=None
    if method.startswith('grid'):
        pool,_=tensor_candidates(d,float(method[4:]),domain)
    elif method.startswith('pool'):
        pool=uniform_parameters(np.random.default_rng(seed+1_000_003),int(method[4:]),d,domain)
    setup_end=time.perf_counter(); params=[]
    for _ in range(d//2):
        H,_=rom.matrices(include_error=False)
        if method=='gd':
            point,_=search(H,uniform_parameters(rng,1,d,domain)[0],steps=updates,domain=domain,alpha=alpha,omega=omega,ball_step_rule=ball_step_rule,diagnostics=False)
        else: point,_=pool_search(H,pool)
        rom.append(point); params.append(point)
    end=time.perf_counter()
    return (end-start)*1000,(setup_end-start)*1000,np.array(params),rom

def timings(low,omega=0.5,ball_step_rule="fixed"):
    result=[]; rng=np.random.default_rng(271828)
    for d in [4,6]:
        data=np.load(EXT/f'low_d{d}.npz'); G0,H0=data['rotated_G0'],data['rotated_H0']
        for domain in ['sphere','ball']:
            # Warm up numerical kernels; warm-up is not a recorded run.
            timed_run(G0,H0,domain,'gd',9999,omega,ball_step_rule)
            jobs=[]
            for method in METHODS:
                seeds=[0] if method.startswith('grid') else range(1000,1008)
                for seed in seeds:
                    jobs += [(method,seed,rep) for rep in range(5 if method.startswith('grid') else 3)]
            rng.shuffle(jobs)
            raw={m:[] for m in METHODS}
            for method,seed,rep in jobs:
                ms,setup,params,rom=timed_run(G0,H0,domain,method,seed,omega,ball_step_rule)
                source=next(r for r in low if r['dimension']==d and r['domain']==domain and r['method']==method
                            and r['seed']==seed and r['orientation']=='rotated')
                expected=np.array([r['parameter'] for r in source['rounds'][:d//2]])
                assert np.max(np.abs(params-expected))<1e-9
                raw[method].append({'seed':seed,'repeat':rep,'wall_ms':ms,'candidate_setup_ms':setup})
            for method,entries in raw.items():
                per_seed=[np.median([r['wall_ms'] for r in entries if r['seed']==seed]) for seed in sorted({r['seed'] for r in entries})]
                ratio=np.median([r['errors'][d//2] for r in low if r['dimension']==d and r['domain']==domain and r['method']==method and r['orientation']=='rotated'])/np.sqrt(la.eigvalsh(G0)[::-1][d//2])
                result.append({'d':d,'N':d//2,'domain':domain,'method':method,'error_over_width_median':float(ratio),
                               'wall_ms_median':float(np.median(per_seed)),'raw_repetitions':entries})
            print('timing completed',d,domain,flush=True)
    dump(OUT/'wall_times.json',result)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        low=json.loads((EXT/'low_dimension_results.json').read_text())['runs']
        high=json.loads((OLD/'results.json').read_text())['runs']
        step_bounds(); print('step-size bounds completed',flush=True)
        validation=test_final_spaces(low,high); print('fixed-final test sets completed',validation,flush=True)
        timings(low)
        dump(OUT/'metadata.json',{'device':'local CPU','platform':platform.platform(),'python':sys.executable,'threadpools':threadpool_info(),
             'test_points_per_domain_dimension':16384,'test_metric':'||S z - u_N(z)||_A / ||S z||_A',
             'test_summary':'For each trained space compute test median and p95; plot their medians across 8 training seeds. Grids and low-d exact reference are deterministic.',
             'timing':'N=d/2, includes candidate setup + matrix formation + search + enrichment; excludes shared FEM/preprocessing, post-hoc error audit, tests and plotting.',
             'timing_repetitions':'3 per random training seed, 5 per deterministic grid, interleaved order',
             'step_bound_settings':{'T':100,'gamma':.9,'dimensions':[8,32,64,128],'rank':'d/2','ball_relaxation':.5},
             'sources':['runs/geometry_cpu_20261012','runs/geometry_extended_cpu_20261015'],'validation':validation,
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print('Complete:',OUT,flush=True)
