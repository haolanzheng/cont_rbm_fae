"""Rerun all sphere/ball comparisons using analytic solutions, on one CPU.

PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.run_analytic_baseline
Archived FEM experiments are preserved. No diffusion solve is used to generate snapshots; the original discrete residual norm is retained.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/analytic_baseline_matplotlib')
import json,hashlib,platform,sys,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits,threadpool_info
from .analytic_model import build_model
from .geometry_cpu import CoordinateROM,uniform_parameters,rom_options
from .run_geometry_cpu import run_greedy
from .run_extended_comparison_cpu import selector_run,tensor_candidates,frozen_space,batch_probability
from . import make_slides_comparison as compare
from . import make_test_error_curves as curves

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/analytic_hf_a06_t40_20261022';FIG=OUT/'figures'
SEEDS=list(range(1000,1008));DIMS=[8,16,32,48,64]
CONFIG={'dimensions':DIMS,'seeds':SEEDS,'updates':40,'sphere_alpha':.6,'alpha':.6,
        'ball_omega':1.,'rho':.78,'contrast':.25,'pool_sizes':[256,4096,10000],
        'step_rule':'eta(z)=alpha/sqrt(1+||gradient J(z)||^2)',
        'device':'local_cpu','dtype':'float64','blas_threads':1,'solution':'analytic solution sampled at original P3 finite-element nodes; no HF diffusion solve',
        'residual_norm':'unchanged: R.T W^-1 R, R=B-A V V.T B'}

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def model(d):return np.load(OUT/f'model_d{d}.npz')

def setup():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    validations=[]
    for d in [4,6]+DIMS:
        path=OUT/f'model_d{d}.npz'
        if path.exists():
            a=np.load(path)
        else:
            a=build_model(d)
            O,R=la.qr(np.random.default_rng(20261015+d).normal(size=(d,d)))
            O*=np.sign(np.diag(R))[None,:]
            np.savez_compressed(path,**a,rotation=O,rotated_G0=O.T@a['G0']@O,rotated_H0=O.T@a['H0']@O,**{'rotated_'+k:O.T@v@O for k,v in rom_options(a).items()})
        validations.append({k:float(a[k]) for k in ['sampled_solution_energy_error','relative_discrete_equation_defect','relative_cross_difference','residual_matrix_check','error_matrix_check','quadrature_order','reference_dofs']}|{'d':d})
        print('Analytic snapshots sampled; original residual norm:',d,flush=True)
    dump('model_validation.json',validations)
    for name in ['paper/continuous_greedy_manifold_attainment.tex','paper/slides_short/slides.tex']:
        dst=OUT/'tex_before'/name.replace('/','__');dst.parent.mkdir(exist_ok=True)
        if not dst.exists():dst.write_bytes((ROOT/name).read_bytes())

def train():
    high=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else {'problems':{},'runs':[]}
    high['config']=CONFIG
    high['runs']=[r for r in high['runs'] if r['dimension'] in DIMS]
    high['problems']={k:v for k,v in high['problems'].items() if int(k) in DIMS}
    existing={(r['dimension'],r['domain'],r['method'],r['seed']) for r in high['runs']}
    for d in DIMS:
        a=model(d);high['problems'][str(d)]={'widths':a['widths'].tolist()}
        for domain in ['sphere','ball']:
            methods=['exact_indicator','exp_rgd40' if domain=='sphere' else 'normalized_pg40',
                     'fixed_pool256','fixed_pool4096','fixed_pool10000','fresh_pool41']
            for method in methods:
                for seed in SEEDS:
                    if (d,domain,method,seed) in existing:continue
                    high['runs'].append(run_greedy(a['G0'],a['H0'],domain,method,seed,CONFIG,rom_kwargs=rom_options(a)))
            print('Greedy histories complete:',domain,d,flush=True)
        dump('results_partial.json',high)
    dump('results.json',high)
    low=json.loads((OUT/'low_dimension_results.json').read_text()) if (OUT/'low_dimension_results.json').exists() else {'widths':{},'grid_counts':[],'runs':[]}
    existing_low={(r['dimension'],r['domain'],r['orientation'],r['method'],r['seed']) for r in low['runs']}
    for d in [4,6]:
        a=model(d);low['widths'][str(d)]=a['widths'].tolist()
        for domain in ['sphere','ball']:
            for orientation in ['rotated','original']:
                G,H=(a['rotated_G0'],a['rotated_H0']) if orientation=='rotated' else (a['G0'],a['H0'])
                opts={k:a['rotated_'+k] for k in rom_options(a)} if orientation=='rotated' else rom_options(a)
                for method in (['oracle']+compare.METHODS[:4] if orientation=='rotated' else ['oracle']):
                    for seed in ([1000] if method=='oracle' else SEEDS):
                        if (d,domain,orientation,method,seed) in existing_low:continue
                        r=selector_run(G,H,domain,method,seed,CONFIG['updates'],d-1,omega=1.,ball_step_rule='normalized',rom_kwargs=opts,alpha=CONFIG['alpha'])
                        r['orientation']=orientation;low['runs'].append(r)
                for h in [1,.5,.25]:
                    if (d,domain,orientation,f'grid{h:g}',0) in existing_low:continue
                    points,meta=tensor_candidates(d,h,domain)
                    if orientation=='rotated':low['grid_counts'].append(meta)
                    r=selector_run(G,H,domain,f'grid{h:g}',0,100,d-1,points,omega=1.,ball_step_rule='normalized',rom_kwargs=opts)
                    r.update(orientation=orientation,grid_h=h,candidate_count=len(points));low['runs'].append(r)
            print('Low-dimensional methods and grids rerun:',domain,d,flush=True)
    dump('results.json',high);dump('low_dimension_results.json',low)
    return high,low

def tests(high,low):
    compare.OUT=OUT;compare.FIG=FIG;compare.ALIASES['normalized_pg40']='gd';compare.ALIASES['exp_rgd40']='gd';compare.LABEL['gd']='GD, 40 updates';curves.LABELS[0]='GD, 40 updates'
    checks=compare.test_final_spaces(low['runs'],high['runs'],model_loader=model)
    curves.SOURCE=OUT;curves.OUT=OUT/'test_error_curves';curves.FIG=FIG;curves.main()
    summary=[];records=[]
    for d in DIMS:
        a=model(d);G0,H0=a['G0'],a['H0']
        for dom in ['sphere','ball']:
            seed=202610180+100*d+(dom=='ball')
            Z=uniform_parameters(np.random.default_rng(seed),16384,d,dom)
            # Preserve the previous independent test samples exactly.
            old=ROOT/f'runs/final_test_errors_20261020/test_{dom}_d{d}.npz'
            if old.exists():assert np.array_equal(Z,np.load(old)['test_parameters'])
            np.savez_compressed(OUT/f'test_{dom}_d{d}.npz',test_parameters=Z)
            den=np.sum((Z@G0)*Z,axis=1)
            for method in ['exp_rgd40' if dom=='sphere' else 'normalized_pg40','fixed_pool256','fixed_pool4096','fixed_pool10000']:
                vals=[]
                for r in [r for r in high['runs'] if r['dimension']==d and r['domain']==dom and r['method']==method]:
                    rom=CoordinateROM(G0,H0,**rom_options(a))
                    for step in r['rounds']:rom.append(step['parameter'])
                    P=np.eye(d)-rom.Q@(rom.Q.T@rom.cross);e=Z@P.T
                    rms=float(np.sqrt(np.mean(np.maximum(np.sum((e@G0)*e,axis=1),0)/den)));vals.append(rms)
                    records.append({'domain':dom,'d':d,'N':d//2,'method':method,'seed':r['seed'],'rms':rms})
                summary.append({'domain':dom,'d':d,'N':d//2,'method':method,'median_rms_relative_energy_error':float(np.median(vals))})
    dump('final_test_errors.json',summary);dump('final_test_errors_per_seed.json',records)
    return checks

def timings(low,high):
    rows=[];rng=np.random.default_rng(271828)
    for d in [4,6,32]:
        a=model(d);G,H=(a['rotated_G0'],a['rotated_H0']) if d<8 else (a['G0'],a['H0'])
        opts={k:a['rotated_'+k] for k in rom_options(a)} if d<8 else rom_options(a)
        for dom in ['sphere','ball']:
            compare.timed_run(G,H,dom,'gd',9999,1.,'normalized',rom_kwargs=opts,updates=CONFIG['updates'],alpha=CONFIG['alpha'])
            methods=compare.METHODS if d<8 else compare.METHODS[:4]
            jobs=[(m,s,r) for m in methods for s in ([0] if m.startswith('grid') else SEEDS)
                  for r in range(5 if m.startswith('grid') else 3)]
            rng.shuffle(jobs);raw={m:[] for m in methods}
            for m,seed,rep in jobs:
                ms,setup,params,rom=compare.timed_run(G,H,dom,m,seed,1.,'normalized',rom_kwargs=opts,updates=CONFIG['updates'],alpha=CONFIG['alpha'])
                key=m if d<8 else (('exp_rgd40' if dom=='sphere' else 'normalized_pg40') if m=='gd' else 'fixed_'+m)
                src=next(r for r in (low['runs'] if d<8 else high['runs']) if r['dimension']==d and r['domain']==dom
                         and r['method']==key and r['seed']==seed and (d>=8 or r['orientation']=='rotated'))
                assert np.max(np.abs(params-np.array([s['parameter'] for s in src['rounds'][:d//2]])))<1e-9
                raw[m].append({'seed':seed,'repeat':rep,'wall_ms':ms,'candidate_setup_ms':setup,
                               'error_over_width':src['errors'][d//2]/a['widths'][d//2]})
            for m,entries in raw.items():
                times=[np.median([r['wall_ms'] for r in entries if r['seed']==seed]) for seed in sorted({r['seed'] for r in entries})]
                rows.append({'d':d,'N':d//2,'domain':dom,'method':m,'wall_ms_median':float(np.median(times)),
                             'error_over_width_median':float(np.median([r['error_over_width'] for r in entries])),
                             'raw_repetitions':entries})
            print('Interleaved wall times:',dom,d,flush=True)
    dump('wall_times.json',rows)

def probabilities():
    rows=[]
    for d in DIMS:
        a=model(d)
        for n in [0,d//4,d//2]:
            H,Q=frozen_space(a['G0'],a['H0'],n,rom_kwargs=rom_options(a))
            np.savez_compressed(OUT/f'reference_d{d}_n{n}.npz',H=H,Q=Q)
            for dom in ['sphere','ball']:
                r=batch_probability(H,dom,4096,CONFIG['updates'],20261015+100*d+n,.9,OUT,n,
                                    ball_omegas=(1.,),ball_step_rule='normalized',alpha=CONFIG['alpha'])
                rows.append(r)
                method='exp_rgd' if dom=='sphere' else 'normalized_pg'
                print('Probability:',dom,d,n,r['methods'][method]['snapshots'][-1]['hits'],flush=True)
            dump('probability_curves.json',rows)
    return rows

def main():
    setup();high,low=train();checks=tests(high,low)
    print('Shared test-set comparisons complete',flush=True)
    if not (OUT/'wall_times.json').exists():timings(low,high)
    probabilities()
    dump('metadata.json',{'config':CONFIG,'platform':platform.platform(),'python':sys.executable,
         'threadpools':threadpool_info(),'test_count':16384,'probability_repetitions':4096,
         'initialization':'uniform surface or uniform volume; all prior seeds retained',
         'timings':'Rerun on the same CPU; 3 interleaved repetitions per seed, 5 per grid',
         'reused_methods':'Exact indicator, fixed pools, grids; GD and fresh 41-point pools rerun',
         'source_run':'runs/analytic_hf_a1_t20_20261022',
         'theoretical_figures':{'source':'runs/analytic_hf_a06_t40_20261022','alpha':.6,'T':40,'sphere_condition':'1-alpha^2*sigma^2/(6*(1+sigma^2))-alpha*gamma^2*sigma > 0'},
         'validation':checks,'model_validation':json.loads((OUT/'model_validation.json').read_text()),
         'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                         ['examples/baseline/analytic_model.py','examples/baseline/run_analytic_baseline.py']}})
    print('ALL EXPERIMENTS COMPLETE',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
