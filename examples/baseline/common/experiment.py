"""Reproducible consistent-FEM baseline, single-thread CPU, all domains.

Training does not use evaluation maxima or test points. Results include all
selected parameters, reduced bases, fixed test sets and configuration hashes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from ..geometry_cpu import CoordinateROM, rom_options, uniform_parameters, search
from ..cube.search import search as cube_search, scores
from .box_bounds import maximum, width
from .cached_pool import CachedEnergyPool
from ..cube.test_set import make_vertex_test_set
ROOT=Path(__file__).resolve().parents[3]


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sample(rng,count,d,domain):
    return rng.uniform(-1,1,(count,d)) if domain=='cube' else uniform_parameters(rng,count,d,domain)


def main(domain):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/domain/'config.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-cache',type=Path,default=ROOT/'runs/consistent_fem_20261113')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if cfg['rho']!=.78 or cfg['contrast']!=.25:raise ValueError('The baseline model currently fixes rho=.78 and contrast=.25; rebuild the model to change them.')
    if cfg['updates']<0 or cfg['alpha']<=0 or cfg['test_count']<1:raise ValueError('Invalid search or test configuration.')
    if any(d<2 or d%2 for d in cfg['dimensions']):raise ValueError('These N=d/2 comparisons require positive even dimensions.')
    if (out/'results.json').exists():raise FileExistsError('Use a fresh output directory; existing results are preserved.')
    dump(out/'config.json',cfg|{'domain':domain,'platform':platform.platform(),'threads':threadpool_info(),
        'model_cache':str(args.model_cache),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    records=[];widths={}
    with threadpool_limits(limits=1):
        for d in cfg['dimensions']:
            path=args.model_cache/f'model_d{d}.npz'
            if not path.exists():
                from ..consistent_fem import build_model
                path=out/f'model_d{d}.npz';np.savez_compressed(path,**build_model(d))
            a=np.load(path)
            if 'relative_discrete_equation_defect' not in a or float(a['relative_discrete_equation_defect'])>1e-10:
                raise ValueError('Model cache does not certify AS=B; rebuild with consistent_fem.build_model.')
            G0=a['G0'];metric=cfg.get('residual_norm','energy_dual')
            if metric not in ('energy_dual','reference_laplacian'):raise ValueError('Unknown residual norm')
            H0=G0 if metric=='energy_dual' else a['H0'];N=d//2
            widths[str(d)]=width(G0,N) if domain=='cube' else [float(np.sqrt(np.linalg.eigvalsh(G0)[-N-1]))]*2
            if domain=='cube':
                vertices,exhaustive=make_vertex_test_set(d,cfg.get('test_vertex_count',65536),cfg['test_seed'])
                Z=vertices
                np.savez_compressed(out/f'test_vertices_d{d}.npz',parameters=vertices,exhaustive_up_to_sign=exhaustive)
            else:
                Z=sample(np.random.default_rng(cfg['test_seed']+d),cfg['test_count'],d,domain)
                np.savez_compressed(out/f'test_d{d}.npz',parameters=Z)
                vertices=Z
            den=scores(G0,Z)
            methods=['gd']+[f'pool{m}' for m in cfg['pool_sizes']]
            if domain=='cube' and d in cfg['grid_dimensions']:methods += [f'grid{h:g}' for h in cfg['grid_spacings']]
            for method in methods:
                for seed in ([cfg['seeds'][0]] if method.startswith('grid') else cfg['seeds']):
                    rng=np.random.default_rng(seed);rom=CoordinateROM(G0,H0,**rom_options(a));rounds=[];curve=[float(np.sqrt(den.max()))]
                    begin=time.perf_counter();pool=None
                    if method.startswith('pool'):pool=sample(np.random.default_rng(seed+1000003),int(method[4:]),d,domain)
                    if method.startswith('grid'):
                        levels=np.linspace(-1,1,round(2/float(method[4:]))+1)
                        pool=np.stack(np.meshgrid(*([levels]*d),indexing='ij'),axis=-1).reshape(-1,d)
                    cached=CachedEnergyPool(pool,a['C']) if pool is not None and metric=='energy_dual' else None
                    elapsed=time.perf_counter()-begin
                    for n in range(N):
                        begin=time.perf_counter()
                        H=rom.matrices(include_error=False)[0] if cached is None else None
                        if method=='gd':
                            initial=sample(rng,1,d,domain)[0]
                            z,audit=(cube_search(H,initial,alpha=cfg['alpha'],T=cfg['updates'],diagnostics=False) if domain=='cube' else search(H,initial,steps=cfg['updates'],domain=domain,alpha=cfg['alpha'],ball_step_rule='normalized',diagnostics=False))
                        elif cached is not None:z=cached.select();audit={}
                        else:z=pool[np.argmax(scores(H,pool))].copy();audit={}
                        value=float(z@H@z) if H is not None else float(cached.scores.max())
                        rom.append(z)
                        if cached is not None and n<N-1:cached.update(a['C']@rom.Q[:,-1])
                        elapsed+=time.perf_counter()-begin
                        rounds.append({'n':n,'parameter':z.tolist(),'selected_value':value,'audit':audit})
                        _,G=rom.matrices();sq=np.maximum(scores(G,Z),0);curve.append(float(np.sqrt(sq.max())))
                    if cached is not None:
                        # Reconstruct only the final search matrix for diagnostics, outside timing.
                        Qprev=rom.Q[:,:-1];P=np.eye(d)-Qprev@(Qprev.T@G0)
                        H=P.T@G0@P;H=(H+H.T)/2
                    # All extrema below are evaluated after training and outside wall time.
                    ml,mu=maximum(H) if domain=='cube' else [float(np.linalg.eigvalsh(H)[-1])]*2
                    el,eu=maximum(G) if domain=='cube' else [float(np.linalg.eigvalsh(G)[-1])]*2
                    wl,wu=widths[str(d)];value=rounds[-1]['selected_value']
                    test_maximum=float(np.sqrt(max(0,scores(G,vertices).max())))
                    exact_width=float(np.sqrt(sum(cfg['rho']**(2*j) for j in range(N,d)))) if domain=='cube' else float(cfg['rho']**N)
                    rec={'domain':domain,'d':d,'N':N,'method':method,'seed':seed,'rounds':rounds,
                         'test_max_error_by_N':curve,'test_max_error':test_maximum,
                         'test_max_error_ratio':test_maximum/exact_width,
                         'test_set':'vertices_max' if domain=='cube' else 'uniform',
                         'vertex_test_count':len(vertices) if domain=='cube' else None,
                         'test_count':len(Z),'wall_ms':elapsed*1000,'width_interval':[wl,wu],
                         'error_interval':np.sqrt([el,eu]).tolist(),
                         'exact_analytic_width':exact_width,
                         'ratio_interval':(np.sqrt([el,eu])/exact_width).tolist() if domain=='cube' else [np.sqrt(el)/wu,np.sqrt(eu)/wl],
                         'ratio_denominator':'exact analytic width' if domain=='cube' else 'sampled model width',
                         'indicator_max_interval':[ml,mu],'indicator_quality_interval':np.sqrt([value/mu,value/ml]).tolist(),
                         'orthogonality_defect':rom.orthogonality_defect()}
                    records.append(rec);np.savez_compressed(out/f'basis_d{d}_{method}_s{seed}.npz',Q=rom.Q)
                    dump(out/'results.json',{'config':cfg,'domain':domain,'widths':widths,'runs':records})
                subset=[r for r in records if r['d']==d and r['method']==method]
                print(d,method,'test',np.median([r['test_max_error_by_N'][-1] for r in subset]),'ratio',np.median([r['ratio_interval'] for r in subset],axis=0),flush=True)
