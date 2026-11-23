"""Reproduce the current meeting experiments using consistent FEM snapshots.

Run: PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.common.run_consistent_fem
Only new run files are written. Reporting is a separate command.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time,argparse
import numpy as np
from scipy.linalg import qr
from threadpoolctl import threadpool_info
from ..consistent_fem import build_model,EnergyROM
from .cached_pool import CachedEnergyPool
from ..geometry_cpu import search,uniform_parameters,pool_search
from ..cube.search import search as cube_search
from ..cube.boundary_strategies import search as boundary_search
from ..cube.test_set import make_vertex_test_set
from ..run_extended_comparison_cpu import tensor_candidates
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'runs/consistent_fem_20261113'
DIMS=[4,6]+list(range(8,65,4))+[128];SEEDS=range(1000,1008)
TABLE_DIMS=[4,8,16,32,64,128];CURVE_DIMS=[6,16,32,64];BOUNDARY_DIMS=[6,8,16,32,48,64]
METHODS=['gd','pool256','pool4096','pool10000'];GRIDS=['grid1','grid0.5','grid0.25']

def dump(name,x):
    path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(path)

def sample(rng,d,dom,count=1):
    return rng.uniform(-1,1,(count,d)) if dom=='cube' else uniform_parameters(rng,count,d,dom)

def model(d):
    p=OUT/f'model_d{d}.npz'
    if not p.exists():
        a=build_model(d);O,R=qr(np.random.default_rng(20261015+d).normal(size=(d,d)));O*=np.sign(np.diag(R))[None,:]
        np.savez_compressed(p,**a,rotation=O)
        print('MODEL',d,'AS-B',a['relative_discrete_equation_defect'],flush=True)
    return dict(np.load(p))

def coordinates(a,dom):return a['C']@a['rotation'] if len(a['C'])<8 and dom!='cube' else a['C']

def tests(d,dom):
    p=OUT/f'test_{dom}_d{d}.npz'
    if p.exists():return np.load(p)['parameters']
    if dom=='cube':Z=make_vertex_test_set(d)[0]
    elif d==6:
        Z=np.load(ROOT/f'runs/analytic_hf_a06_t40_20261022/fixed_final_spaces_{dom}_d6.npz')['test_parameters']
    else:
        Z=uniform_parameters(np.random.default_rng(202610180+100*d+(dom=='ball')),16384,d,dom)
        if d==4:
            f=ROOT/f'runs/analytic_hf_a06_t40_20261022/fixed_final_spaces_{dom}_d4.npz'
            if f.exists():Z=np.load(f)['test_parameters']
    np.savez_compressed(p,parameters=Z);return Z

def train(C,K,dom,method,seed,N,grid=None,*,cache_pool=True,diagnostics=True):
    start=time.perf_counter();rom=EnergyROM(C,K);d=len(C);rng=np.random.default_rng(seed)
    pool=grid
    if method.startswith('pool'):pool=sample(np.random.default_rng(seed+1000003),d,dom,int(method[4:]))
    elif method.startswith('grid') and pool is None:
        h=float(method[4:]);axis=np.linspace(-1,1,round(2/h)+1)
        pool=np.stack(np.meshgrid(*([axis]*d)),axis=-1).reshape(-1,d) if dom=='cube' else tensor_candidates(d,h,dom)[0]
    cached = CachedEnergyPool(pool,C) if pool is not None and cache_pool and np.array_equal(K,np.eye(d)) else None
    params=[];novelty=[];elapsed=[];stop=None;final_H=None;selected=None
    for n in range(N):
        H=rom.matrix() if cached is None else None
        if method=='gd':
            initial=sample(rng,d,dom)[0]
            z,_=(cube_search(H,initial,alpha=.6,T=80,diagnostics=False) if dom=='cube' else search(H,initial,steps=40,domain=dom,alpha=.6,ball_step_rule='normalized',diagnostics=False))
        elif method in ['active_face','normal_scaled']:
            z,_=boundary_search(H,sample(rng,d,dom)[0],method=method,steps=80,alpha=.6)
        elif cached is not None:z=cached.select()
        else:z,_=pool_search(H,pool)
        if diagnostics and n==d//2-1:
            final_H=(rom.matrix() if H is None else H).copy();selected=z.copy()
        try:v=rom.append(z)
        except RuntimeError as e:stop=str(e);break
        if cached is not None and n<N-1:cached.update(rom.Q[:,-1])
        params.append(z.tolist());novelty.append(v);elapsed.append((time.perf_counter()-start)*1000)
    return rom,dict(parameters=params,novelty=novelty,elapsed_ms=elapsed,stop=stop,rank=rom.Q.shape[1]),final_H,selected

def error(C,Q,Z):
    E=C-Q@(Q.T@C);v=Z@E.T
    return float(np.sqrt(np.max(np.sum(v*v,axis=1))))

def main():
    OUT.mkdir(exist_ok=True)
    dump('config.json',dict(dimensions=DIMS,seeds=list(SEEDS),alpha=.6,T_sphere_ball=40,T_cube=80,rho=.78,
         hf='AS=B solved with the original assembled loads; energy-coordinate Galerkin projection',
         n_grid=40,element_order=3,blas=threadpool_info(),timing='training only, excludes shared HF setup and diagnostics',
         boundary_dimensions=BOUNDARY_DIMS))
    rows=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    keys={(r['d'],r['domain'],r['method'],r['seed']) for r in rows}
    for d in DIMS:
        a=model(d)
        for dom in ['sphere','ball','cube']:
            C=coordinates(a,dom);Z=tests(d,dom)
            methods=METHODS+(GRIDS if d<8 else [])+(['active_face','normal_scaled'] if dom=='cube' and d in BOUNDARY_DIMS else [])
            for method in methods:
                seeds=[0] if method.startswith('grid') else SEEDS
                for seed in seeds:
                    if (d,dom,method,seed) in keys:continue
                    maxN=d-1 if d in CURVE_DIMS and method not in ['active_face','normal_scaled'] else d//2
                    rom,r,H,z=train(C,a['K'],dom,method,seed,maxN)
                    target=min(d//2,r['rank']);r.update(d=d,domain=dom,method=method,seed=seed,target_rank=d//2)
                    if d in CURVE_DIMS or (dom=='cube' and method in ['active_face','normal_scaled']):
                        r['errors']=[error(C,rom.Q[:,:n],Z) for n in range(r['rank']+1)]
                        r['test_max_error']=r['errors'][target]
                    else:r['test_max_error']=error(C,rom.Q[:,:target],Z)
                    E=C-rom.Q[:,:target]@(rom.Q[:,:target].T@C)
                    r['exact_domain_error']=float(np.linalg.norm(E,2)) if dom!='cube' else None
                    r['width']=float(.78**target*(np.sqrt((1-.78**(2*(d-target)))/(1-.78**2)) if dom=='cube' else 1))
                    r['ratio']=(r['exact_domain_error'] if dom!='cube' else r['test_max_error'])/r['width']
                    r['orthogonality_defect']=float(np.linalg.norm(rom.Q.T@rom.Q-np.eye(r['rank'])))
                    np.savez_compressed(OUT/f'basis_{dom}_d{d}_{method}_s{seed}.npz',Q=rom.Q,final_search_H=H,selected=z)
                    rows.append(r);keys.add((d,dom,method,seed));dump('results.json',rows)
                print('TRAIN',dom,d,method,'ranks',[r['rank'] for r in rows if (r['d'],r['domain'],r['method'])==(d,dom,method)],flush=True)
    dump('validation.json',dict(runs=len(rows),stopped=[{k:r[k] for k in ['d','domain','method','seed','rank','stop']} for r in rows if r['stop']],max_orthogonality_defect=max(r['orthogonality_defect'] for r in rows)))
    print('TRAINING COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
