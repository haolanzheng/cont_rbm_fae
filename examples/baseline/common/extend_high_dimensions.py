"""Extend the unchanged analytic baseline to d=128,256; retain failed enrichment.

No change to rho, alpha, search budget, pool reuse, grid, or enrichment tolerance.
Run with --output under runs/. Each training run records actual rank and elapsed
training time; only complete runs qualify for the N=d/2 tables.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json,time,platform,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, norm
from threadpoolctl import threadpool_limits,threadpool_info
from ..analytic_model import build_model
from ..geometry_cpu import CoordinateROM,rom_options,uniform_parameters,search,pool_search
from ..cube.search import search as cube_search
from ..cube.test_set import make_vertex_test_set
from .benchmark_training import sample
ROOT=Path(__file__).resolve().parents[3]


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def train(a,domain,method,seed):
    d=len(a['G0']);rng=np.random.default_rng(seed)
    begin=time.perf_counter();rom=CoordinateROM(a['G0'],a['H0'],**rom_options(a));pool=None
    if method!='gd':pool=sample(np.random.default_rng(seed+1000003),d,domain,int(method[4:]))
    elapsed=time.perf_counter()-begin
    params=[];ranks=[];stop=None
    for n in range(d//2):
        begin=time.perf_counter();H,_=rom.matrices(include_error=False)
        if method=='gd':
            initial=sample(rng,d,domain)[0]
            z,_=(cube_search(H,initial,alpha=.6,T=80,diagnostics=False) if domain=='cube' else
                 search(H,initial,alpha=.6,steps=40,domain=domain,omega=1,ball_step_rule='normalized',diagnostics=False))
        else:z,_=pool_search(H,pool)
        try:rom.append(z)
        except RuntimeError as e:stop=str(e)
        elapsed+=time.perf_counter()-begin
        if stop:break
        params.append(z.tolist())
    return elapsed*1000,rom,params,stop


def evaluate(a,rom,Z,domain):
    # Factor the error as U(I-Q Q^T cross). This avoids eigendecomposing
    # a matrix with squared, exponentially small eigenvalues.
    P=np.eye(len(a['G0']))-rom.Q@(rom.Q.T@rom.cross)
    U=cholesky(a['G0'],lower=False)
    F=U@P
    maximum=0.
    for start in range(0,len(Z),2048):
        values=Z[start:start+2048]@F.T
        maximum=max(maximum,float(np.max(np.sum(values*values,axis=1))))
    return float(np.sqrt(maximum)),float(norm(F,2)) if domain!='cube' else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';rows=json.loads(path.read_text())['runs'] if path.exists() else []
    completed={(r['domain'],r['d'],r['method'],r['seed']) for r in rows}
    with threadpool_limits(limits=1):
        dump(out/'config.json',dict(dimensions=[128,256],seeds=list(range(1000,1008)),rho=.78,alpha=.6,
            sphere_ball_steps=40,cube_steps=80,target_rank='d/2',blas_threads=threadpool_info(),platform=platform.platform(),
            sphere_ball_test_count=16384,cube_vertex_count=65536,test_seed=202610270,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        for d in [128,256]:
            modelpath=out/f'model_d{d}.npz'
            if not modelpath.exists():
                if d==128:
                    # Same analytic-snapshot model/grid as the current experiments.
                    a=dict(np.load(ROOT/'runs/analytic_hf_20261022/model_d128.npz'))
                else:
                    print(f'Building unchanged analytic model d={d}',flush=True);a=build_model(d)
                np.savez_compressed(modelpath,**a)
            a=dict(np.load(modelpath)); print(f'Model ready d={d}',flush=True)
            for domain in ['sphere','ball','cube']:
                testpath=out/f'test_{domain}_d{d}.npz'
                if testpath.exists():Z=np.load(testpath)['parameters']
                else:
                    Z=(make_vertex_test_set(d,65536,202610270)[0] if domain=='cube' else
                       uniform_parameters(np.random.default_rng(202610270+d),16384,d,domain))
                    np.savez_compressed(testpath,parameters=Z)
                for method in ['gd','pool256','pool4096','pool10000']:
                    for seed in range(1000,1008):
                        if (domain,d,method,seed) in completed:continue
                        ms,rom,params,stop=train(a,domain,method,seed)
                        error,exact=evaluate(a,rom,Z,domain)
                        width=float(.78**(d//2)* (np.sqrt((1-.78**(2*(d-d//2)))/(1-.78**2)) if domain=='cube' else 1))
                        row=dict(domain=domain,d=d,method=method,seed=seed,target_rank=d//2,rank=rom.Q.shape[1],
                            reached_target=rom.Q.shape[1]==d//2,stop=stop,wall_ms=ms,parameters=params,
                            test_max_error=error,exact_domain_error=exact,width=width,
                            ratio=(exact if exact is not None else error)/width,orthogonality_defect=rom.orthogonality_defect())
                        rows.append(row);np.savez_compressed(out/f'basis_{domain}_d{d}_{method}_s{seed}.npz',Q=rom.Q)
                        dump(path,dict(runs=rows))
                        print(domain,d,method,seed,'rank',rom.Q.shape[1],'test error',error,'ms',ms,flush=True)
    print('Completed',len(rows),'runs',flush=True)

if __name__=='__main__':main()
