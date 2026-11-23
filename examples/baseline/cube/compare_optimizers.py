"""Compare GD, SLSQP, Coleman-Li reflective TR and trust-constr on the existing cube baseline.

Python optimizer comparison on a common objective, not a full paper replication.
Install fides==0.7.8 and make it available on PYTHONPATH for reflective runs.
Run: PYTHONPATH=. .venv_cpu/bin/python -u -m examples.baseline.cube.compare_optimizers
Existing FEM models, starts and test sets are reused; old results are never changed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize, Bounds
from threadpoolctl import threadpool_info
from ..consistent_fem import EnergyROM
from .search import search as gd_search

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'runs/consistent_fem_20261113'
OUTPUT=ROOT/'runs/cube_optimizer_comparison_20261115'
METHODS=('gd','sqp','reflective','trust')
INITS=('random','pool256')


def dump(path, obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)


def optimize(H, initial, method, max_steps=None):
    """Minimize -z^T H z / ||H||_F; retain the best feasible evaluation."""
    if method not in METHODS:raise ValueError(method)
    limit=500 if max_steps is None else int(max_steps)
    gd_steps=80 if max_steps is None else int(max_steps)
    if limit<1:raise ValueError("max_steps must be positive")
    ell=np.linalg.norm(H);A=H/ell if ell else H.copy()
    best=initial.copy();best_value=float(best@A@best);calls=0
    last_x=None;last_hz=None
    def evaluate(z):
        nonlocal best,best_value,calls,last_x,last_hz
        if last_x is None or not np.array_equal(z,last_x):
            last_x=z.copy();last_hz=A@z;calls+=1
            value=float(z@last_hz)
            if np.max(np.abs(z))<=1+1e-12 and value>best_value:
                best=np.clip(z,-1,1);best_value=float(best@A@best)
        return -float(z@last_hz),-2*last_hz
    if method=='gd':
        best,_=gd_search(H,initial,alpha=.6,T=gd_steps,diagnostics=False)
        stats=dict(nit=gd_steps,nfev=gd_steps+1,njev=gd_steps,nhev=0,cg_niter=0,h_actions=gd_steps+1,solver_success=None,status='fixed_budget')
    elif method=='reflective':
        import logging
        import fides
        from fides.constants import SubSpaceDim, StepBackStrategy
        def objective(z):
            value,grad=evaluate(z)
            return value,grad,-2*A
        solver=fides.Optimizer(objective,ub=np.ones(len(initial)),lb=-np.ones(len(initial)),
                    verbose=logging.ERROR,options=dict(maxiter=limit,fatol=1e-10,frtol=1e-10,
                    xtol=1e-10,gatol=1e-8,subspace_solver=SubSpaceDim.TWO,
                    stepback_strategy=StepBackStrategy.SINGLE_REFLECT))
        result=solver.minimize(initial)
        evaluate(result[1])
        stats=dict(nit=int(solver.iteration),nfev=calls,njev=calls,nhev=calls,cg_niter=None,
                   h_actions=calls,solver_success=bool(solver.converged),status=str(solver.exitflag))
    else:
        if method=='sqp':
            result=minimize(evaluate,initial,jac=True,method='SLSQP',bounds=[(-1,1)]*len(initial),
                            options=dict(ftol=1e-9,maxiter=limit))
        else:
            result=minimize(evaluate,initial,jac=True,hess=lambda z:-2*A,method='trust-constr',
                            bounds=Bounds(-np.ones(len(initial)),np.ones(len(initial)),keep_feasible=True),
                            options=dict(gtol=1e-12,xtol=1e-10,barrier_tol=1e-10,maxiter=limit,sparse_jacobian=True))
        evaluate(result.x)
        stats={k:int(result.get(k,0)) for k in ('nit','nfev','njev','nhev','cg_niter')}
        stats.update(h_actions=calls,solver_success=bool(result.success),status=str(result.message))
    g=2*A@best
    stats.update(value=float(best@A@best),initial_value=float(initial@A@initial),
                 projected_gradient_inf=float(np.max(np.abs(np.clip(best+g,-1,1)-best))),
                 feasibility_defect=float(np.maximum(np.abs(best)-1,0).max()))
    return best,stats


def train(C,K,method,initialization,seed,N,max_steps=None):
    begin=time.perf_counter();rom=EnergyROM(C,K);rng=np.random.default_rng(seed)
    pool=np.random.default_rng(seed+1000003).uniform(-1,1,(256,len(C))) if initialization=='pool256' else None
    elapsed=[];params=[];rounds=[];stop=None
    for n in range(N):
        H=rom.matrix();initial=rng.uniform(-1,1,len(C));init_evals=0
        if pool is not None:
            values=np.sum((pool@H)*pool,axis=1);initial=pool[np.argmax(values)].copy();init_evals=len(pool)
        z,stats=optimize(H,initial,method,max_steps=max_steps)
        try:rom.append(z)
        except RuntimeError as err:stop=str(err);break
        elapsed.append(1000*(time.perf_counter()-begin));params.append(z.tolist())
        stats.update(n=n,initialization_evaluations=init_evals);rounds.append(stats)
    return rom,dict(rounds=rounds,elapsed_ms=elapsed,parameters=params,stop=stop,rank=len(params))


def test_curve(C,Q,Z):
    """Compute all prefix errors without subtracting nearly equal squared norms.

The complete orthonormal frame has the same nested prefix spaces as Q.
Tail sums of squared coefficients remain accurate at the last greedy rounds.
"""
    d=len(C);N=Q.shape[1]
    U=np.linalg.qr(Q,mode='complete')[0] if N else np.eye(d)
    L=U.T@C;maximum=np.zeros(N+1)
    for start in range(0,len(Z),8192):
        values=Z[start:start+8192]@L.T
        tails=np.cumsum((values*values)[:,::-1],axis=1)[:,::-1]
        maximum=np.maximum(maximum,tails[:,:N+1].max(axis=0))
    errors=np.sqrt(maximum)
    # Independent direct residual check at the requested final dimension.
    target=min(d//2,N);E=C-Q[:,:target]@(Q[:,:target].T@C)
    direct=max(np.linalg.norm(Z[i:i+8192]@E.T,axis=1).max() for i in range(0,len(Z),8192))
    assert np.isclose(errors[target],direct,rtol=2e-7,atol=2e-14),(errors[target],direct)
    assert np.all(np.diff(errors)<=1e-12)
    return errors.tolist()


def validate():
    rng=np.random.default_rng(55);B=rng.normal(size=(5,5));H=B.T@B;z=rng.uniform(-1,1,5)
    g=2*H@z/np.linalg.norm(H);h=1e-6
    fd=np.array([((z+h*np.eye(5)[i])@H@(z+h*np.eye(5)[i])-(z-h*np.eye(5)[i])@H@(z-h*np.eye(5)[i]))/(2*h*np.linalg.norm(H)) for i in range(5)])
    assert np.allclose(g,fd,rtol=1e-7,atol=1e-8)
    results={}
    for method in METHODS:
        x,stats=optimize(np.diag([1.,.4,.1]),np.array([.2,-.3,.5]),method)
        assert stats['feasibility_defect']<1e-12
        assert abs(stats['value']-1.5/np.linalg.norm([1,.4,.1]))<2e-5
        results[method]=stats
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dimensions',nargs='+',type=int,default=[4,8,16,32,64,128]);parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1000,1008)));parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True);dump(OUTPUT/'validation.json',validate())
    config=dict(dimensions=args.dimensions,seeds=args.seeds,methods=METHODS,initializations=INITS,
                target='N=d/2; curves through d-1 for d=16,32,64',alpha=.6,T=80,pool_size=256,
                sqp='SciPy SLSQP, ftol=1e-9, maxiter=500',trust='SciPy trust-constr, exact Hessian, sparse bounds, gtol=1e-12, xtol=barrier_tol=1e-10, maxiter=500',
                reflective='Fides 0.7.8, Coleman-Li reflective trust region, exact Hessian, two-dimensional Newton/gradient subspace, single reflection, fatol=frtol=xtol=1e-10, gatol=1e-8, maxiter=500',
                scope='Python implementations of SQP and Coleman-Li reflective trust region; same baseline J rather than original paper objectives. trust-constr retained as supplementary interior-point comparison.',
                objective='-J_n=-z^T H_n z / ||H_n||_F',source=str(SOURCE),rho=.78,
                timing='Single CPU thread. Includes H construction, random/coarse-pool initialization, optimization and enrichment; excludes HF setup, test-error evaluation and file I/O; includes solver bookkeeping.',
                scipy=scipy.__version__,numpy=np.__version__,platform=platform.platform(),blas=threadpool_info())
    dump(OUTPUT/'config.json',config)
    records=json.loads((OUTPUT/'results.json').read_text()) if (OUTPUT/'results.json').exists() else []
    done={(r['d'],r['seed'],r['method'],r['initialization']) for r in records}
    for d in args.dimensions:
        model=np.load(SOURCE/f'model_d{d}.npz');C,K=model['C'],model['K'];Z=np.load(SOURCE/f'test_cube_d{d}.npz')['parameters']
        N=d-1 if d in [16,32,64] else d//2
        jobs=[(m,i) for m in args.methods for i in INITS];order=np.random.default_rng(20261115+d)
        for seed in args.seeds:
            for k in order.permutation(len(jobs)):
                method,init=jobs[k]
                if (d,seed,method,init) in done:continue
                rom,r=train(C,K,method,init,seed,N)
                r.update(d=d,seed=seed,method=method,initialization=init)
                r['errors']=test_curve(C,rom.Q,Z);r['test_max_error']=r['errors'][min(d//2,r['rank'])]
                r['wall_ms']=r['elapsed_ms'][min(d//2,r['rank'])-1]
                r['width']=float(.78**(d//2)*np.sqrt((1-.78**(2*(d-d//2)))/(1-.78**2)))
                r['ratio']=r['test_max_error']/r['width'];r['orthogonality_defect']=float(np.linalg.norm(rom.Q.T@rom.Q-np.eye(rom.Q.shape[1])))
                np.savez_compressed(OUTPUT/f'basis_d{d}_{method}_{init}_s{seed}.npz',Q=rom.Q)
                records.append(r);dump(OUTPUT/'results.json',records)
                print(f"DONE d={d} seed={seed} {method}/{init} error={r['test_max_error']:.6g} ms={r['wall_ms']:.2f} rank={r['rank']} solver_failures={sum(x['solver_success'] is False for x in r['rounds'])}",flush=True)
    print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
