"""Equal outer-iteration budget for GD, SQP and reflective trust region.

All three receive at most 80 updates per greedy round. Natural early stopping
is retained and logged, without artificial idle iterations. Every optimizer
uses its own line search/subproblem work within an outer iteration; wall time
and objective evaluations are therefore reported separately.
"""
from .compare_optimizers import *
from . import compare_optimizers as base
OUT=ROOT/'runs/cube_optimizer_equal_steps_20261115'
PRIMARY=('gd','sqp','reflective')
DIMS=[4,8,16,32,64,128]
STEPS=80


def validate_budget():
    rng=np.random.default_rng(17);B=rng.normal(size=(8,8));H=B.T@B;initial=rng.uniform(-1,1,8)
    checks={}
    for method in PRIMARY:
        for cap in [1,5,80]:
            z,s=optimize(H,initial,method,max_steps=cap)
            assert s['nit']<=cap,(method,cap,s['nit'])
            assert s['feasibility_defect']<1e-12
            assert s['value']>=s['initial_value']-1e-12
            if method=='gd':
                expected,_=gd_search(H,initial,alpha=.6,T=cap,diagnostics=False)
                assert np.array_equal(z,expected)
            checks[f'{method}_{cap}']=s
    return checks


def main():
    OUT.mkdir(exist_ok=True);dump(OUT/'validation.json',validate_budget())
    config=json.loads((base.OUTPUT/'config.json').read_text())
    config.update(methods=PRIMARY,equal_step_budget=STEPS,T=STEPS,
                  sqp='SciPy SLSQP, ftol=1e-9, maxiter=80',
                  reflective='Fides 0.7.8, Coleman-Li reflective trust region, exact Hessian, two-dimensional Newton/gradient subspace, single reflection, fatol=frtol=xtol=1e-10, gatol=1e-8, maxiter=80',
                  scope='Equal outer-iteration budgets for three Python optimizers; no trust-constr comparator.',
                  stopping='Natural convergence/failure stops are retained and actual iteration counts recorded. Reaching 80 is budget exhaustion, not a failed experiment.',
                  fairness='Same initial RNG streams, model, tests and CPU thread count. Equal outer iteration budgets; line searches and subproblem costs are included in wall time.',
                  blas=threadpool_info())
    config.pop('trust',None);dump(OUT/'config.json',config)
    records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    done={(r['d'],r['seed'],r['method'],r['initialization']) for r in records}
    for d in DIMS:
        a=np.load(SOURCE/f'model_d{d}.npz');C,K=a['C'],a['K'];Z=np.load(SOURCE/f'test_cube_d{d}.npz')['parameters']
        N=d-1 if d in [16,32,64] else d//2
        jobs=[(m,i) for m in PRIMARY for i in INITS];order=np.random.default_rng(20261115+d)
        for seed in range(1000,1008):
            for j in order.permutation(len(jobs)):
                method,init=jobs[j]
                if (d,seed,method,init) in done:continue
                rom,r=train(C,K,method,init,seed,N,max_steps=STEPS)
                assert all(s['nit']<=STEPS for s in r['rounds'])
                r.update(d=d,seed=seed,method=method,initialization=init,step_budget=STEPS)
                r['errors']=test_curve(C,rom.Q,Z);r['test_max_error']=r['errors'][min(d//2,r['rank'])]
                r['wall_ms']=r['elapsed_ms'][min(d//2,r['rank'])-1]
                r['width']=float(.78**(d//2)*np.sqrt((1-.78**(2*(d-d//2)))/(1-.78**2)))
                r['ratio']=r['test_max_error']/r['width'];r['orthogonality_defect']=float(np.linalg.norm(rom.Q.T@rom.Q-np.eye(rom.Q.shape[1])))
                np.savez_compressed(OUT/f'basis_d{d}_{method}_{init}_s{seed}.npz',Q=rom.Q)
                records.append(r);dump(OUT/'results.json',records)
                print(f"DONE d={d} seed={seed} {method}/{init} error={r['test_max_error']:.6g} ms={r['wall_ms']:.2f} rank={r['rank']}",flush=True)
    print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
