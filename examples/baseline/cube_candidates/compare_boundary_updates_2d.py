"""Compare boundary rules on normalized-load 2D Poisson, on one CPU thread.

Run from the repository root with:
 .venv_cpu/bin/python -m examples.baseline.cube_candidates.compare_boundary_updates_2d
All budgets retrain the entire reduced space. Identical per-round starts are
used across methods/budgets. Validation is excluded from training timings.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/normalized_boundary_mpl')
from pathlib import Path
import json,time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from .normalized_load_2d import NormalizedLoad2D

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'runs/normalized_load_2d_20261105'
OUT=ROOT/'runs/normalized_load_2d_boundary_20261109'
METHODS=('clip','active_face','normal_scaled')
BUDGETS=(10,20,40,80,160,200,240,280,320,640)
SEEDS=tuple(range(1000,1004))
LABELS={'clip':'Clipping','active_face':'Active face','normal_scaled':'Adaptive normal updates'}

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def search(model,q,initial,method,steps,alpha=.6):
    h=model.matrix(q);ell=max(float(np.linalg.norm(h,'fro')),1e-30)
    z=np.asarray(initial,dtype=float).copy();best=z.copy();best_value=model.objective(h,z)
    clipped=blocked_steps=0;min_increment=0.;last_value=best_value
    for t in range(steps):
        _,g=model.objective(h,z,True);g/=ell
        if method=='active_face':
            blocked=((z>=1-1e-14)&(g>0))|((z<=-1+1e-14)&(g<0))
            blocked_steps+=int(np.any(blocked));g[blocked]=0
        eta=alpha/np.sqrt(1+np.sum(g*g))
        step=eta*g
        if method=='clip':
            clipped+=int(np.any(np.abs(z+step)>1));trial=np.clip(z+step,-1,1)
        elif method=='active_face':
            nz=step!=0
            fraction=min(1.,max(0.,float(np.min(np.where(step>0,1-z,1+z)[nz]/np.abs(step[nz]))))) if np.any(nz) else 1.
            clipped+=int(fraction<1);trial=np.clip(z+fraction*step,-1,1)
        elif method=='normal_scaled':
            distance=np.where(g>=0,1-z,1+z)
            trial=z+np.minimum(1,np.maximum(0,distance))*step
            assert np.max(np.abs(trial))<=1+2e-14
            trial=np.clip(trial,-1,1)
        else:raise ValueError(method)
        value=model.objective(h,trial)
        min_increment=min(min_increment,float(value-last_value))
        if value>best_value:best,best_value=trial.copy(),value
        z=trial;last_value=value
    return best,float(best_value),dict(updates=steps,boundary_truncations=clipped,
        outward_blocked_steps=blocked_steps,min_increment=min_increment,
        returned_boundary_coordinates=int(np.count_nonzero(np.abs(best)>=1-1e-12)))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    save('settings.json',dict(dimensions=[15,35,63],N='(d+1)/2 including center snapshot',
        budgets=BUDGETS,seeds=SEEDS,methods=METHODS,alpha=.6,source=str(SOURCE),
        initialization='seed+1000003; identical iid uniform start per greedy round for every method and budget',
        update='normalized objective gradient; active face renormalizes after removing outward components; adaptive normal uses original gradient normalization',
        returned_parameter='best objective visited, including initial point',
        test='original common augmented test set plus spectral witnesses and 16-start T320 clipping validation for EVERY new final space; union shared across all methods and budgets',
        target='median final test maximum error of clipping T320, reevaluated on the identical common union',
        timing='search and basis enrichment, excludes validation',device='CPU float64, one BLAS thread',
        interpretation='smallest tested budget meeting target; not a guaranteed minimal budget or equal-wall-time comparison'))
    cached_rows=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    cached={(r['d'],r['seed'],r['method'],r['T']):r for r in cached_rows}
    allrows=[];checks=json.loads((OUT/'reproduction_checks.json').read_text()) if (OUT/'reproduction_checks.json').exists() else []
    for m in (4,6,8):
        model=NormalizedLoad2D(m);d=model.d;N=(d+1)//2
        local=[];extras=[];starts=2*qmc.Sobol(d,scramble=True,seed=202611090+d).random_base2(4)-1
        for seed in SEEDS:
            for method in METHODS:
                for T in BUDGETS:
                    key=(d,seed,method,T)
                    if key in cached:
                        q=np.load(OUT/f'd{d}_{method}_T{T}_s{seed}.npz')['Q']
                        local.append((cached[key],q))
                        continue
                    rng=np.random.default_rng(seed+1000003);q=model.initial_basis()
                    selected=[];history=[];stats=[]
                    begin=time.perf_counter()
                    for n in range(1,N):
                        z,value,counts=search(model,q,rng.uniform(-1,1,d),method,T)
                        selected.append(z);history.append(value);stats.append(counts);q=model.append(q,z)
                    elapsed=time.perf_counter()-begin
                    lo,hi,witness=model.error_bounds(q)
                    row=dict(d=d,N=N,seed=seed,method=method,T=T,seconds=elapsed,
                        total_updates=(N-1)*T,E_spectral_lower=float(lo),E_spectral_upper=float(hi),
                        boundary_truncations=sum(x['boundary_truncations'] for x in stats),
                        outward_blocked_steps=sum(x['outward_blocked_steps'] for x in stats),
                        mean_returned_boundary_coordinates=float(np.mean([x['returned_boundary_coordinates'] for x in stats])),
                        minimum_J_increment=min(x['min_increment'] for x in stats),
                        orthogonality=float(np.linalg.norm(q.T@q-np.eye(N))))
                    assert row['orthogonality']<1e-12
                    if method=='clip' and T in (80,320):
                        old=np.load(SOURCE/f'd{d}_gd{T}_s{seed}.npz')
                        difference=float(np.max(np.abs(np.asarray(selected)-old['selected_parameters'])))
                        checks.append(dict(d=d,T=T,seed=seed,max_parameter_difference=difference))
                        assert difference<1e-8
                    np.savez_compressed(OUT/f'd{d}_{method}_T{T}_s{seed}.npz',Q=q,selected_parameters=selected,selected_squared_errors=history)
                    extras.extend([witness[None,:],model.search(q,starts,alpha=.6,steps=320)[0]])
                    local.append((row,q))
            print(f'd={d}, seed={seed}: all three methods and requested budgets completed',flush=True)
        original_count=len(np.load(SOURCE/f'test_d{d}.npz')['parameters'])
        oldtest=np.load((OUT if (OUT/f'test_d{d}.npz').exists() else SOURCE)/f'test_d{d}.npz')['parameters'];test=np.vstack((oldtest,*extras))
        np.savez_compressed(OUT/f'test_d{d}.npz',parameters=test,original_count=original_count)
        solutions=model.solution(test);norms=np.sum(solutions**2,axis=1)
        for row,q in local:
            errors=np.sqrt(np.maximum(norms[:,None]-np.cumsum((solutions@q)**2,axis=1),0))
            row['E_test']=float(errors[:,-1].max());row['E_original_test']=float(errors[:original_count,-1].max())
            row['test_error_by_N']=errors.max(axis=0).tolist();row['test_count']=len(test)
            assert row['E_test']<=row['E_spectral_upper']*(1+1e-10)
            allrows.append(row)
        save('results.json',allrows);save('reproduction_checks.json',checks)
        print(f'd={d}: common validation on {len(test)} points completed',flush=True)
    summary=[];targets=[]
    for d in (15,35,63):
        for method in METHODS:
            for T in BUDGETS:
                rr=[r for r in allrows if (r['d'],r['method'],r['T'])==(d,method,T)]
                summary.append(dict(d=d,method=method,T=T,N=rr[0]['N'],
                    E_test=float(np.median([r['E_test'] for r in rr])),
                    E_min=min(r['E_test'] for r in rr),E_max=max(r['E_test'] for r in rr),
                    seconds=float(np.median([r['seconds'] for r in rr])),
                    E_upper=float(np.median([r['E_spectral_upper'] for r in rr])),
                    mean_boundary_truncations=float(np.mean([r['boundary_truncations'] for r in rr])),
                    mean_outward_blocked_steps=float(np.mean([r['outward_blocked_steps'] for r in rr]))))
        target=next(r['E_test'] for r in summary if (r['d'],r['method'],r['T'])==(d,'clip',320))
        for method in METHODS:
            passing=[r['T'] for r in summary if r['d']==d and r['method']==method and r['E_test']<=target*(1+1e-12)]
            paired=[]
            for seed in SEEDS:
                ref=next(r['E_test'] for r in allrows if (r['d'],r['method'],r['T'],r['seed'])==(d,'clip',320,seed))
                ps=[r['T'] for r in allrows if (r['d'],r['method'],r['seed'])==(d,method,seed) and r['E_test']<=ref*(1+1e-12)]
                paired.append(dict(seed=seed,reference=ref,first_tested_budget=min(ps) if ps else None))
            targets.append(dict(d=d,method=method,target=target,first_tested_budget=min(passing) if passing else None,paired=paired))
    save('summary.json',summary);save('target_budgets.json',targets)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for d,ax in zip((15,35,63),axes):
        for method,col,mk in zip(METHODS,('#245E91','#BC762C','#477A68'),('o','^','s')):
            rr=[r for r in summary if r['d']==d and r['method']==method]
            ax.plot(BUDGETS,[r['E_test'] for r in rr],color=col,marker=mk,label=LABELS[method])
        target=next(r['target'] for r in targets if r['d']==d)
        ax.axhline(target,color='.45',ls=':',lw=1,label='Clipping, T=320 reference')
        ax.set_xscale('log',base=2);ax.set_xticks((10,20,40,80,160,320,640),labels=(10,20,40,80,160,320,640),rotation=45)
        ax.set(title=f'd = {d}',xlabel='GD updates per greedy round T');ax.grid(alpha=.18)
    axes[0].set_ylabel('Final test maximum error')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(OUT/'error_vs_budget.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('COMPLETE',flush=True)



def fixed_space_check():
    """Paired optimization trajectories in identical fixed reduced spaces.

    References are prefixes of the previously saved clipping-T320 run (seed
    1000), not spaces retrained by each tested boundary method. For each of
    32 common starts, the target is that start's best clipping objective at
    step 320, with relative tolerance 1e-8. Censored times are kept explicit.
    """
    rows=[]
    for m in (4,6,8):
        model=NormalizedLoad2D(m);d=model.d;N=(d+1)//2
        qfull=np.load(SOURCE/f'd{d}_gd320_s1000.npz')['Q']
        initial=np.random.default_rng(2026110900+d).uniform(-1,1,(32,d))
        for n in (1,N//2,N-1):
            q=qfull[:,:n];h=model.matrix(q);ell=max(float(np.linalg.norm(h,'fro')),1e-30)
            paths={}
            for method in METHODS:
                z=initial.copy();best=model.objective(h,z);values=[best.copy()]
                for t in range(640):
                    _,g=model.objective(h,z,True);g/=ell
                    if method=='active_face':
                        blocked=((z>=1-1e-14)&(g>0))|((z<=-1+1e-14)&(g<0));g[blocked]=0
                    eta=.6/np.sqrt(1+np.sum(g*g,axis=1,keepdims=True));step=eta*g
                    if method=='clip':z=np.clip(z+step,-1,1)
                    elif method=='active_face':
                        distances=np.where(step>0,1-z,1+z)
                        ratios=np.full_like(step,np.inf);np.divide(distances,np.abs(step),out=ratios,where=step!=0)
                        fraction=np.clip(np.min(ratios,axis=1,keepdims=True),0,1)
                        z=np.clip(z+fraction*step,-1,1)
                    else:
                        z=z+np.minimum(1,np.maximum(0,np.where(g>=0,1-z,1+z)))*step
                        assert np.max(np.abs(z))<=1+2e-14
                        z=np.clip(z,-1,1)
                    best=np.maximum(best,model.objective(h,z));values.append(best.copy())
                paths[method]=np.asarray(values)
            target=paths['clip'][320]
            for method in METHODS:
                reached=paths[method]>=target[None,:]*(1-1e-8)
                hits=[int(np.argmax(reached[:,j])) if np.any(reached[:,j]) else None for j in range(32)]
                observed=[x for x in hits if x is not None]
                rows.append(dict(d=d,n=n,method=method,initial_count=32,
                    target='per-start clipping best objective at T320, relative tolerance 1e-8',
                    hit_times=hits,reached_by_320=sum(x is not None and x<=320 for x in hits),
                    reached_by_640=len(observed),median_hit_time=float(np.median([641 if x is None else x for x in hits])),
                    median_objective_ratio_at_80=float(np.median(paths[method][80]/target)),
                    median_objective_ratio_at_320=float(np.median(paths[method][320]/target))))
            np.savez_compressed(OUT/f'fixed_space_d{d}_n{n}.npz',initial=initial,target=target,**paths)
    save('fixed_space_comparison.json',rows)
    print('Fixed-space comparison complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
        fixed_space_check()
