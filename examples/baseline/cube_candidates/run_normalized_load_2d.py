"""Reproducible CPU comparison for normalized-load square Poisson.

All methods use exact energy coordinates, identical initial spaces, and a
common validation set. Validation searches and error bounds are outside
training timing. Empirical maxima are not asserted to be exact maxima.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/normalized_load_2d_mpl')
from pathlib import Path
import json
import time
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .normalized_load_2d import NormalizedLoad2D

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'runs/normalized_load_2d_20261105'
METHODS = ['gd80','gd320','pool256','pool4096','pool10000']


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = dict(m_values=[4,6,8], dimensions=[15,35,63], seeds=list(range(1000,1004)),
        methods=METHODS, alpha=.6, updates=[80,320], N='(d+1)/2 including center snapshot',
        pool='iid uniform cube, fixed and reused within each greedy run',
        gd='one iid uniform cube start per greedy round; best iterate retained',
        norm='absolute H1_0 energy error, exact analytic solution coordinates',
        normalizer='Frobenius norm of residual Gram matrix',
        tests='16384 scrambled Sobol cube points, +/- coordinate axes, center; augmented with spectral witnesses and 64-start T320 search on every final space, shared by all methods and seeds',
        device='CPU float64, one BLAS thread', timing='pool generation/solution encoding, search and basis enrichment; excludes validation',
        width='No n-width claimed')
    save('settings.json', settings)
    rows = []
    for m in settings['m_values']:
        model = NormalizedLoad2D(m); d = model.d; N = (d+1)//2
        q = model.initial_basis(); rng = np.random.default_rng(18)
        q = model.append(q, rng.uniform(-1,1,d)); h = model.matrix(q)
        z = rng.uniform(-.7,.7,d); value, grad = model.objective(h,z,True)
        numeric = np.array([(model.objective(h,z+1e-6*e)-model.objective(h,z-1e-6*e))/2e-6 for e in np.eye(d)])
        gradient_error = float(np.linalg.norm(numeric-grad)/np.linalg.norm(grad))
        assert gradient_error < 1e-7
        w = np.r_[1,z]; assert np.isclose(np.linalg.norm(w/np.linalg.norm(w)),1)
        qnew = model.append(q,z); hz = h[1:,1:]@z
        rankone = h[1:,1:]-np.outer(hz,hz)/(z@hz)
        assert np.allclose(model.matrix(qnew)[1:,1:],rankone,atol=1e-14)
        np.savez_compressed(OUT/f'model_d{d}.npz', modes=model.modes, eigenvalues=model.eigenvalues, weights=model.weights)
        test = np.vstack((2*qmc.Sobol(d,scramble=True,seed=202611401+d).random_base2(14)-1,
                          np.eye(d),-np.eye(d),np.zeros((1,d))))
        starts = 2*qmc.Sobol(d,scramble=True,seed=202611402+d).random_base2(6)-1
        local = []; extras = []
        for seed in settings['seeds']:
            for method in METHODS:
                start = time.perf_counter(); rng = np.random.default_rng(seed+1000003)
                q = model.initial_basis(); selected = []; history = []
                if method.startswith('pool'):
                    pool = rng.uniform(-1,1,(int(method[4:]),d)); solutions = model.solution(pool)
                    pool_values = np.sum(solutions[:,1:]**2,axis=1)
                for n in range(1,N):
                    if method.startswith('gd'):
                        z, val = model.search(q,rng.uniform(-1,1,d),alpha=.6,steps=int(method[2:]))
                    else:
                        idx = int(np.argmax(pool_values)); z = pool[idx].copy(); val = pool_values[idx]
                    selected.append(z.copy()); history.append(float(val)); q = model.append(q,z)
                    if method.startswith('pool'):
                        pool_values = np.maximum(pool_values-(solutions@q[:,-1])**2,0)
                elapsed = time.perf_counter()-start
                lo,hi,witness = model.error_bounds(q)
                extras.extend([witness[None,:],model.search(q,starts,alpha=.6,steps=320)[0]])
                local.append((dict(d=d,m=m,N=N,seed=seed,method=method,seconds=elapsed,
                    E_spectral_lower=float(lo),E_spectral_upper=float(hi),gradient_check=gradient_error),q))
                np.savez_compressed(OUT/f'd{d}_{method}_s{seed}.npz', Q=q, selected_parameters=selected,selected_squared_errors=history)
                print(f'd={d} seed={seed} {method}: train {elapsed:.3f}s, spectral error [{lo:.5f}, {hi:.5f}]',flush=True)
        test_all = np.vstack((test,*extras)); solutions = model.solution(test_all)
        np.savez_compressed(OUT/f'test_d{d}.npz',parameters=test_all,base_count=len(test))
        norms = np.sum(solutions**2,axis=1)
        for row,q in local:
            squared = np.maximum(norms[:,None]-np.cumsum((solutions@q)**2,axis=1),0)
            curve = np.sqrt(squared.max(axis=0))
            row['E_test_base'] = float(np.sqrt(squared[:len(test),-1].max()))
            row['E_test'] = float(curve[-1]); row['test_error_by_N'] = curve.tolist()
            row['test_count'] = len(test_all)
            row['orthogonality'] = float(np.linalg.norm(q.T@q-np.eye(N)))
            assert row['E_test'] <= row['E_spectral_upper']*(1+1e-9)
            rows.append(row)
        save('results.json',rows)
        print(f'd={d} complete; shared test set {len(test_all)}',flush=True)
    summary=[]
    for d in settings['dimensions']:
        for method in METHODS:
            rr=[r for r in rows if r['d']==d and r['method']==method]
            summary.append(dict(d=d,N=rr[0]['N'],method=method,
                E_test=float(np.median([r['E_test'] for r in rr])),
                E_min=min(r['E_test'] for r in rr),E_max=max(r['E_test'] for r in rr),
                E_base=float(np.median([r['E_test_base'] for r in rr])),
                E_upper=float(np.median([r['E_spectral_upper'] for r in rr])),
                seconds=float(np.median([r['seconds'] for r in rr]))))
    save('summary.json',summary)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    styles=[('#245E91','--'),('#245E91','-'),('#BC762C',':'),('#477A68','-.'),('#B44E49','--')]
    for d,ax in zip(settings['dimensions'],axes):
        for method,(col,style) in zip(METHODS,styles):
            rr=[r for r in rows if r['d']==d and r['method']==method]
            curve=np.median([r['test_error_by_N'] for r in rr],axis=0)
            label=('GD, T='+method[2:]) if method.startswith('gd') else ('Pool '+method[4:])
            ax.semilogy(range(1,len(curve)+1),curve,color=col,ls=style,label=label,lw=2)
        ax.set(title=f'd = {d}',xlabel='Reduced dimension N'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Maximum energy error on shared test set')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1)); fig.savefig(OUT/'test_error_vs_N.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
