"""Separate incomplete inner GD convergence from snapshot enrichment quality.

Retain all old paths and retrain GD with the same per-round starts until a
projected fixed point. Evaluate all paths on one enlarged common test set.
Also compare GD80, converged GD and pool10000 from each identical V_n on
the newly trained path. No global optimality of inner GD is assumed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_forcing_2d_matplotlib')
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dimension_study import rectangular_model, PARTITIONS, SEEDS
from run import ascend, append, residual_gram, extrema
from enrichment_study import aggregate

BASE = Path(__file__).resolve().parents[4] / 'runs/piecewise_forcing_2d_20261112'
OLD = BASE / 'enrichment_study'
OUT = BASE / 'convergence_study'
METHODS = ['gd80', 'gd_converged', 'pool256', 'pool4096', 'pool10000']
MAX_STEPS = 100000
UPDATE_TOL = 1e-12
PROJECTED_GRADIENT_TOL = 1e-10


def converge(H, start):
    ell = max(float(np.linalg.norm(H)), 1e-30)
    z = start.copy()
    z80 = None
    consecutive_small = 0
    previous = float(z @ H @ z)
    smallest_change = 0.
    for t in range(1, MAX_STEPS + 1):
        g = 2 * H @ z / ell
        new = np.clip(z + .6 * g / np.sqrt(1 + g @ g), -1, 1)
        delta = float(np.max(np.abs(new-z)))
        z = new
        J = float(z @ H @ z)
        smallest_change = min(smallest_change, J-previous)
        previous = J
        if t == 80:
            z80 = z.copy()
        consecutive_small = consecutive_small + 1 if delta <= UPDATE_TOL else 0
        if delta == 0 or consecutive_small >= 10:
            g = 2 * H @ z / ell
            projected = float(np.max(np.abs(np.clip(z+g, -1, 1)-z)))
            if projected <= PROJECTED_GRADIENT_TOL:
                break
    else:
        raise RuntimeError(f'No convergence in {MAX_STEPS} steps')
    if z80 is None:
        z80 = ascend(H, start, 80)
    assert smallest_change >= -1e-12 * ell
    # Verify stability by continuing beyond the reported stopping point.
    continued = z.copy()
    for _ in range(100):
        g = 2 * H @ continued / ell
        continued = np.clip(continued + .6*g/np.sqrt(1+g@g), -1, 1)
    drift = float(np.max(np.abs(continued-z)))
    assert drift < 1e-8
    g80 = 2 * H @ z80 / ell
    stats = dict(steps=t, exact_fixed_point=delta == 0, last_update=delta,
                 projected_gradient_residual=projected, extra_100_steps_drift=drift,
                 projected_gradient_residual_at_80=float(np.max(np.abs(np.clip(z80+g80,-1,1)-z80))),
                 J80=float(z80@H@z80), J_converged=float(z@H@z),
                 saturated_components=int(np.sum(np.abs(z)==1)), smallest_J_increment=smallest_change)
    return z, z80, stats


def final_direction(B, Q):
    w = np.linalg.qr(Q, mode='complete')[0][:, -1]
    return B.T @ w


def study(mx, my):
    tick = time.perf_counter()
    d = mx*my
    B, G, checks = rectangular_model(mx, my)
    E0 = float(np.sqrt(np.ones(d)@G@np.ones(d)))
    old = np.load(OLD/f'd{d}_snapshots.npz')
    spaces, snapshots, training, alternative_directions = {}, {}, [], {}
    exact_corners = []
    for seed in SEEDS:
        for method in METHODS:
            if method == 'gd_converged':
                continue
            Z = old[f'z_s{seed}_{method}']
            Q = np.zeros((d,0))
            for z in Z:
                Q, _ = append(B,Q,z)
            spaces[(seed,method)] = Q
        pool = np.random.default_rng(seed).uniform(-1,1,(10000,d))
        pool_u = pool@B.T
        scores = np.sum(pool_u**2,axis=1)
        init = np.random.default_rng(seed+10000)
        Q = np.zeros((d,0))
        selected, z80s, zps, rows, q80s, qps = [], [], [], [], [], []
        for n in range(d-1):
            H = residual_gram(B,G,Q)
            z, z80, stats = converge(H,init.uniform(-1,1,d))
            zp = pool[np.argmax(scores)].copy()
            assert stats['J_converged'] >= stats['J80'] - 1e-14
            Q80,q80 = append(B,Q,z80)
            Qp,qp = append(B,Q,zp)
            if n == d-2:
                for trial in [Q80,Qp]:
                    h=final_direction(B,trial)
                    exact_corners.extend([np.where(h>=0,1.,-1.)[None,:], np.where(h>=0,-1.,1.)[None,:]])
            Q,q = append(B,Q,z)
            scores -= (pool_u@q)**2
            selected.append(z); z80s.append(z80); zps.append(zp)
            q80s.append(q80); qps.append(qp)
            rows.append(dict(N=n+1,J_pool=float(zp@H@zp),**stats))
        spaces[(seed,'gd_converged')] = Q
        snapshots[f'z_s{seed}_gd_converged'] = np.asarray(selected)
        snapshots[f'z_s{seed}_same_space_gd80'] = np.asarray(z80s)
        snapshots[f'z_s{seed}_same_space_pool10000'] = np.asarray(zps)
        alternative_directions[seed] = {'gd80':np.asarray(q80s).T,'pool10000':np.asarray(qps).T}
        training.append(dict(seed=seed,rounds=rows))
        print(json.dumps(dict(d=d,seed=seed,median_steps=float(np.median([r['steps'] for r in rows])),
                             max_steps=max(r['steps'] for r in rows))),flush=True)
    np.savez_compressed(OUT/f'd{d}_snapshots.npz',**snapshots)

    tests=[np.load(OLD/f'd{d}_test.npz')['parameters']]+exact_corners
    rng=np.random.default_rng(292000+d)
    if d>16:
        for seed in SEEDS:
            Q=spaces[(seed,'gd_converged')]
            for n in range(1,d):
                R=B-Q[:,:n]@(Q[:,:n].T@B)
                tests.append(extrema(R.T@R,rng.uniform(-1,1,(128,d)),steps=160))
    # Include candidate-enriched worst-point searches, with equal effort for
    # GD80 and pool alternatives, in the same shared test set.
    if d>16:
        for seed in SEEDS:
            Q=spaces[(seed,'gd_converged')]
            for n in range(d-1):
                R=B-Q[:,:n]@(Q[:,:n].T@B)
                for directions in alternative_directions[seed].values():
                    q=directions[:,n]
                    residual=R-q[:,None]*(q@B)[None,:]
                    tests.append(extrema(residual.T@residual,rng.uniform(-1,1,(64,d)),steps=160))
    for Q in spaces.values():
        h=final_direction(B,Q)
        corner=np.where(h>=0,1.,-1.)
        tests.extend([corner[None,:],-corner[None,:]])
    test=np.unique(np.vstack(tests),axis=0)
    np.savez_compressed(OUT/f'd{d}_test.npz',parameters=test)
    U=test@B.T
    unorm=np.sum(U**2,axis=1)
    curves=[]
    for (seed,method),Q in spaces.items():
        coeff=U@Q
        squared=np.maximum(0.,unorm[:,None]-np.cumsum(coeff**2,axis=1))
        curve=np.r_[1.,np.sqrt(squared.max(axis=0))/E0]
        exact=np.sum(np.abs(final_direction(B,Q)))/E0
        assert abs(curve[-1]-exact)/exact<1e-10
        curve[-1]=exact
        assert np.max(np.diff(curve))<1e-11
        drops=curve[:-1]-curve[1:]
        curves.append(dict(seed=seed,method=method,error_over_E0=curve.tolist(),
                           drop_over_E0=drops.tolist(),relative_drop=(drops/curve[:-1]).tolist()))
    same_space=[]
    for seed in SEEDS:
        Q=spaces[(seed,'gd_converged')]
        cumulative=np.cumsum((U@Q)**2,axis=1)
        before_squared=np.maximum(0.,unorm[:,None]-np.column_stack([np.zeros(len(test)),cumulative[:,:-1]]))
        before=np.sqrt(before_squared.max(axis=0))/E0
        after={}
        for method,directions in alternative_directions[seed].items():
            squared=np.maximum(0.,before_squared-(U@directions)**2)
            after[method]=np.sqrt(squared.max(axis=0))/E0
        after['gd_converged']=np.asarray(next(c for c in curves if c['seed']==seed and c['method']=='gd_converged')['error_over_E0'][1:])
        rows=next(r['rounds'] for r in training if r['seed']==seed)
        for n in range(d-1):
            same_space.append(dict(seed=seed,N=n+1,before_error_over_E0=float(before[n]),
                after_error_over_E0={m:float(v[n]) for m,v in after.items()},
                drop_over_E0={m:float(before[n]-v[n]) for m,v in after.items()},
                J={'gd80':rows[n]['J80'],'gd_converged':rows[n]['J_converged'],'pool10000':rows[n]['J_pool']}))
    summary={m:{key:aggregate([c[key] for c in curves if c['method']==m])
                for key in ['error_over_E0','drop_over_E0','relative_drop']} for m in METHODS}
    steps=[r['steps'] for tr in training for r in tr['rounds']]
    comparison={}
    for other in ['gd80','pool10000']:
        differences=np.array([r['after_error_over_E0'][other]-r['after_error_over_E0']['gd_converged'] for r in same_space])
        comparison[other]=dict(lower_after_error=int(np.sum(differences>1e-10)),
                               equal_after_error=int(np.sum(np.abs(differences)<=1e-10)),
                               higher_after_error=int(np.sum(differences< -1e-10)),
                               higher_J=int(sum(r['J']['gd_converged']>r['J'][other]+1e-14 for r in same_space)))
    report=dict(settings=dict(d=d,partition=[mx,my],N=d-1,alpha=.6,seeds=SEEDS,
                initial_global_maximum=E0,test_count=len(test),update_tolerance=UPDATE_TOL,
                projected_gradient_tolerance=PROJECTED_GRADIENT_TOL,step_cap=MAX_STEPS,
                interpretation='Inner convergence means a projected fixed point, not a global maximizer. Same-space comparisons use the converged-GD greedy path.',
                all_ranks_global=d==16,final_rank_global=True),checks=checks,training=training,
                curves=curves,summary=summary,same_space=same_space,same_space_comparison=comparison,
                convergence=dict(rounds=len(steps),median_steps=float(np.median(steps)),max_steps=max(steps),
                  exact_fixed_points=sum(r['exact_fixed_point'] for tr in training for r in tr['rounds']),
                  more_than_80_steps=sum(t>80 for t in steps),
                  max_extra_100_steps_drift=max(r['extra_100_steps_drift'] for tr in training for r in tr['rounds'])),
                elapsed_seconds=time.perf_counter()-tick)
    (OUT/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(d=d,final={m:summary[m]['error_over_E0']['median'][-1] for m in METHODS},
                         convergence=report['convergence'],same_space=comparison,seconds=report['elapsed_seconds'])),flush=True)
    return report


def plots(reports):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for name,key,ylabel in [('error_vs_N','error_over_E0',r'$E_N/E_0$'),
                           ('drop_per_basis','drop_over_E0',r'$(E_{N-1}-E_N)/E_0$')]:
        fig,axes=plt.subplots(2,3,figsize=(12.8,7.4))
        for ax,r in zip(axes.flat,reports):
            d=r['settings']['d']
            n=np.arange(d) if key=='error_over_E0' else np.arange(1,d)
            for m,color,style,label in [('gd80','#666666',':','GD, T=80'),
                ('gd_converged','#0072B2','-','GD, until convergence'),
                ('pool10000','#D55E00','--','Pool 10000')]:
                ax.plot(n,r['summary'][m][key]['median'],color=color,ls=style,lw=2,label=label)
            ax.set_title(f'd = {d}');ax.set_xlabel('Reduced dimension N');ax.set_ylabel(ylabel)
            ax.set_yscale('log');ax.set_xlim(n[0],n[-1])
            ax.set_xticks(np.unique(np.linspace(n[0],n[-1],5).round().astype(int)))
            ax.grid(alpha=.15,which='both')
        axes[1,2].axis('off')
        handles,labels=axes[0,0].get_legend_handles_labels()
        axes[1,2].legend(handles,labels,loc='upper left',frameon=False)
        axes[1,2].text(.04,.56,'Median over 4 seeds\nSame per-round initialization sequence\nEach method follows its own greedy path\n\nShared test set for each d\nExact final-rank maximum\nConvergence does not imply global optimality',
                      transform=axes[1,2].transAxes,va='top',fontsize=10,linespacing=1.6)
        fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=180);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)


def audit_d16():
    """Independently enumerate all vertices for two illustrative fixed spaces."""
    B, _, _ = rectangular_model(4,4)
    saved = np.load(OUT/'d16_snapshots.npz')
    report = json.loads((OUT/'d16.json').read_text())
    E0 = report['settings']['initial_global_maximum']
    bits = (np.arange(2**16,dtype=np.uint32)[:,None] >> np.arange(16)) & 1
    vertices = 2*bits.astype(float)-1
    results=[]
    for seed,n in [(1001,10),(1002,11)]:
        prefix=saved[f'z_s{seed}_gd_converged'][:n]
        Q=np.linalg.qr(B@prefix.T)[0]
        R=B-Q@(Q.T@B)
        H=R.T@R
        maximum=float(np.sum((vertices@R.T)**2,axis=1).max())
        reference=next(r for r in report['same_space'] if r['seed']==seed and r['N']==n+1)
        candidates={}
        for method,key in [('gd80',f'z_s{seed}_same_space_gd80'),
                           ('gd_converged',f'z_s{seed}_gd_converged'),
                           ('pool10000',f'z_s{seed}_same_space_pool10000')]:
            z=saved[key][n]
            q=R@z;q/=np.linalg.norm(q)
            residual=R-q[:,None]*(q@R)[None,:]
            after=float(np.sqrt(np.sum((vertices@residual.T)**2,axis=1).max())/E0)
            assert abs(after-reference['after_error_over_E0'][method])<1e-11
            candidates[method]=dict(J=float(z@H@z),J_over_global_maximum=float(z@H@z)/maximum,
                                    exact_after_error_over_E0=after)
        results.append(dict(seed=seed,N_before=n,exact_before_error_over_E0=np.sqrt(maximum)/E0,
                            candidates=candidates))
    (OUT/'d16_exact_checks.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[study(mx,my) for mx,my in PARTITIONS]
    audit_d16()
    plots(reports)
    (OUT/'summary.json').write_text(json.dumps([{k:r[k] for k in ['settings','summary','convergence','same_space_comparison']} for r in reports],indent=2)+'\n')
    print('COMPLETE',flush=True)
