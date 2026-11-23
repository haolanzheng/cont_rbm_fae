"""Extend the GD80/pool experiments through N=d-1.

Compute errors using orthogonal-complement coordinates to avoid cancellation
near full rank. Check every existing error-curve prefix and GD snapshot prefix.
Fits describe finite observed intervals; they are not asymptotic claims.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, csv, hashlib, json, platform
from pathlib import Path
import numpy as np
from training import train_method
from run import model

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/piecewise_forcing_2d_20261112/slides_summary'
DIMS = [4,16] + [m*m for m in range(5,17)]
SEEDS = list(range(1000,1004))
GRIDS = ['grid1','grid0.5','grid0.25']

METHODS=['gd80','pool256','pool4096','pool10000']

def fit(curve,lo,hi):
    n=np.arange(lo,hi+1);v=np.log(np.asarray(curve)[n])
    if len(n)<4:return None
    p=np.polyfit(np.log(n),v,1);e=np.polyfit(n,v,1)
    return dict(N_start=int(lo),N_end=int(hi),power_exponent=float(-p[0]),
        exponential_rate=float(-e[0]),power_log_rmse=float(np.sqrt(np.mean((v-np.polyval(p,np.log(n)))**2))),
        exponential_log_rmse=float(np.sqrt(np.mean((v-np.polyval(e,n))**2))))

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('d*.json')):raise FileExistsError('Preserve previous runs; use a new directory.')
    old=json.loads((BASE/'shared_test_errors.json').read_text());reports=[]
    for d in DIMS:
        B,G,checks=model(round(np.sqrt(d)));test_file=BASE/f'd{d}_test.npz'
        test=np.load(test_file)['parameters'];U=test@B.T
        gdref=np.load(BASE/'full_rank_20261116'/f'd{d}_snapshots.npz')
        methods=METHODS+(GRIDS if d==4 else []);rows=[];snapshots={}
        for method in methods:
            for seed in ([1000] if method.startswith('grid') else SEEDS):
                Q,Z,timing=train_method(B,G,seed,'gd' if method=='gd80' else method,80,d-1)
                assert np.isfinite(Q).all() and np.isfinite(Z).all()
                orth=float(np.linalg.norm(Q.T@Q-np.eye(d-1)));assert orth<1e-10
                # The complete QR preserves every prefix span and supplies its
                # orthogonal complement. No subtraction of nearly equal norms.
                full,_=np.linalg.qr(Q,mode='complete');coeff=U@full
                tails=np.cumsum((coeff**2)[:,::-1],axis=1)[:,::-1]
                curve=np.sqrt(tails.max(axis=0))
                assert len(curve)==d and np.all(np.diff(curve)<=1e-13)
                validation=dict(orthogonality_defect=orth)
                direct_errors=[]
                for n in sorted(set([d//2,3*d//4,d-1])):
                    value=float(np.linalg.norm(test@(B-Q[:,:n]@(Q[:,:n].T@B)).T,axis=1).max())
                    np.testing.assert_allclose(curve[n],value,atol=1e-13,rtol=1e-10)
                    direct_errors.append(dict(N=n,absolute_difference=float(abs(value-curve[n]))))
                validation['direct_projection_checks']=direct_errors
                ref=next(r for r in old[str(d)]['curves'] if r['method']==method and r['seed']==seed)['energy_error']
                np.testing.assert_allclose(curve[:len(ref)],ref,atol=1e-12,rtol=1e-9)
                validation['prefix_error_max_difference']=float(np.max(abs(curve[:len(ref)]-ref)))
                if method=='gd80':
                    refz=gdref[f'z_s{seed}_gd80'];np.testing.assert_array_equal(Z[:len(refz)],refz)
                    validation['GD_prefix_parameters_identical']=True
                rows.append(dict(method=method,seed=seed,energy_error=curve.tolist(),validation=validation))
                snapshots[f'z_s{seed}_{method}']=Z
        summary=[]
        for method in methods:
            curve=np.median([r['energy_error'] for r in rows if r['method']==method],axis=0)
            windows=[('8_to_half',8,d//2),('quarter_to_half',max(1,d//4),d//2),
                     ('half_to_three_quarters',d//2,3*d//4),('three_quarters_to_last',3*d//4,d-1)]
            fits={name:fit(curve,lo,hi) for name,lo,hi in windows if hi>=lo+3}
            summary.append(dict(method=method,median_curve=curve.tolist(),fits=fits,
                checkpoints=[dict(N=n,E=float(curve[n])) for n in sorted(set([d//4,d//2,3*d//4,d-1]))]))
        report=dict(d=d,T=80,alpha=.6,N_max=d-1,checks=checks,curves=rows,summary=summary,
                    test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest())
        (out/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/f'd{d}_snapshots.npz',**snapshots);reports.append(report)
        gd=summary[0];print(json.dumps(dict(d=d,checkpoints=gd['checkpoints'],fits=gd['fits'])),flush=True)
    from threadpoolctl import threadpool_info
    libs=threadpool_info();assert all(x['num_threads']==1 for x in libs)
    (out/'summary.json').write_text(json.dumps(dict(settings=dict(T=80,alpha=.6,seeds=SEEDS,dimensions=DIMS,
        N_max='d-1',test='Unchanged saved 4096 uniform interior + 4096 uniform vertices per dimension',
        statistics='Four-seed median test-maximum energy error; grids at d=4 use seed 1000 only',
        evaluation='Reverse cumulative squared orthogonal-complement coefficients; direct projection checks',
        platform=platform.platform(),threadpools=libs),reports=[dict(d=r['d'],summary=r['summary']) for r in reports]),indent=2)+'\n')
    with (out/'checkpoints.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','method','N','E_N'])
        for r in reports:
            for s in r['summary']:
                for v in s['checkpoints']:w.writerow([r['d'],s['method'],v['N'],v['E']])
    with (out/'fits.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','method','interval','N_start','N_end','power_exponent','power_log_rmse','exponential_log_rmse'])
        for r in reports:
            for s in r['summary']:
                for name,v in s['fits'].items():
                    w.writerow([r['d'],s['method'],name,v['N_start'],v['N_end'],v['power_exponent'],v['power_log_rmse'],v['exponential_log_rmse']])
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
    print('COMPLETE', flush=True)
