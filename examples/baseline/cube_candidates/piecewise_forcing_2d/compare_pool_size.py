"""Compare optimized GD with nested fixed pools on the unchanged slide tests.

Full training from V0 to floor(d/2), with randomized interleaved timings.
All diagnostics, test evaluation and I/O are outside the training timers.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,hashlib,json,platform
from pathlib import Path
import numpy as np
from equal_time_pool import train
from run import model

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'runs/piecewise_forcing_2d_20261112'
SUMMARY=BASE/'slides_summary'
SEEDS=[1000,1001,1002,1003]

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Use a fresh output directory.')
    previous=json.loads((SUMMARY/'shared_test_errors.json').read_text())
    reports=[]
    for d in args.dims:
        B,G,checks=model(round(np.sqrt(d)));N=d//2
        test_path=SUMMARY/f'd{d}_test.npz';test=np.load(test_path)['parameters']
        Utest=test@B.T;norm2=np.sum(Utest**2,axis=1)
        assert test.shape==(8192,d)
        spaces={};snapshots={};results=[];validation=[]
        methods=[0]+args.sizes
        for M in methods:
            method='gd80' if M==0 else f'pool{M}'
            for seed in SEEDS:
                Q,Z,_=train(B,G,seed,M);spaces[(M,seed)]=Q;snapshots[f'z_s{seed}_{method}']=Z
                e2=np.maximum(0,norm2[:,None]-np.cumsum((Utest@Q)**2,axis=1))
                curve=np.r_[np.sqrt(norm2.max()),np.sqrt(e2.max(axis=0))]
                direct=float(np.linalg.norm(test@(B-Q@(Q.T@B)).T,axis=1).max())
                np.testing.assert_allclose(curve[-1],direct,rtol=1e-11,atol=1e-13)
                old=next((r for r in previous[str(d)]['curves'] if r['seed']==seed and r['method']==method),None)
                if old is not None:
                    np.testing.assert_allclose(curve,old['energy_error'][:N+1],rtol=1e-10,atol=1e-12)
                if M:
                    pool=np.random.default_rng(seed).uniform(-1,1,(M,d))
                    if M==max(args.sizes):
                        np.testing.assert_array_equal(pool[:min(args.sizes)],np.random.default_rng(seed).uniform(-1,1,(min(args.sizes),d)))
                    # Independent direct residual evaluation checks the final
                    # greedy selection made by cached rank-one score updates.
                    Qprev=Q[:,:-1];R=B-Qprev@(Qprev.T@B)
                    scores=np.sum((pool@R.T)**2,axis=1)
                    np.testing.assert_array_equal(Z[-1],pool[np.argmax(scores)])
                orth=float(np.linalg.norm(Q.T@Q-np.eye(N)))
                assert orth<1e-11
                validation.append(dict(method=method,seed=seed,orthogonality_defect=orth,
                    old_error_curve_checked=old is not None,final_pool_selection_direct_checked=bool(M)))
                results.append(dict(method=method,M=M,seed=seed,energy_error=direct,error_curve=curve.tolist()))
        print(f'd={d}: errors and cached-score selections verified',flush=True)
        jobs=[(M,seed) for M in methods for seed in SEEDS];timings=[];rng=np.random.default_rng(441000+d)
        for repeat in range(args.repeats):
            for index in rng.permutation(len(jobs)):
                M,seed=jobs[index];_,_,t=train(B,G,seed,M)
                timings.append(dict(M=M,seed=seed,repeat=repeat,**t))
        rows=[]
        for M in methods:
            e=np.array([r['energy_error'] for r in results if r['M']==M]);t=np.array([r['wall_seconds'] for r in timings if r['M']==M])
            rows.append(dict(method='gd80' if M==0 else f'pool{M}',M=M,error_median=float(np.median(e)),
                error_min=float(e.min()),error_max=float(e.max()),wall_ms_median=float(1000*np.median(t)),
                wall_ms_q25=float(1000*np.quantile(t,.25)),wall_ms_q75=float(1000*np.quantile(t,.75))))
        report=dict(d=d,N=N,checks=checks,test_file=str(test_path.relative_to(ROOT)),
                    test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),results=results,timings=timings,summary=rows,validation=validation)
        np.savez_compressed(out/f'd{d}_snapshots.npz',**snapshots)
        (out/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        print(json.dumps(dict(d=d,N=N,summary=rows)),flush=True)
    from threadpoolctl import threadpool_info
    env=threadpool_info();assert all(t['num_threads']==1 for t in env)
    settings=dict(dims=args.dims,N='floor(d/2)',seeds=SEEDS,repeats=args.repeats,pool_sizes=args.sizes,
                  T=80,alpha=.6,test='Same saved 4096 uniform interior + 4096 uniform vertex points per d.',
                  timing='Whole greedy training, including sampling and precomputation; excludes shared model, diagnostics, test evaluation and I/O.',
                  order='Sequential randomized interleaving after validated warmup',numpy=np.__version__,platform=platform.platform(),threadpools=env,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('run.py'),Path(__file__).with_name('equal_time_pool.py')]})
    (out/'summary.json').write_text(json.dumps(dict(settings=settings,reports=[dict(d=r['d'],N=r['N'],summary=r['summary']) for r in reports]),indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['d','N','method','error_median','wall_ms_median'])
        for r in reports:
            for s in r['summary']:writer.writerow([r['d'],r['N'],s['method'],s['error_median'],s['wall_ms_median']])
    print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dims',nargs='+',type=int,default=[4,16,25,64,100,144,256])
    p.add_argument('--sizes',nargs='+',type=int,default=[10000,40000])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
