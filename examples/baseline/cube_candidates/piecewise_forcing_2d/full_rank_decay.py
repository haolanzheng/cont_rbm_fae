"""Extend the unchanged GD80/pool slide experiments through N=d-1.

Compute errors using orthogonal-complement coordinates to avoid cancellation
near full rank. Check every existing error-curve prefix and GD snapshot prefix.
Fits describe finite observed intervals; they are not asymptotic claims.
"""
from rerun_slides import (np,plt,Path,json,hashlib,platform,train_method,model,
                         BASE,DIMS,SEEDS,GRIDS,COLORS,MARKERS,STYLES)
import argparse,csv

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
        gdref=np.load(BASE/'gd20_20261116'/f'd{d}_snapshots.npz')
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

def plots(reports,out):
    labels=['GD ($T=80$)','Pool 256','Pool 4096','Pool 10000']
    bydim={r['d']:r for r in reports}
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,d in zip(axes.flat,[64,100,144,256]):
        r=bydim[d]
        for method,label,c,m,ls in zip(METHODS,labels,COLORS,MARKERS,STYLES):
            y=np.array(next(s['median_curve'] for s in r['summary'] if s['method']==method));n=np.arange(1,d)
            ax.loglog(n,y[1:],label=label,color=c,marker=m,markevery=max(1,d//12),ms=3,ls=ls,lw=1.5)
        y=np.array(r['summary'][0]['median_curve']);n=np.arange(8,d)
        ax.loglog(n,y[8]*np.sqrt(8/n),color='#999999',ls=(0,(6,3)),lw=1,label=r'$N^{-1/2}$ (reference)')
        ax.axvline(d/2,color='#BBBBBB',lw=.8);ax.axvline(3*d/4,color='#BBBBBB',lw=.8,ls=':')
        ax.set_title(f'$d={d}$');ax.set_xlabel('$N$');ax.set_ylabel('$E_N$');ax.grid(alpha=.16,which='both')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=3,frameon=False,fontsize=10)
    fig.savefig(out/'full_rank_decay.pdf');fig.savefig('/tmp/piecewise_full_rank_decay.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(14,12),layout='constrained')
    for ax,r in zip(axes.flat,reports):
        d=r['d']
        for method,label,c,m,ls in zip(METHODS,labels,COLORS,MARKERS,STYLES):
            y=np.array(next(s['median_curve'] for s in r['summary'] if s['method']==method))
            ax.semilogy(np.arange(d),y,label=label,color=c,marker=m,markevery=max(1,d//10),ms=2.5,ls=ls,lw=1.2)
        ax.set_title(f'$d={d}$');ax.set_xlabel('$N$');ax.set_ylabel('$E_N$');ax.grid(alpha=.16)
    for ax in list(axes.flat)[len(reports):]:ax.set_visible(False)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=4,frameon=False)
    fig.savefig(out/'all_dimensions.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--plots-only',action='store_true');args=p.parse_args()
    reports=[json.loads((args.output/f'd{d}.json').read_text()) for d in DIMS] if args.plots_only else run(args.output)
    plots(reports,args.output);print('COMPLETE',flush=True)
