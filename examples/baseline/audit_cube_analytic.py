"""Additional cube checks: low-dimensional grid, width brackets and actual GD spaces."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/cube_analytic_mpl')
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
from .geometry_cpu import CoordinateROM,rom_options
from .run_cube_analytic import SOURCE,OUT,dump,search,scores,vertex_bounds,face_certificate,diagonal_pool_upper

def main():
    runs=json.loads((OUT/'greedy_runs.json').read_text());d=6;a=np.load(SOURCE/f'model_d{d}.npz');opts=rom_options(a)
    Z=np.random.default_rng(202610270+d).uniform(-1,1,(16384,d));den=scores(a['G0'],Z)
    np.savez_compressed(OUT/'test_cube_d6.npz',test_parameters=Z)
    low=[]
    for method in ['gd','pool256','pool4096','pool10000','grid1','grid0.5','grid0.25']:
        for seed in ([0] if method.startswith('grid') else range(1000,1008)):
            rng=np.random.default_rng(seed);start=time.perf_counter();rom=CoordinateROM(a['G0'],a['H0'],**opts)
            if method.startswith('grid'):
                h=float(method[4:]);axis=np.linspace(-1,1,round(2/h)+1)
                pool=np.stack(np.meshgrid(*([axis]*d),indexing='ij'),-1).reshape(-1,d)
            elif method.startswith('pool'):pool=np.random.default_rng(seed+1000003).uniform(-1,1,(int(method[4:]),d))
            else:pool=None
            pars=[]
            for n in range(d//2):
                H,_=rom.matrices()
                z=search(H,rng.uniform(-1,1,d))[0] if pool is None else pool[scores(H,pool).argmax()].copy()
                rom.append(z);pars.append(z.tolist())
            ms=1000*(time.perf_counter()-start);_,G=rom.matrices();emax,_,_=vertex_bounds(G)
            w,V=np.linalg.eigh(a['G0']);n=d//2;L=(V[:,:d-n]*w[:d-n])@V[:,:d-n].T
            wlow=np.sqrt(w[:d-n].sum());_,wup,_=vertex_bounds(L);wup=np.sqrt(wup)
            low.append({'d':d,'N':n,'method':method,'seed':seed,'root_mean_square_relative_error':float(np.sqrt(np.mean(np.maximum(scores(G,Z),0)/den))),
                        'error_over_width_interval':[float(np.sqrt(emax)/wup),float(np.sqrt(emax)/wlow)],'wall_ms':ms,'parameters':pars,
                        'candidate_count':None if pool is None else len(pool)})
        print('LOW',method,flush=True)
    dump(OUT/'low_dimension_runs.json',low)
    summary=[]
    for m in dict.fromkeys(r['method'] for r in low):
        rows=[r for r in low if r['method']==m]
        summary.append({'method':m,'test_error':float(np.median([r['root_mean_square_relative_error'] for r in rows])),
          'error_over_width_interval':np.median([r['error_over_width_interval'] for r in rows],0).tolist(),
          'wall_ms':float(np.median([r['wall_ms'] for r in rows]))})
    dump(OUT/'low_dimension_summary.json',summary)
    # Inspect matrices from actual cube-GD histories, not only old reference spaces.
    audit=[];widths=[]
    for d in [8,16,32,48,64]:
        a=np.load(SOURCE/f'model_d{d}.npz');w,V=np.linalg.eigh(a['G0']);n=d//2
        E=(V[:,:d-n]*w[:d-n])@V[:,:d-n].T
        widths.append({'d':d,'N':n,'lower':float(np.sqrt(w[:d-n].sum())),
                       'upper':float(np.sqrt(np.abs(E).sum()))})
        r=next(r for r in runs if r['d']==d and r['method']=='gd' and r['seed']==1000)
        rom=CoordinateROM(a['G0'],a['H0'],**rom_options(a))
        for st in r['rounds']:rom.append(st['parameter'])
        H,_=rom.matrices();ml,mu,s=vertex_bounds(H);cert=face_certificate(H,.6,40,.9,mu)
        Z=np.random.default_rng(202610279+d).uniform(-1,1,(4096,d));B,info=search(H,Z,threshold=.81*mu);val=scores(H,B)
        audit.append({'d':d,'n':n,'source':'cube GD seed 1000, after N enrichments',
                      'lower_certificate':cert['lower'],'eligible_coordinate_count':int(np.sum(np.diag(H)>np.abs(H).sum(1)-np.diag(H))),
                      'maximum_lower':ml,'maximum_upper':mu,
                      'empirical_interval':[float(np.mean(val>=.81*mu)),float(np.mean(val>=.81*ml))],
                      'interpretation':'Zero certificate is inconclusive, not zero actual probability.'})
        np.savez_compressed(OUT/f'actual_cube_gd_d{d}_n{n}.npz',H=H,Q=rom.Q)
        print('ACTUAL',audit[-1],flush=True)
    dump(OUT/'actual_cube_space_audit.json',audit);dump(OUT/'width_brackets.json',widths)
    # Convolution resolution check on the smallest probability gap of interest.
    H=np.load(OUT/'reference_cube_d32_n16.npz')['H'];ml,mu,_=vertex_bounds(H)
    checks=[diagonal_pool_upper(H,.81*ml,b) for b in [32768,65536,131072,262144]]
    dump(OUT/'pool_bound_resolution.json',checks)
    # Plot only the data produced here. No slide or manuscript is modified.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((OUT/'greedy_summary.json').read_text())
    fig,ax=plt.subplots(figsize=(6.4,4))
    styles=[('gd','GD, T=40','o-'),('pool256','Pool 256','s--'),('pool4096','Pool 4096','^--'),('pool10000','Pool 10000','d:')]
    for m,label,style in styles:
        r=[r for r in rows if r['method']==m];ax.semilogy([v['d'] for v in r],[v['median_root_mean_square_relative_error'] for v in r],style,label=label)
    ax.set(xlabel='Parameter dimension d',ylabel='Root mean square relative energy error',xticks=[8,16,32,48,64]);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/f'cube_test_error.{ext}',dpi=180)
    plt.close(fig)
    rows=[r for r in json.loads((OUT/'probability_bounds.json').read_text()) if r['n']==r['d']//2]
    fig,ax=plt.subplots(figsize=(6.4,4))
    ax.semilogy([r['d'] for r in rows],[r['gd_lower_certificate']['lower'] for r in rows],'o-',label='GD: lower bound, T=40')
    ax.semilogy([r['d'] for r in rows],[r['pool_upper']['41'] for r in rows],'s--',label='41 independent samples: upper bound')
    ax.set(xlabel='Parameter dimension d',ylabel='Probability bound',xticks=[8,16,32,48,64]);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/f'cube_probability_bounds.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
