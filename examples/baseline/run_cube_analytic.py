"""Cube baseline with sampled analytic HF solutions and normalized projected GD.

Archived cube scripts use different steps and are deliberately unchanged.
Run: PYTHONPATH=src:. .venv_cpu/bin/python -u -m examples.baseline.run_cube_analytic
All analytical formula evaluations are float64, not interval certificates.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, time, hashlib, platform
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from threadpoolctl import threadpool_limits,threadpool_info
from .geometry_cpu import CoordinateROM,rom_options,sym
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/analytic_hf_a06_t40_20261022'
OUT=ROOT/'runs/cube_analytic_a06_t40_20261027'

def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def scores(H,Z):return np.sum((Z@H)*Z,axis=-1)

def search(H,starts,alpha=.6,T=40,threshold=None):
    z=np.atleast_2d(starts).copy();best=z.copy();v=scores(H,z);ell=np.linalg.norm(H)
    times=sorted(set([0,1,5,10,20,T]));history=[];mn=0.;old=v.copy()
    for t in range(T+1):
        current=scores(H,z);improved=current>v;best[improved]=z[improved];v=np.maximum(v,current)
        if t:mn=min(mn,float(np.min(current-old)))
        old=current
        if t in times:history.append({'T':t,'hits':None if threshold is None else int(np.sum(v>=threshold)),
                                      'mean_best_value':float(v.mean()),'mean_boundary_coordinates':float(np.mean(np.abs(z)>=1-1e-12))})
        if t==T:break
        grad=2*(z@H)/ell if ell else np.zeros_like(z)
        eta=alpha/np.sqrt(1+np.sum(grad*grad,axis=1,keepdims=True))
        z=np.clip(z+eta*grad,-1,1)
    return (best[0] if np.ndim(starts)==1 else best),{'history':history,'min_increment':mn,'feasibility_defect':float(np.max(np.maximum(np.abs(z)-1,0)))}

def vertex_bounds(H,exact_dimension=16):
    """Max of PSD quadratic is attained at a vertex. Return lower/upper bounds."""
    d=len(H);upper=float(np.abs(H).sum())
    if d<=exact_dimension:
        best=-np.inf;zbest=None
        for st in range(0,1<<(d-1),4096):
            ids=np.arange(st,min(st+4096,1<<(d-1)),dtype=np.int64)
            Z=np.ones((len(ids),d));Z[:,1:]=2*((ids[:,None]>>np.arange(d-1))&1)-1
            v=scores(H,Z);j=v.argmax()
            if v[j]>best:best=float(v[j]);zbest=Z[j].copy()
        return best,best,zbest
    # Every returned vertex gives a rigorous mathematical lower bound.
    rng=np.random.default_rng(6283+d);Z=np.sign(rng.standard_normal((128,d)));Z[0]=1
    _,vec=np.linalg.eigh(H);Z[1]=np.where(vec[:,-1]>=0,1.,-1.)
    for _ in range(50):
        changes=0
        for i in range(d):
            h=Z@H[:,i]-H[i,i]*Z[:,i];new=np.where(h>=0,1.,-1.)
            changes+=np.count_nonzero(new!=Z[:,i]);Z[:,i]=new
        if not changes:break
    v=scores(H,Z);i=v.argmax()
    return float(v[i]),upper,Z[i]

def face_certificate(H,alpha,T,gamma,maximum_upper,max_k=13):
    """All initial absolute coordinates are independent Uniform[0,1].

    b_i=sum_{j!=i}|H_ij|, r_i=b_i/H_ii, c_i=1+2 eta_min H_ii/ell.
    If |z_i(0)| >= r_i+(1-r_i)c_i^-T and r_i<1, coordinate i
    reaches its initial sign at time T and stays there. For each sign s on I,
    J_raw >= s^T H_II s - 2 ||H_JI s||_1 on the entire resulting face.
    Count all signs that put this face above gamma^2*maximum_upper.
    """
    ell=np.linalg.norm(H);di=np.diag(H);row=np.abs(H).sum(1);off=row-np.abs(di)
    eta=alpha/np.sqrt(1+4*np.sum(row**2)/ell**2)
    order=[int(i) for i in np.argsort(-di) if di[i]>off[i] and di[i]>0]
    best={'lower':0.,'k':0,'indices':[],'good_sign_count':0,'eta_min':float(eta)}
    candidates=[]
    for k in range(1,min(max_k,len(order))+1):
        I=np.array(order[:k]);J=np.array([j for j in range(len(H)) if j not in I],dtype=int)
        S=2*((np.arange(1<<k)[:,None]>>np.arange(k))&1)-1
        L=scores(H[np.ix_(I,I)],S)-2*np.abs(S@H[np.ix_(I,J)]).sum(1)
        # Buffer only for conservative float evaluation; not interval arithmetic.
        good=L>=gamma**2*maximum_upper+1e-12*maximum_upper
        r=off[I]/di[I];c=1+2*eta*di[I]/ell
        widths=(1-r)*(-np.expm1(-T*np.log(c)))
        lower=float(np.mean(good)*np.prod(widths))
        rec={'lower':lower,'k':k,'indices':I.tolist(),'good_sign_count':int(good.sum()),
             'total_signs':1<<k,'eta_min':float(eta),'r':r.tolist(),'c':c.tolist(),
             'initial_abs_threshold':(1-widths).tolist()}
        candidates.append({'k':k,'lower':lower,'good_sign_fraction':float(good.mean())})
        if lower>best['lower']:best=rec
    return best|{'candidates':candidates}

def diagonal_pool_upper(H,target,bins=131072):
    """Deterministic discrete convolution upper bound for P(z'H z>=target).

    H <= diag(row absolute sums). For U_i iid Uniform[0,1], round
    a_i U_i^2 UP to multiples of delta. The discrete sum majorizes the
    quadratic. Its exact probabilities are sqrt differences. FFT only
    evaluates this finite formula; numerical error is not interval certified.
    """
    a=np.abs(H).sum(1);delta=float(a.sum()/bins);dist=np.array([1.]);clipped=0.
    for w in a:
        if w<=0:continue
        k=int(np.ceil(w/delta));edges=np.minimum(np.arange(k+1)*delta/w,1)
        pmf=np.r_[0.,np.diff(np.sqrt(edges))]
        dist=fftconvolve(dist,pmf);clipped+=float(-dist[dist<0].sum());dist=np.maximum(dist,0)
    cut=max(0,int(np.ceil(target/delta)))
    upper=float(min(1,dist[cut:].sum()+1e-12))
    return {'upper':upper,'bins':bins,'delta':delta,'max_rounding_slack':len(H)*delta,
            'fft_negative_mass':clipped,'qualification':'Exact finite upper-bound formula evaluated in float64; no outward rounding.'}

def validate():
    rng=np.random.default_rng(382);H=np.array([[2.,.1],[.1,.8]])
    Z=rng.uniform(-1,1,(1000,2));_,audit=search(H,Z)
    assert audit['feasibility_defect']==0 and audit['min_increment']>-1e-11
    lo,hi,v=vertex_bounds(H);assert abs(lo-3.)<1e-12 and lo==hi
    # Check face events against actual updates using a coupled PSD example.
    cert=face_certificate(H,.6,40,.9,hi)
    assert cert['lower']>0
    I=cert['indices'];r=np.array(cert['initial_abs_threshold']);S=np.sign(Z[:,I]);inside=np.all(np.abs(Z[:,I])>=r,1)
    J=np.array([j for j in range(2) if j not in I],dtype=int)
    good=scores(H[np.ix_(I,I)],S)-2*np.abs(S@H[np.ix_(I,J)]).sum(1)>=.81*hi+1e-12*hi
    B,_=search(H,Z[inside&good]);assert np.all(scores(H,B)>=.81*hi-1e-11)
    # Independent two-coordinate diagonal check by quadrature upper vs volume.
    from scipy.integrate import quad
    D=np.diag([1.,.5]);target=1.1
    exact=quad(lambda u:1-np.sqrt(np.clip((target-u*u)/.5,0,1)),0,1,epsabs=1e-12,points=[np.sqrt(.6)])[0]
    ub=diagonal_pool_upper(D,target,bins=32768)['upper'];assert ub>=exact-1e-10 and ub-exact<1e-3
    return {'face_event_test_points':int(np.sum(inside&good)),'diagonal_probability':exact,'discrete_upper':ub,'normalized_update':audit}

def train(dimensions,seeds):
    summary=[];records=[];all_runs=[]
    for d in dimensions:
        data=np.load(SOURCE/f'model_d{d}.npz');opts=rom_options(data);G0=data['G0'];H0=data['H0']
        Z=np.random.default_rng(202610270+d).uniform(-1,1,(16384,d));den=scores(G0,Z)
        np.savez_compressed(OUT/f'test_cube_d{d}.npz',test_parameters=Z)
        for method in ['gd','pool256','pool4096','pool10000']:
            values=[];times=[]
            for seed in seeds:
                rng=np.random.default_rng(seed);rom=CoordinateROM(G0,H0,**opts);rounds=[];errors=[]
                begin=time.perf_counter()
                pool=None if method=='gd' else np.random.default_rng(seed+1000003).uniform(-1,1,(int(method[4:]),d))
                setup=time.perf_counter()-begin;elapsed=setup
                for n in range(d//2):
                    start=time.perf_counter();H,G=rom.matrices()
                    if method=='gd':z,audit=search(H,rng.uniform(-1,1,d))
                    else:z=pool[np.argmax(scores(H,pool))].copy();audit={}
                    rom.append(z);elapsed+=time.perf_counter()-start
                    rounds.append({'n':n,'parameter':z.tolist(),'selected_value':float(z@H@z),'audit':audit})
                    _,G=rom.matrices();errors.append(float(np.sqrt(np.mean(np.maximum(scores(G,Z),0)/den))))
                values.append(errors[-1]);times.append(1000*elapsed)
                rec={'d':d,'method':method,'seed':seed,'N':d//2,'test_error_by_N':errors,'wall_ms':1000*elapsed,'rounds':rounds,'orthogonality_defect':rom.orthogonality_defect()}
                all_runs.append(rec)
            summary.append({'d':d,'N':d//2,'method':method,'test_count':16384,'seeds':len(seeds),
                            'median_root_mean_square_relative_error':float(np.median(values)),
                            'median_wall_ms':float(np.median(times))})
            dump(OUT/'greedy_summary.json',summary);dump(OUT/'greedy_runs.json',all_runs)
            print('TRAIN',summary[-1],flush=True)
    return summary

def probabilities(dimensions,bins):
    results=[]
    for d in dimensions:
        a=np.load(SOURCE/f'model_d{d}.npz')
        for n in [0,d//2]:
            if n==0:H=a['H0']
            else:H=np.load(SOURCE/f'reference_d{d}_n{n}.npz')['H']
            ml,mu,s=vertex_bounds(H);lower=face_certificate(H,.6,40,.9,mu)
            upper=diagonal_pool_upper(H,.81*ml,bins)
            up=upper['upper'];pool={str(m):float(-np.expm1(m*np.log1p(-up))) if up<1 else 1. for m in [41,256,4096,10000]}
            Z=np.random.default_rng(2026102700+100*d+n).uniform(-1,1,(4096,d))
            best,audit=search(H,Z,threshold=.81*mu);v=scores(H,best)
            # The two empirical values enclose the frequency for the unknown true maximum.
            gd_range=[float(np.mean(v>=.81*mu)),float(np.mean(v>=.81*ml))]
            np.savez_compressed(OUT/f'reference_cube_d{d}_n{n}.npz',H=H,maximizing_vertex_candidate=s)
            rec={'d':d,'n':n,'gamma':.9,'alpha':.6,'T':40,'max_lower':ml,'max_upper':mu,
                 'max_ratio':mu/ml,'gd_lower_certificate':lower,'one_sample_upper':upper,
                 'pool_upper':pool,'gd_empirical_frequency_interval':gd_range,'audit':audit,
                 'provable_advantage_over_41':lower['lower']>pool['41'],
                 'reference':'fixed sphere exact-indicator reference space, admissible also for cube; NOT a cube-greedy reference'}
            results.append(rec);dump(OUT/'probability_bounds.json',results)
            print('PROB',d,n,'maxratio',mu/ml,'lower',lower['lower'],'pool41up',pool['41'],'emp',gd_range,flush=True)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--dimensions',type=int,nargs='+',default=[8,16,32,48,64]);p.add_argument('--seeds',type=int,nargs='+',default=list(range(1000,1008)));p.add_argument('--bins',type=int,default=131072);p.add_argument('--probability-only',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'validation.json',validate())
    dump(OUT/'metadata.json',{'alpha':.6,'T':40,'gamma':.9,'rho':.78,'domain':'[-1,1]^d',
        'dimensions':args.dimensions,'seeds':args.seeds,'source_models':str(SOURCE),
        'update':'clip(z + alpha*(2Hz/ell)/sqrt(1+||2Hz/ell||^2), -1,1)',
        'cpu':platform.platform(),'threadpools':threadpool_info(),'test_count':16384,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualification':'No HF solve. Actual assembled H, full coupled matrices. Formula evaluations float64, not rigorous machine interval enclosures.'})
    if not args.probability_only:train(args.dimensions,args.seeds)
    probabilities(args.dimensions,args.bins)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
