"""Exact energy Gram for -u''=sum z_j 1_{I_j}, z in [-1,1]^d.

No spatial truncation: G_ij=h^2(min(c_i,c_j)-c_i*c_j-h*delta_ij/6).
Global max solution energy norm is exactly 1/sqrt(12). Error values are
absolute energy errors divided by that constant when reported normalized.
Numerical worst-error lower bounds use shared vertices found on ALL spaces.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys,json,time
from pathlib import Path
import numpy as np

DEFAULT_OUTPUT=Path(__file__).resolve().parents[4]/'runs/alignment_baseline_20261111/piecewise_forcing'

def model(d):
    h=1/d;x=(np.arange(d)+.5)*h
    G=h*h*(np.minimum(x[:,None],x[None,:])-np.outer(x,x)-h*np.eye(d)/6)
    ev,U=np.linalg.eigh(G)
    B=np.sqrt(ev)[:,None]*U.T
    return B,G

def ascend(H,z,steps):
    z=np.array(z,copy=True);best=z.copy()
    val=np.einsum('...i,...i->...',z@H,z);ell=max(np.linalg.norm(H),1e-30)
    for _ in range(steps):
        g=2*(z@H)/ell
        z=np.clip(z+.6*g/np.sqrt(1+np.sum(g*g,axis=-1,keepdims=True)),-1,1)
        v=np.einsum('...i,...i->...',z@H,z)
        better=v>val;best=np.where(better[...,None],z,best);val=np.maximum(val,v)
    return best,val

def vertex_search(H,z,steps=100):
    z=np.array(z,copy=True)
    for _ in range(steps):
        new=np.where(z@H>=0,1.,-1.)
        if np.array_equal(new,z):break
        z=new
    return z

def append(B,Q,z):
    v=B@z
    for _ in range(2):v-=Q@(Q.T@v)
    v/=np.linalg.norm(v)
    return np.column_stack([Q,v]),v

def Hmatrix(B,G,Q):
    L=B.T@Q
    return G-L@L.T

def scalar_stats(B,G,Q,z):
    u=B@z;r=u-Q@(Q.T@u);J=float(r@r);q=r/np.sqrt(J)
    avg_gain=float(np.linalg.norm(B.T@q)**2/3)
    d=len(z);k=np.arange(1,min(d,16)+1)[:,None];left=np.arange(d)[None,:]/d
    F=np.sqrt(2)*(np.cos(np.pi*k*left)-np.cos(np.pi*k*(left+1/d)))/(np.pi*k)
    shape={'sign_changes':int(np.count_nonzero(z[1:]*z[:-1]<0)),
           'forcing_L2':float(np.linalg.norm(z)/np.sqrt(d)),
           'first16_fourier_energy_fraction':float(np.linalg.norm(F@z)**2/(z@z/d))}
    return J,avg_gain,shape

def errors(B,Q,z):
    u=z@B.T;r=u-(u@Q)@Q.T
    return np.sum(r*r,axis=1)

def run(d, output_dir=DEFAULT_OUTPUT):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if d not in (64,128,256):
        raise ValueError("This recorded experiment uses d = 64, 128, or 256.")
    B,G=model(d);rngtest=np.random.default_rng(54000+d)
    shapes={};choices={};training=[];same=[];start_all=time.monotonic()
    checkpoints=[8,16,32,48]+([64] if d>=128 else [])+([128] if d>=256 else [])
    for seed in range(1000,1004):
        rng=np.random.default_rng(seed);pool=rng.uniform(-1,1,(10000,d))
        for method in ['gd80','gd320','pool256','pool10000']:
            Q=np.zeros((d,0));rngs=np.random.default_rng(seed+10000);zs=[];diag=[];started=time.monotonic()
            if method.startswith('pool'):
                candidates=pool[:int(method[4:])];pu=candidates@B.T;scores=np.sum(pu*pu,axis=1)
            for n in range(max(checkpoints)):
                H=Hmatrix(B,G,Q)
                if method.startswith('gd'):z,_=ascend(H,rngs.uniform(-1,1,d),int(method[2:]))
                else:z=candidates[np.argmax(scores)]
                J,avg,shape=scalar_stats(B,G,Q,z);diag.append({'round':n+1,'J':J,'exact_mean_square_gain':avg,**shape});zs.append(z)
                Q,v=append(B,Q,z)
                if method.startswith('pool'):scores-=np.square(pu@v)
                if n+1 in checkpoints:shapes[(seed,method,n+1)]=Q.copy()
            choices[(seed,method)]=np.array(zs)
            training.append({'seed':seed,'method':method,'seconds':time.monotonic()-started,'selected':diag})
        # Compare selected candidates starting at each exact same pool10000 space.
        for n in [8,16,32]+([64] if d>=256 else []):
            Q=shapes[(seed,'pool10000',n)];H=Hmatrix(B,G,Q);rec={'seed':seed,'N_before':n,'candidates':{}}
            for method in ['gd80','gd320','pool256','pool10000']:
                if method.startswith('gd'):
                    # Identical independent initial point for T=80/320.
                    z0=np.random.default_rng(seed+20000+n).uniform(-1,1,d)
                    z,_=ascend(H,z0,int(method[2:]))
                else:
                    p=pool[:int(method[4:])];vals=np.einsum('ij,ij->i',p@H,p);z=p[vals.argmax()]
                J,gain,shape=scalar_stats(B,G,Q,z);qa,_=append(B,Q,z)
                key=(seed,'same_'+method,n+1);shapes[key]=qa
                rec['candidates'][method]={'J':J,'exact_mean_square_gain':gain,**shape}
            same.append(rec)
        print(json.dumps({'status':'trained','d':d,'seed':seed,'elapsed':time.monotonic()-start_all}),flush=True)
    # Independent held-out starts; deterministic PSD quadratic sign iteration
    # is monotone in J and produces feasible vertices. These are lower bounds.
    base=np.vstack([rngtest.choice([-1.,1.],(3000,d)),rngtest.uniform(-1,1,(3000,d)),np.ones((1,d)),-np.ones((1,d))])
    tests=[base]
    for key,Q in shapes.items():
        H=Hmatrix(B,G,Q)
        z=vertex_search(H,rngtest.uniform(-1,1,(96,d)))
        tests.append(z)
    test=np.unique(np.vstack(tests),axis=0)
    U=test@B.T
    cache={}
    for key,Q in shapes.items():
        rr=U-(U@Q)@Q.T;sq=np.sum(rr*rr,axis=1);cache[key]=sq
    full=[]
    for seed in range(1000,1004):
        for method in ['gd80','gd320','pool256','pool10000']:
            for n in checkpoints:
                Q=shapes[(seed,method,n)];sq=cache[(seed,method,n)]
                eigmax=float(np.linalg.eigvalsh(Hmatrix(B,G,Q))[-1])
                full.append({'seed':seed,'method':method,'N':n,'max_energy_error':float(np.sqrt(sq.max())),
                  'global_error_spectral_upper':float(np.sqrt(d*max(eigmax,0))),
                  'global_error_spectral_upper_normalized':float(np.sqrt(12*d*max(eigmax,0))),
                  'lambda_max_H':eigmax,
                  'normalized_by_global_maxnorm':float(np.sqrt(12*sq.max())),
                  'exact_uniform_mean_square_error':float(np.trace(Hmatrix(B,G,Q))/3),
                  'test_p99_squared_error':float(np.quantile(sq,.99))})
    for rec in same:
        seed=rec['seed'];n=rec['N_before'];before=cache[(seed,'pool10000',n)]
        rec['E_before']=float(np.sqrt(before.max()));rec['mean_square_before']=float(np.trace(Hmatrix(B,G,shapes[(seed,'pool10000',n)]))/3)
        # Tail membership fixed before enrichment to compare the exact same points.
        tail=before>=np.quantile(before,.99)
        for method,data in rec['candidates'].items():
            after=cache[(seed,'same_'+method,n+1)]
            data.update({'E_after':float(np.sqrt(after.max())),
             'squared_worst_error_gain':float(before.max()-after.max()),
             'fixed_top1pct_mean_square_gain':float(np.mean(before[tail]-after[tail])),
             'fixed_top1pct_mean_square_before':float(np.mean(before[tail]))})
    settings={'d':d,'PDE':"-u''=sum z_j 1_Ij, zero Dirichlet, z in [-1,1]^d",'alpha':.6,
      'seeds':list(range(1000,1004)),'pool':'fixed iid uniform nested 256 in10000',
      'normalizer':'Frobenius norm of exact projected energy Gram',
      'validation':'96 independent starts per every compared space, 100 monotone PSD sign iterations; union plus 3000 random vertices and 3000 uniform points',
      'n_test':len(test),'validation_status':'test values are lower bounds; analytic global upper sqrt(d*lambda_max(H)) evaluated float64, not interval arithmetic','global_max_solution_norm':1/np.sqrt(12),
      'runtime_seconds':time.monotonic()-start_all}
    result={'settings':settings,'training':training,'full_greedy':full,'same_space':same}
    path=output_dir/f'piecewise_extended_d{d}.json';path.write_text(json.dumps(result,indent=2))
    save={f'Q_s{s}_{m}_n{n}':q for (s,m,n),q in shapes.items()}
    save.update({f'z_s{s}_{m}':z for (s,m),z in choices.items()})
    save.update({'test':test,'B':B,'G':G})
    np.savez_compressed(output_dir/f'piecewise_extended_d{d}.npz',**save)
    print(json.dumps({'status':'done','d':d,'path':str(path),**settings}),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('d',type=int,choices=[64,128,256])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    run(args.d,args.output)
