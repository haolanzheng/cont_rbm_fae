"""Small full-RB screen; no presentation or earlier result is overwritten."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.stats import qmc
from .ridge_background import RidgeBackground

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/cube_candidate_screen_20261104'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def optimize(model,Q,z,T=80,alpha=.6):
    ell=model.normalizer(Q);best=z.copy();maximum=-np.inf
    for t in range(T+1):
        value,g=model.objective(z,Q,gradient=True)
        if value>maximum:best=z.copy();maximum=float(value)
        if t<T:
            g/=ell
            z=np.clip(z+alpha*g/np.sqrt(1+g@g),-1,1)
    return best,maximum

def adversarial_points(model,Q,initial,T=160,alpha=.6):
    """Independent batch of projected-gradient searches on a saved RB space."""
    ell=model.normalizer(Q);z=initial.copy();best=z.copy()
    maximum=np.full(len(z),-np.inf)
    for t in range(T+1):
        value,g=model.objective(z,Q,gradient=True)
        improve=value>maximum
        best[improve]=z[improve];maximum[improve]=value[improve]
        if t<T:
            g=g/ell
            z=np.clip(z+alpha*g/np.sqrt(1+np.sum(g*g,axis=1))[:,None],-1,1)
    return best,maximum

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dimensions',type=int,nargs='+',default=[16,32])
    parser.add_argument('--ridges',type=int,default=4)
    parser.add_argument('--epsilon',type=float,default=.1)
    parser.add_argument('--beta',type=float,nargs='+',default=[.05,.1])
    parser.add_argument('--seeds',type=int,nargs='+',default=[1000,1001])
    parser.add_argument('--tag',default='ridge_background')
    parser.add_argument('--adversarial-starts',type=int,default=0)
    parser.add_argument('--adversarial-T',type=int,default=160)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    rows=[];curves=[];audits=[]
    save(args.tag+'_settings.json',dict(dimensions=args.dimensions,ridges=args.ridges,
        epsilon=args.epsilon,beta=args.beta,seeds=args.seeds,T=80,alpha=.6,
        initial='actual center snapshot z=0',normalizer='Frobenius norm of exact center residual-Jacobian Gram, fixed per round',
        physical_modes='D=d+R',N='1..2R',methods=['gd','pool256','pool4096','pool10000'],
        test='8192 Sobol interior,4096 random vertices, positive/negative ridge vertices, ridge rays and perturbed ridge vertices',
        device='CPU float64, one BLAS thread',error='exact Galerkin energy error, maximum over common saved test points',
        N_definition='center snapshot counts as N=1; N=4 adds three snapshots; N=8 adds seven',
        pool_nesting='same random seed and uniform draw order: pool256 is prefix of pool4096, itself prefix of pool10000',
        adversarial_starts=args.adversarial_starts,adversarial_T=args.adversarial_T,
        adversarial_method='independent projected-gradient starts on every saved reduced space; union shared across all methods and seeds',
        timing='candidate generation, solution evaluation, greedy search, and enrichment; common tests and all adversarial validation excluded'))
    for d in args.dimensions:
      for beta in args.beta:
        model=RidgeBackground(d,args.ridges,args.epsilon,beta)
        rng=np.random.default_rng(2026110400+d)
        interior=2*qmc.Sobol(d,scramble=True,seed=202611040+d).random_base2(13)-1
        vertices=rng.choice([-1.,1.],size=(4096,d))
        rays=np.concatenate([np.linspace(-1,1,101)[:,None]*v for v in model.directions])
        perturbed=np.concatenate([v[None,:]*(1-rng.uniform(0,.2,(256,d))) for v in model.directions])
        test=np.vstack((interior,vertices,model.directions,-model.directions,rays,perturbed,np.zeros((1,d))))
        test_solutions=model.solution(test)
        np.savez_compressed(OUT/f'{args.tag}_test_d{d}_b{beta}.npz',parameters=test)
        spaces={}
        for seed in args.seeds:
          for method in ('gd','pool256','pool4096','pool10000'):
            rng=np.random.default_rng(seed+1000003)
            Q=model.initial_basis();elapsed=0.;selected=[]
            if method!='gd':
                start=time.perf_counter();pool=rng.uniform(-1,1,(int(method[4:]),d))
                pool_solutions=model.solution(pool);elapsed+=time.perf_counter()-start
            for n in range(1,2*args.ridges+1):
                spaces[(method,seed,n)]=Q.copy()
                E=float(model.errors_from_solutions(test_solutions,Q).max())
                peakE=float(model.errors_from_solutions(model.solution(model.directions),Q).max())
                curves.append(dict(d=d,R=args.ridges,beta=beta,epsilon=args.epsilon,
                    seed=seed,method=method,N=n,E=E,peak_error=peakE,
                    global_upper=model.global_error_upper_bound(Q)))
                if n==2*args.ridges:break
                start=time.perf_counter()
                if method=='gd':z,value=optimize(model,Q,rng.uniform(-1,1,d))
                else:
                    values=model.errors_from_solutions(pool_solutions,Q)**2
                    k=int(np.argmax(values));z=pool[k].copy();value=float(values[k])
                Qnew=model.append_basis(Q,z)
                if Qnew.shape[1]!=n+1:raise AssertionError('Dependent selected snapshot')
                Q=Qnew;elapsed+=time.perf_counter()-start;selected.append(z.tolist())
            row=dict(d=d,R=args.ridges,D=model.state_dimension,beta=beta,epsilon=args.epsilon,
                seed=seed,method=method,N=2*args.ridges,E=E,peak_error=peakE,seconds=elapsed,
                global_upper=model.global_error_upper_bound(Q))
            rows.append(row);print(row,flush=True)
            np.savez_compressed(OUT/f'{args.tag}_d{d}_b{beta}_{method}_s{seed}.npz',Q=Q,
                selected_parameters=selected,
                **{f'Q_N{n}':spaces[(method,seed,n)] for n in range(1,2*args.ridges+1)})
          save(args.tag+'_results.json',rows);save(args.tag+'_curves.json',curves)
        if args.adversarial_starts:
            candidates=[]
            for index,((method,seed,n),Q) in enumerate(spaces.items()):
                rng=np.random.default_rng(20261104000+100000*d+1000*seed+100*n+METHOD_INDEX[method])
                initial=rng.uniform(-1,1,(args.adversarial_starts,d))
                before=float(np.sqrt(model.objective(initial,Q)).max())
                points,values=adversarial_points(model,Q,initial,args.adversarial_T)
                candidates.append(points)
                audits.append(dict(d=d,R=args.ridges,beta=beta,epsilon=args.epsilon,
                    method=method,seed=seed,N=n,starts=args.adversarial_starts,T=args.adversarial_T,
                    maximum_initial_error=before,maximum_returned_error=float(np.sqrt(values).max())))
                if (index+1)%32==0:
                    print('ADVERSARIAL',d,beta,index+1,'/',len(spaces),flush=True)
            adversarial=np.concatenate(candidates)
            common=np.unique(np.vstack((test,adversarial)),axis=0)
            common_solutions=model.solution(common)
            np.savez_compressed(OUT/f'{args.tag}_test_d{d}_b{beta}.npz',parameters=common,
                                base_parameters=test,adversarial_parameters=adversarial)
            for rec in curves:
                if rec['d']==d and rec['beta']==beta:
                    Q=spaces[(rec['method'],rec['seed'],rec['N'])]
                    rec['E_base']=rec['E']
                    rec['E']=float(model.errors_from_solutions(common_solutions,Q).max())
                    rec['common_test_count']=len(common)
            for rec in rows:
                if rec['d']==d and rec['beta']==beta:
                    final=next(c for c in curves if c['d']==d and c['beta']==beta
                        and c['method']==rec['method'] and c['seed']==rec['seed'] and c['N']==rec['N'])
                    rec['E_base']=rec['E'];rec['E']=final['E'];rec['common_test_count']=len(common)
            save(args.tag+'_results.json',rows);save(args.tag+'_curves.json',curves)
            save(args.tag+'_adversarial.json',audits)
            print('COMMON ADVERSARIAL TEST',d,beta,len(common),flush=True)

METHOD_INDEX={'gd':0,'pool256':1,'pool4096':2,'pool10000':3}

if __name__=='__main__':main()
