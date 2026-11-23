"""Reproducible CPU screen; no slides or existing results are overwritten."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .moving_heat_sources import MovingHeatSources

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--widths',nargs='+',type=float,default=[.05,.1]);parser.add_argument('--dims',nargs='+',type=int,default=[16,64]);parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--N',type=int,default=16);args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    settings=dict(vars(args),methods=['gd80','gd320','pool256','pool10000','oracle'],alpha=.6,modes=128,validation='8193 single-source points; convex hull identity gives exact parameter reduction, spatial grid supremum is numerical',error='max energy error / max single-source energy norm',device='CPU float64, one BLAS thread',objective='squared exact energy error in truncated analytic sine basis',normalizer='max residual derivative norm squared / d on 257 source locations',N='number of snapshots; starts empty')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    rows=[]
    for width in args.widths:
      for d in args.dims:
        model=MovingHeatSources(d,width);single=model.single(np.linspace(0,1,8193));scale=np.linalg.norm(single,axis=1).max()
        rng=np.random.default_rng(101);q=model.append(np.empty((model.modes,0)),rng.uniform(0,1,d));z=rng.uniform(.1,.9,d)
        val,g=model.objective(z,q,True);eps=1e-6;gn=np.array([(model.objective(z+eps*e,q)-model.objective(z-eps*e,q))/(2*eps) for e in np.eye(d)])
        check=float(np.linalg.norm(g-gn)/np.linalg.norm(gn));assert check<1e-6,check
        for seed in range(args.seeds):
          rng=np.random.default_rng(1000+seed);pool=rng.uniform(0,1,(10000,d));U=np.concatenate([model.solution(p) for p in np.array_split(pool,100)])
          starts=rng.uniform(0,1,(args.N,d))
          for method in settings['methods']:
            if method=='oracle' and seed>0:continue
            q=np.empty((model.modes,0));selected=[];history=[];curve=[];begin=time.perf_counter()
            for n in range(args.N):
              if method.startswith('gd'):z,value=model.search(q,starts[n],steps=int(method[2:]))
              elif method.startswith('pool'):
                M=int(method[4:]);r=U[:M]-(U[:M]@q)@q.T;v=np.sum(r*r,axis=1);idx=int(v.argmax());z=pool[idx];value=float(v[idx])
              else:
                r=single-(single@q)@q.T;v=np.sum(r*r,axis=1);idx=int(v.argmax());z=np.full(d,idx/(len(single)-1));value=float(v[idx])
              q=model.append(q,z);selected.append(z.copy());history.append(value)
              r=single-(single@q)@q.T;curve.append(float(np.linalg.norm(r,axis=1).max()/scale))
            seconds=time.perf_counter()-begin
            row=dict(d=d,width=width,seed=seed,method=method,curve=curve,seconds_including_grid_validation=seconds,gradient_check=check)
            rows.append(row)
            np.savez_compressed(out/f'd{d}_h{width}_{method}_s{seed}.npz',Q=q,selected=np.array(selected),selected_J=np.array(history))
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print(d,width,seed,method,'N4,8,16',[round(curve[n-1],5) for n in [4,8,16] if n<=len(curve)],flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):run()
