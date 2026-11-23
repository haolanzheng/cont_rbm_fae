"""Evaluate deterministic cube probability bounds for frozen FEM residual matrices.

This CPU-only calculation uses actual coupled H_n, with no diagonal surrogate.
Mathematical formulas are rigorous; their numerical evaluations use float64,
not outward-rounded interval arithmetic. GD classification includes a relative
1e-12*M safety buffer. The direct-volume upper uses ordinary float64 comparisons
without a directed-rounding guarantee; all unresolved boxes are counted.

Run from any directory, e.g.:
  OPENBLAS_NUM_THREADS=1 python examples/baseline/certify_cube_probability_cpu.py
The analysis partition is not the uniform-grid greedy search comparator.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def vertices(d):
    ids=np.arange(2**(d-1),dtype=np.uint64)[:,None]
    bits=((ids >> np.arange(d-1,dtype=np.uint64)) & 1).astype(float)
    return np.concatenate((np.ones((len(ids),1)),2*bits-1),axis=1)

def certify(H,M,m,T=100,batch=4096,omega=.5,gamma=.9):
 d=len(H);A=np.eye(d)+H/np.linalg.norm(H);Ap=np.maximum(A,0);Am=np.minimum(A,0)
 times=[0,1,5,10,20,50,100];counts=np.zeros(len(times),dtype=np.int64)
 total=m**d
 for st in range(0,total,batch):
  ids=np.arange(st,min(st+batch,total),dtype=np.int64)[:,None]
  ix=(ids//(m**np.arange(d,dtype=np.int64)))%m
  lo=-1+2*ix/m;hi=lo+2/m
  hit=np.zeros(len(lo),dtype=bool)
  for t in range(T+1):
   mid=(hi+lo)/2;rad=(hi-lo)/2;hm=mid@H
   # J(z)>=2 v^THz-v^THv, v=mid, exact min of affine on box.
   lower=np.sum(mid*hm,1)-2*np.sum(np.abs(hm)*rad,1)
   hit|=lower>=gamma**2*M+1e-12*M
   if t in times:counts[times.index(t)]+=np.sum(hit)
   if t<T:
    a=lo@Ap.T+hi@Am.T;b=hi@Ap.T+lo@Am.T
    lo=(1-omega)*lo+omega*np.clip(a,-1,1)
    hi=(1-omega)*hi+omega*np.clip(b,-1,1)
  if st%65536==0:print('progress',d,m,st,total,flush=True)
 return {'m':m,'cells':total,'times':times,'certified_cells':counts.tolist(),'lower':(counts/total).tolist()}


def upper(H,target,maxdepth=36,maxcells=3000000):
 d=len(H);c=np.zeros((1,d));r=np.ones((1,d));mass=np.ones(1);inside=0.;out=[]
 for depth in range(maxdepth+1):
  hc=c@H;v=np.sum(c*hc,1);lin=2*np.sum(np.abs(hc)*r,1)
  lo=v-lin
  hi=v+lin+np.sum((r@np.abs(H))*r,1)
  good=lo>=target;inside+=mass[good].sum()
  keep=(hi>=target)&~good
  c=c[keep];r=r[keep];mass=mass[keep]
  ub=inside+mass.sum();out.append({'depth':depth,'cells':len(c),'upper':float(ub),'lower':float(inside)})
  if depth%4==0:print('DEPTH',d,depth,len(c),ub,flush=True)
  if len(c)==0 or len(c)>maxcells or depth==maxdepth:break
  hc=c@H;score=r*(np.abs(hc)+r@np.abs(H))
  idx=np.argmax(score,1);r[np.arange(len(c)),idx]/=2
  delta=np.zeros_like(c);delta[np.arange(len(c)),idx]=r[np.arange(len(c)),idx]
  c=np.concatenate([c-delta,c+delta]);r=np.concatenate([r,r]);mass=np.concatenate([mass/2,mass/2])
 return out

def main():
    repo=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=repo/'runs/cube_cpu_20261015')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--dimensions',type=int,nargs='+',default=[4,6,8,10,12])
    args=parser.parse_args()
    output=args.output_dir or args.run_dir/'theory'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in args.dimensions:
        source=args.run_dir/f'frozen_cube_d{d}_n{d//2}.npz'
        H=np.load(source)['H'];V=vertices(d)
        M=float(np.max(np.einsum('bi,ij,bj->b',V,H,V)))
        m={4:8,6:4,8:4,10:4,12:3}[d]
        row=certify(H,M,m)
        audit=upper(H,.81*M)
        bound=audit[-1]['upper']
        row.update({'d':d,'rank':d//2,'file':source.name,'gamma':.9,'omega':.5,
          'direct_probability_upper':bound,
          'pool_upper':{str(size):float(-np.expm1(size*np.log1p(-bound)))
                         for size in [101,256,4096,10000]},
          'qualification':'Deterministic mathematical bounds, evaluated in float64; not directed-rounding machine verification.',
          'gd_classification_buffer':1e-12*M,
          'direct_upper_classification_buffer':0.0})
        rows.append(row)
        (output/'cube_certified_comparison.json').write_text(json.dumps(rows,indent=2))
        (output/f'direct_upper_d{d}.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    main()
