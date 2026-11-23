"""Correlation-only comparison on one unchanged physical coefficient cube.

A Gaussian copula changes joint spatial dependence while every z_j remains
exactly Uniform[-1,1]. The PDE, full parameter box, marginal coefficient
variance, test set and optimizer all remain unchanged. This studies the
initialization / pool law, not a restriction to a smoother coefficient set.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from .model import ThermalBlock,RB,Pool
from .run import gd,dump
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/thermal_block_cube_20261103';OUT=ROOT/'runs/thermal_block_contrast_correlation_20261103'
METHODS=['gd','pool256','pool4096','pool10000','mixed4096'];SEEDS=list(range(1000,1008));NS=[24]
def sample(rng,count,L,vertices=False):
 x=rng.normal(size=(count,14))@L.T
 return np.where(x>=0,1.,-1.) if vertices else 2*ndtr(x)-1

def train(p,method,seed,L,*,steps=80):
 start=time.perf_counter();rng=np.random.default_rng(seed);rb=RB(p,24);rb.add(p.fom(np.zeros(14)));pool=None
 if method.startswith('pool'):
  Z=sample(np.random.default_rng(seed+1000003),int(method[4:]),L);pool=Pool(rb,Z)
 elif method=='mixed4096':
  prng=np.random.default_rng(seed+1000003);Z=np.vstack([sample(prng,2048,L),sample(prng,2048,L,True)]);pool=Pool(rb,Z)
 rounds=[]
 for n in range(1,24):
  if method=='gd':z,stat=gd(rb,sample(rng,1,L)[0],steps=steps)
  else:
   vals=pool.values();k=int(np.argmax(vals));z=pool.Z[k].copy();stat=dict(log_residual=float(vals[k]),indicator_evaluations=len(vals))
  try:rb.add(p.fom(z))
  except RuntimeError:break
  rounds.append(dict(n=n,z=z.tolist(),**stat))
 return rb,time.perf_counter()-start,rounds

def main():
 orig=np.load(BASE/'test_d14.npz');Z=orig['Z'];groups=orig['groups'];data=np.load(OUT/'test_contrast_100.npz');U=data['U'];norm2=data['norm2']
 p=ThermalBlock(2,7,28);p.coefficient_center=np.full(14,50.5);p.coefficient_jacobian=49.5*np.eye(14)
 centers=np.array([[(j+.5)/7,(i+.5)/2] for i in range(2) for j in range(7)]);dist=np.sum((centers[:,None]-centers[None,:])**2,axis=2)
 records=json.loads((OUT/'prior_results.json').read_text()) if (OUT/'prior_results.json').exists() else [];done={(r['length'],r['method'],r['seed']) for r in records}
 dump(OUT/'prior_config.json',dict(contrast=100,lengths=[0,.15,.35],seeds=SEEDS,methods=METHODS,N=24,T=80,alpha=.6,cells=28,law='z=2*Phi(K_l^(1/2) xi)-1, xi iid N(0,1), K_ii=1; exact uniform marginals',uniform_coefficient_correlation='(6/pi)*arcsin(K_ij/2)',vertex_law='sign(K_l^(1/2) xi)',test='unchanged complete test_d14 parameters and contrast_100 truth',interpretation='same PDE and physical parameter domain; only joint initialization/pool law varies'))
 for length in [0.,.15,.35]:
  K=np.eye(14) if length==0 else np.exp(-dist/(2*length**2));v,w=np.linalg.eigh(K);L=(w*np.sqrt(np.maximum(v,0)))@w.T
  jobs=[(m,s) for m in METHODS for s in SEEDS];np.random.default_rng(617).shuffle(jobs)
  for method,seed in jobs:
   if (length,method,seed) in done:continue
   rb,seconds,rounds=train(p,method,seed,L);rank=rb.n;curve=[]
   np.savez_compressed(OUT/f'basis_prior_{length:g}_{method}_s{seed}.npz',V=rb.V[:,:rank])
   for n in NS:
    if n>rank:continue
    rb.n=n;e=rb.errors(Z,U,norm2);curve.append(dict(N=n,max_error=float(e.max()),max_relative_error=float((e/np.sqrt(norm2)).max()),interior_max=float(e[groups==0].max()),boundary_max=float(e[(groups>0)&(groups<4)].max())))
   records.append(dict(length=length,method=method,seed=seed,rank=rank,training_seconds=seconds,rounds=rounds,curve=curve));dump(OUT/'prior_results.json',records);print('done',length,method,seed,curve[-1]['max_error'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
