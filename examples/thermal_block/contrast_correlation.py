"""Separate contrast and variance-matched spatial-correlation experiments.

The correlated family is affine in the same box coordinates. Its coefficient
covariance under uniform independent z is exactly beta**2 K_l / 3. beta and
center are held fixed across l, so pointwise variance is not reduced by
smoothing. Common coefficient bounds are enforced using one normalization
for the entire predeclared length sweep; physical parameter domains differ.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json,copy,time,hashlib
from pathlib import Path
import numpy as np
from .model import ThermalBlock
from .run import train,validate,dump
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/thermal_block_cube_20261103'
OUT=ROOT/'runs/thermal_block_contrast_correlation_20261103';OUT.mkdir(exist_ok=True)
D=14;SEEDS=list(range(1000,1008));METHODS=['gd','pool256','pool4096','pool10000','mixed4096'];CHECKPOINTS=[8,14,16,24]
def maps():
 cases=[]
 for c in (10,30,100,300):cases.append(dict(name=f'contrast_{c}',group='contrast',contrast=c,center=(1+c)/2,jacobian=(c-1)/2*np.eye(D)))
 centers=np.array([[(j+.5)/7,(i+.5)/2] for i in range(2) for j in range(7)])
 distance=np.sum((centers[:,None,:]-centers[None,:,:])**2,axis=2);lengths=(0.,.15,.35);roots=[];kernels=[]
 for length in lengths:
  K=np.eye(D) if length==0 else np.exp(-distance/(2*length**2));val,vec=np.linalg.eigh(K);L=(vec*np.sqrt(np.maximum(val,0)))@vec.T;roots.append(L);kernels.append(K)
 normalization=max(np.max(np.sum(np.abs(L),axis=1)) for L in roots);beta=49.5/normalization
 for length,L,K in zip(lengths,roots,kernels):cases.append(dict(name=f'correlation_{length:g}',group='correlation',length=length,center=50.5,jacobian=beta*L,kernel=K,beta=beta,normalization=normalization))
 return cases

def main():
 original=np.load(BASE/'test_d14.npz');Z=original['Z'];groups=original['groups'];pbase=ThermalBlock(2,7,28);cases=maps()
 serialized=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in cases]
 dump(OUT/'config.json',dict(cases=serialized,seeds=SEEDS,methods=METHODS,N=24,checkpoints=CHECKPOINTS,test_points=len(Z),cells=28,threads=1,gd='same log-residual projected GD, alpha .6, T80, Armijo backtracking',test_points_sha256=hashlib.sha256(Z.tobytes()).hexdigest(),correlation_warning='The three correlation cases share variance and bounds, but are distinct coefficient domains. They are not the full independent [1,100]^14 box.'))
 records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else [];done={(r['case'],r['method'],r['seed']) for r in records}
 for case in cases:
  p=copy.copy(pbase);p.coefficient_center=np.full(D,case['center']);p.coefficient_jacobian=case['jacobian']
  bounds=np.stack([p.coefficient_center-np.sum(np.abs(p.coefficient_jacobian),axis=1),p.coefficient_center+np.sum(np.abs(p.coefficient_jacobian),axis=1)],axis=1)
  v=validate(p);v.update(coefficient_bounds=bounds.tolist(),coefficient_variance=(np.sum(p.coefficient_jacobian**2,axis=1)/3).tolist());dump(OUT/f"validation_{case['name']}.json",v)
  if case['name']=='contrast_10':
   previous=json.loads((BASE/'results.json').read_text())
   for r in previous:
    if r['d']==14 and (case['name'],r['method'],r['seed']) not in done:
     records.append(dict(r,case=case['name'],curve=[x for x in r['curve'] if x['N'] in CHECKPOINTS],reused=True));done.add((case['name'],r['method'],r['seed']))
   dump(OUT/'results.json',records);continue
  path=OUT/f"test_{case['name']}.npz"
  if path.exists():data=np.load(path);U=data['U'];norm2=data['norm2']
  else:
   U=[]
   for i,z in enumerate(Z):
    U.append(p.fom(z))
    if i%4096==0:print('truth',case['name'],i,flush=True)
   U=np.array(U).T;norm2=np.sum(U*(p.X@U),axis=0);np.savez_compressed(path,U=U,norm2=norm2)
  jobs=[(m,s) for m in METHODS for s in SEEDS];np.random.default_rng(328).shuffle(jobs)
  for method,seed in jobs:
   if (case['name'],method,seed) in done:continue
   rb,seconds,rounds=train(p,method,seed,24);rank=rb.n
   np.savez_compressed(OUT/f"basis_{case['name']}_{method}_s{seed}.npz",V=rb.V[:,:rank])
   curve=[]
   for n in CHECKPOINTS:
    if n>rank:continue
    rb.n=n;e=rb.errors(Z,U,norm2);curve.append(dict(N=n,max_error=float(e.max()),max_relative_error=float((e/np.sqrt(norm2)).max()),interior_max=float(e[groups==0].max()),boundary_max=float(e[(groups>0)&(groups<4)].max()),worst_index=int(np.argmax(e))))
   records.append(dict(case=case['name'],d=D,method=method,seed=seed,rank=rank,training_seconds=seconds,rounds=rounds,curve=curve))
   dump(OUT/'results.json',records);print('done',case['name'],method,seed,rank,curve[-1]['max_error'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
