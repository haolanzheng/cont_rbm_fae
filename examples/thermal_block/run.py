"""Single-CPU comparison on standard thermal-block coefficient boxes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/thermal_block_matplotlib')
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from .model import ThermalBlock,RB,Pool
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/thermal_block_cube_20261103';OUT.mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def validate(p):
 rng=np.random.default_rng(327);rb=RB(p,5)
 for z in rng.uniform(-1,1,(4,p.d)):rb.add(p.fom(z))
 z=rng.uniform(-.6,.6,p.d);val,g=rb.fg(z);eps=1e-5;fd=np.array([(rb.fg(z+eps*np.eye(p.d)[j],False)-rb.fg(z-eps*np.eye(p.d)[j],False))/(2*eps) for j in range(p.d)])
 e=float(np.linalg.norm(g-fd)/max(1,np.linalg.norm(g)));assert e<1e-5,e
 Z=rng.uniform(-1,1,(10,p.d));vals=Pool(rb,Z).values();explicit=np.array([np.log(rb.explicit(z)/p.ff) for z in Z]);ce=float(np.max(abs(vals-explicit)));assert ce<1e-7,ce
 V=rb.V[:,:rb.n];od=float(np.linalg.norm(V.T@(p.X@V)-np.eye(rb.n)));assert od<1e-10
 return dict(gradient_relative_error=e,pool_log_residual_error=ce,basis_orthogonality=od)
def test_data(p):
 path=OUT/f'test_d{p.d}.npz'
 if path.exists():
  a=np.load(path);return a['Z'],a['U'],a['norm2'],a['groups']
 rng=np.random.default_rng(20261103+p.d);inter=2*qmc.Sobol(p.d,scramble=True,seed=20261103+p.d).random_base2(9)-1
 faces=rng.uniform(-1,1,(256,p.d));i=rng.integers(p.d,size=256);faces[np.arange(256),i]=rng.choice([-1.,1.],256)
 edges=rng.choice([-1.,1.],(256,p.d));i=rng.integers(p.d,size=256);edges[np.arange(256),i]=rng.uniform(-1,1,256)
 bits=np.arange(2**p.d)[:,None];vertices=2*((bits>>np.arange(p.d))&1)-1
 Z=np.vstack([inter,faces,edges,vertices,np.zeros((1,p.d))]);groups=np.concatenate([np.full(len(z),i) for i,z in enumerate((inter,faces,edges,vertices,np.zeros((1,p.d))))])
 U=[]
 for k,z in enumerate(Z):
  U.append(p.fom(z))
  if k%4096==0:print('test',p.d,k,'/',len(Z),flush=True)
 U=np.array(U).T;norm2=np.sum(U*(p.X@U),axis=0);np.savez_compressed(path,Z=Z,U=U,norm2=norm2,groups=groups);return Z,U,norm2,groups
def gd(rb,z,steps=80,alpha=.6):
 best=z.copy();bestv=-np.inf;evals=0
 for t in range(steps):
  value,g=rb.fg(z);evals+=1
  if value>bestv:bestv=value;best=z.copy()
  eta=alpha/np.sqrt(1+g@g);moved=False
  for _ in range(12):
   trial=np.clip(z+eta*g,-1,1);step=trial-z
   if np.linalg.norm(step)<1e-9:break
   nv=rb.fg(trial,False);evals+=1
   if nv>=value+1e-4*(g@step):moved=True;break
   eta*=.5
  if not moved:break
  z=trial
  if nv>bestv:bestv=nv;best=z.copy()
 return best,dict(iterations=t+1,indicator_evaluations=evals,log_residual=bestv)
def train(p,method,seed,N,*,steps=80):
 rng=np.random.default_rng(seed);start=time.perf_counter();rb=RB(p,N);rb.add(p.fom(np.zeros(p.d)));pool=None
 if method.startswith('pool'):
  m=int(method[4:]);Z=np.random.default_rng(seed+1000003).uniform(-1,1,(m,p.d));pool=Pool(rb,Z)
 elif method=='mixed4096':
  r=np.random.default_rng(seed+1000003);Z=np.vstack([r.uniform(-1,1,(2048,p.d)),r.choice([-1.,1.],(2048,p.d))]);pool=Pool(rb,Z)
 elif method.startswith('grid'):
  h=float(method[4:]);lev=np.linspace(-1,1,round(2/h)+1);Z=np.stack(np.meshgrid(*([lev]*p.d),indexing='ij'),axis=-1).reshape(-1,p.d);pool=Pool(rb,Z)
 rounds=[]
 for n in range(1,N):
  if method=='gd':z,stat=gd(rb,rng.uniform(-1,1,p.d),steps=steps)
  else:
   values=pool.values();k=int(np.argmax(values));z=pool.Z[k].copy();stat=dict(log_residual=float(values[k]),indicator_evaluations=len(values))
  try:rb.add(p.fom(z))
  except RuntimeError:
   rounds.append(dict(n=n,z=z.tolist(),stopped='dependent snapshot',**stat));break
  rounds.append(dict(n=n,z=z.tolist(),**stat))
 return rb,time.perf_counter()-start,rounds
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dims',nargs='+',type=int,default=[4,14]);ap.add_argument('--seeds',nargs='+',type=int,default=[1000,1001,1002]);ap.add_argument('--rank',type=int,default=24);args=ap.parse_args()
 dump(OUT/'config.json',dict(dimensions=args.dims,seeds=args.seeds,rank=args.rank,cells=28,conductivity_range=[1,10],rhs=1,bc='zero Dirichlet',discretization='conforming P1 triangular FEM',threads=1,device='CPU',platform=platform.platform(),objective='squared residual dual norm for reference unit diffusion; GD maximizes its log',gd=dict(steps=80,alpha=.6,normalized_direction=True,armijo=1e-4,max_backtracks=12),test='512 scrambled Sobol interior + 256 faces + 256 edges + all 2^d vertices + center',norm='reference unit-diffusion energy norm',timing='initial snapshot, offline residual cache, search, enrichment; excludes test evaluation and common assembly',source='https://docs.pymor.org/latest/getting_started.html'))
 records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else [];done={(r['d'],r['method'],r['seed']) for r in records}
 for d in args.dims:
  p=ThermalBlock(2,d//2,28);dump(OUT/f'validation_d{d}.json',validate(p));Z,U,norm2,groups=test_data(p)
  jobs=[(m,s) for m in ['gd','pool256','pool4096','pool10000','mixed4096'] for s in args.seeds]
  if d==4:jobs +=[(m,args.seeds[0]) for m in ['grid1','grid0.5']]
  np.random.default_rng(931+d).shuffle(jobs)
  for method,seed in jobs:
   if (d,method,seed) in done:continue
   rb,seconds,rounds=train(p,method,seed,args.rank);print('trained',d,method,seed,seconds,flush=True)
   np.savez_compressed(OUT/f'basis_d{d}_{method}_s{seed}.npz',V=rb.V[:,:rb.n])
   curve=[]
   for n in range(1,rb.n+1):
    rb.n=n;e=rb.errors(Z,U,norm2);curve.append(dict(N=n,max_error=float(e.max()),max_relative_error=float((e/np.sqrt(norm2)).max()),interior_max=float(e[groups==0].max()),boundary_max=float(e[(groups>0)&(groups<4)].max())))
   # Stable direct norm audit of selected final errors.
   n=rb.n;V=rb.V[:,:n];check=[]
   for k in [int(np.argmax(e)),0,len(Z)-1]:
    A=p.matrix(Z[k]);y=np.linalg.solve(V.T@(A@V),V.T@p.f);err=U[:,k]-V@y;direct=float(np.sqrt(max(err@(p.X@err),0)));check.append(dict(index=k,direct=direct,formula=float(e[k])))
   records.append(dict(d=d,method=method,seed=seed,rank=n,training_seconds=seconds,rounds=rounds,curve=curve,direct_error_check=check))
   dump(OUT/'results.json',records);print('evaluated',d,method,seed,curve[-1]['max_error'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
