import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
import scipy.optimize as so
import scipy.fft as sf
import json,time
from pathlib import Path

class Model:
 def __init__(self,d=16,m=128,kappa=1.5,p=1.5):
  self.d=d; self.m=m
  x=np.arange(1,m+1)/(m+1); j=np.arange(1,d+1)
  self.F=kappa*np.cos(np.pi*x[:,None]*j)/(j**p)
  lam=np.sqrt(2*(m+1)**2*(1-np.cos(np.pi*np.arange(1,m+1)/(m+1))))
  self.T=sf.dst(np.eye(m),type=1,norm='ortho',axis=0)/lam[:,None]/np.sqrt(m+1)
 def snap(self,z):return np.exp(z@self.F.T)@self.T.T
 def obj(self,z,Q,grad=False):
  f=np.exp(z@self.F.T); y=f@self.T.T; r=y-(y@Q)@Q.T
  J=np.sum(r*r,axis=-1)
  if grad:return J,2*((r@self.T)*f)@self.F
  return J

def enrich(mod,Q,z):
 v=mod.snap(z)
 for _ in range(2):v-=Q@(Q.T@v)
 return np.column_stack([Q,v/np.linalg.norm(v)])

def maximize(mod,Q,rng,starts=3,nit=100):
 best=(-1,None); nfev=0
 for z in rng.uniform(-1,1,(starts,mod.d)):
  def fn(x):
   v,g=mod.obj(x,Q,True);return -v,-g
  r=so.minimize(fn,z,jac=True,bounds=[(-1,1)]*mod.d,method='L-BFGS-B',options={'maxiter':nit,'ftol':1e-13,'gtol':1e-11,'maxls':30})
  nfev+=r.nfev
  if -r.fun>best[0]:best=(-r.fun,r.x.copy())
 return best[1],best[0],nfev

def run(kappa,p,seed,d=16,N=10):
 mod=Model(d=d,kappa=kappa,p=p);rng=np.random.default_rng(seed)
 pool=rng.uniform(-1,1,(10000,d));ys=mod.snap(pool)
 base=np.vstack([rng.uniform(-1,1,(5000,d)),rng.choice([-1.,1.],(5000,d))])
 chain={};ev={};selected={}
 for meth in ['gd','pool']:
  Q=np.zeros((mod.m,0));Q=enrich(mod,Q,np.zeros(d)); seq=[Q];sel=[];ne=0
  for n in range(N):
   if meth=='gd':z,J,nf=maximize(mod,Q,rng,starts=3);ne+=nf
   else:
    rr=ys-(ys@Q)@Q.T;vals=np.sum(rr*rr,axis=1);i=np.argmax(vals);z=pool[i];J=vals[i]
   Q=enrich(mod,Q,z);seq.append(Q);sel.append([float(J),z.tolist()])
  chain[meth]=seq;selected[meth]=sel;ev[meth]=ne
 met=[]
 for n in [2,4,6,8,10]:
  if n>N:continue
  test=[base]
  for meth in chain:
   q=chain[meth][n]
   for _ in range(16):
    z,_,_=maximize(mod,q,rng,starts=1);test.append(z[None])
  test=np.vstack(test);ee={meth:float(np.sqrt(mod.obj(test,chain[meth][n]).max())) for meth in chain}
  met.append({'N':n+1,'error':ee,'ratio':ee['pool']/ee['gd']})
 # same space fixed V3
 q=chain['pool'][2]; z,j,nf=maximize(mod,q,rng,starts=8)
 vals=mod.obj(pool,q);i=np.argmax(vals); zp=pool[i];jp=vals[i]
 qg=enrich(mod,q,z);qp=enrich(mod,q,zp)
 test=[base]
 for qq in [q,qg,qp]:
  for _ in range(32):
   zz,_,_=maximize(mod,qq,rng,starts=1);test.append(zz[None])
 test=np.vstack(test);eb,eg,ep=[float(np.sqrt(mod.obj(test,qq).max())) for qq in [q,qg,qp]]
 return {'kappa':kappa,'p':p,'seed':seed,'d':d,'greedy':met,'evaluations':ev,'same':{'Jratio':float(j/jp),'Jg':float(j),'Jp':float(jp),'Ebefore':eb,'Egd':eg,'Epool':ep,'gain_gd':eb**2-eg**2,'gain_pool':eb**2-ep**2}}
if __name__=='__main__':
 out=Path(__file__).resolve().parents[4]/'runs/alignment_baseline_20261111/candidate_screens/exponential_forcing_screen.json';out.parent.mkdir(parents=True,exist_ok=True);res=[]
 for k,p in [(1,1),(1.5,1),(2,1),(1,1.5),(1.5,1.5),(2,1.5),(1,2),(2,2)]:
  for seed in [101,102]:
   t=time.time();rr=run(k,p,seed);rr['seconds']=time.time()-t;res.append(rr);out.write_text(json.dumps(res,indent=2));print(json.dumps(rr),flush=True)
