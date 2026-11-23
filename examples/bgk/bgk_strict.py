"""Exact-kernel, explicitly scaled inner solve for BGK mechanism controls."""
import ctypes
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import digamma
from examples.bgk.boltzmann_bgk_screen import FixedROM

from functools import lru_cache

@lru_cache(maxsize=1)
def _kernel():
 path=Path(__file__).with_name('bgk_exact_kernel.so')
 if not path.exists():
  raise RuntimeError("BGK kernel not built; run python run.py build-bgk")
 library=ctypes.CDLL(str(path))
 pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 library.bgk_rj.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*4+[pointer]*9
 library.bgk_rj.restype=ctypes.c_int
 return library

class StrictROM(FixedROM):
 def __init__(self,model,snapshots,scale=1000.,tol=1e-12,solver='gn'):
  super().__init__(model,snapshots,inner_solver='least_squares')
  self.scale=scale;self.tol=tol;self.solver=solver;self.solve_records=[]
  self.vx=np.ascontiguousarray(model.v1.ravel());self.vy=np.ascontiguousarray(model.v2.ravel())
  for name in ['base_field','directions','base_transport','direction_transport']:
   setattr(self,name,np.ascontiguousarray(getattr(self,name)))

 def _residual_jacobian(self,y,a,b):
  self.eval_count+=1
  r=np.empty(self.base_field.size);j=np.empty((len(r),self.n-1))
  m=self.model
  bad=_kernel().bgk_rj(m.cfg.nx,m.cfg.nv**2,self.n-1,m.vel_weight,m.weight,float(a),float(b),
       self.vx,self.vy,self.base_field,self.directions,self.base_transport,self.direction_transport,
       np.ascontiguousarray(y),r,j)
  if bad:
   return FixedROM._residual_jacobian(self,y,a,b)
  return r,j

 def solve_inner(self,a,b,warm=False):
  key=(round(float(a),12),round(float(b),12))
  if not warm and key in self.cache:
   z=self.cache[key];return {**z,'y':z['y'].copy(),'gradient':z['gradient'].copy()}
  t=time.perf_counter();nfev=0;used='none';max_condition=1.
  if self.n==1:
   y=np.empty(0);r,j=self._residual_jacobian(y,a,b);status=1;message='no coefficients'
  else:
   start=(self.last_y if warm else self.y0).copy()
   gn_ok=False
   if self.solver=='gn':
    y=start.copy()
    for it in range(40):
     r,j=self._residual_jacobian(y,a,b);nfev+=1
     H=j.T@j;g=j.T@r;eig=np.linalg.eigvalsh(H)
     cond=float(eig[-1]/max(eig[0],1e-300));max_condition=max(max_condition,cond)
     # Normal equations are used only when their conditioning is benign.
     if eig[0]<=0 or cond>1e8:break
     step=np.linalg.solve(H,-g)
     if np.linalg.norm(step)<=5e-13*(1+np.linalg.norm(y)):
      gn_ok=True;break
     if np.max(np.abs(y+step))>=12.:break
     value0=float(r@r);accepted=False;fac=1.
     for backtrack in range(14):
      trial=y+fac*step;tr,tj=self._residual_jacobian(trial,a,b);nfev+=1
      if float(tr@tr)<value0:
       y=trial;accepted=True;break
      fac*=.5
     if not accepted:break
    if gn_ok:
     status=1;message='Gauss-Newton step tolerance';used='gn'
   if not gn_ok:
    used='scipy_trf'
    saved=[None,None,None]
    def ev(yy):
     if saved[0] is None or not np.array_equal(yy,saved[0]):
      rr,jj=self._residual_jacobian(yy,a,b);saved[:]=[yy.copy(),rr*self.scale,jj*self.scale]
     return saved[1],saved[2]
    opt=least_squares(lambda yy:ev(yy)[0],start,
       jac=lambda yy:ev(yy)[1],bounds=(-12.,12.),ftol=self.tol,xtol=self.tol,
       gtol=1e-11,max_nfev=self.model.cfg.inner_maxiter,method='trf')
    y=opt.x;r,j=self._residual_jacobian(y,a,b);status=int(opt.status);message=opt.message;nfev+=int(opt.nfev)
  value=float(r@r);cg=2*j.T@r
  f=self.field(y);q,eq,nu,temp=self.model.collision(f,a,b)
  dqa=(.5*(np.log(4*temp)+digamma(1+.5*a))*nu)[:,None,None]*(eq-f)
  rr=r.reshape(f.shape)/np.sqrt(self.model.weight)
  grad=-2*np.array([np.sum(rr*dqa),np.sum(rr*q/b)])*self.model.weight
  kkt=float(np.max(np.abs(cg))) if len(cg) else 0.
  bound=float(np.max(np.abs(y))) if len(y) else 0.
  ans=dict(value=value,gradient=grad,y=y,success=status>0,nit=nfev,status=status,
    coefficient_gradient_inf=kkt,scaled_coefficient_gradient_inf=kkt*self.scale**2,
    max_abs_coefficient=bound,seconds=time.perf_counter()-t,solver=used,normal_condition=max_condition)
  if bound>11.99:raise RuntimeError('Artificial coefficient box active')
  if not np.isfinite(value) or value>1e8:raise RuntimeError('Unphysical inner solution')
  self.solve_records.append({k:v for k,v in ans.items() if k not in ['gradient','y']})
  if warm:self.last_y=y.copy()
  else:self.cache[key]={**ans,'y':y.copy(),'gradient':grad.copy()}
  return ans
