"""Conforming P1 thermal block and affine reduced residual evaluation."""
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import splu,spsolve

class ThermalBlock:
 def __init__(self,rows=2,cols=7,cells=28,dense_factor=True):
  assert cells%rows==0 and cells%cols==0
  self.d=rows*cols;self.q=self.d;self.rows=rows;self.cols=cols;self.cells=cells
  self.coefficient_center=np.full(self.d,5.5);self.coefficient_jacobian=4.5*np.eye(self.d)
  x=np.linspace(0,1,cells+1);xx,yy=np.meshgrid(x,x,indexing='ij');points=np.column_stack([xx.ravel(),yy.ravel()])
  tris=[]
  for i in range(cells):
   for j in range(cells):
    k=i*(cells+1)+j;tris.extend([(k,k+cells+1,k+cells+2),(k,k+cells+2,k+1)])
  tris=np.array(tris);p=points[tris];M=np.concatenate([np.ones((*p.shape[:2],1)),p],axis=2)
  grad=np.linalg.inv(M)[:,1:,:];area=np.abs(np.linalg.det(M))/2
  local=area[:,None,None]*np.einsum('tki,tkj->tij',grad,grad)
  cent=p.mean(axis=1);block=(np.minimum((cent[:,1]*rows).astype(int),rows-1)*cols+np.minimum((cent[:,0]*cols).astype(int),cols-1))
  interior=np.flatnonzero(np.all((points>0)&(points<1),axis=1));self.n=len(interior)
  rr=np.broadcast_to(tris[:,:,None],local.shape);cc=np.broadcast_to(tris[:,None,:],local.shape)
  self.parts=[]
  for j in range(self.d):
   mask=block==j;A=sparse.coo_matrix((local[mask].ravel(),(rr[mask].ravel(),cc[mask].ravel())),shape=(len(points),len(points))).tocsc()
   self.parts.append(A[interior,:][:,interior].tocsc())
  f=np.zeros(len(points));np.add.at(f,tris.ravel(),np.repeat(area/3,3));self.f=f[interior]
  self.X=sum(self.parts).tocsc();self.riesz=splu(self.X);self.xf=self.riesz.solve(self.f);self.ff=float(self.f@self.xf)
  if dense_factor:self.L=linalg.cholesky(self.X.toarray(),lower=True);self.wf=linalg.solve_triangular(self.L,self.f,lower=True)
 def coeff(self,z):return self.coefficient_center+np.asarray(z)@self.coefficient_jacobian.T
 def matrix(self,z):return sum(float(c)*a for c,a in zip(self.coeff(z),self.parts)).tocsc()
 def fom(self,z):return spsolve(self.matrix(z),self.f)

class RB:
 def __init__(self,p,capacity):
  self.p=p;self.n=0;self.capacity=capacity;h=p.n;q=p.d
  self.V=np.zeros((h,capacity));self.A=np.zeros((q,capacity,capacity));self.fr=np.zeros(capacity)
  self.D=np.zeros((h,1+capacity*q));self.D[:,0]=p.wf;self.R=None
 def add(self,u):
  p=self.p;n=self.n;q=p.d;v=u.copy();V=self.V[:,:n]
  for _ in range(2):v-=V@(V.T@(p.X@v))
  norm=float(np.sqrt(max(0,v@(p.X@v))))
  if norm<1e-13:raise RuntimeError('Dependent snapshot')
  v/=norm;self.V[:,n]=v;self.fr[n]=v@p.f
  b=np.column_stack([a@v for a in p.parts]);k=n*q
  av=self.V[:,:n+1].T@b;self.A[:,:n+1,n]=av.T;self.A[:,n,:n+1]=av.T;self.n+=1
  self.D[:,1+k:1+k+q]=linalg.solve_triangular(p.L,b,lower=True,check_finite=False)
  self.R=linalg.qr(self.D[:,:1+(n+1)*q],mode='r',check_finite=False)[0][:min(p.n,1+(n+1)*q)]
 def fg(self,z,gradient=True):
  n=self.n;q=self.p.d;c=self.p.coeff(z);A=np.einsum('q,qij->ij',c,self.A[:,:n,:n]);cf=linalg.cho_factor(A,check_finite=False);y=linalg.cho_solve(cf,self.fr[:n],check_finite=False)
  x=(y[:,None]*c).ravel();res=self.R[:,0]-self.R[:,1:]@x;num=float(res@res);den=max(num,self.p.ff*1e-28)
  val=float(np.log(den/self.p.ff))
  if not gradient:return val
  w=(-self.R[:,1:].T@res).reshape(n,q);jy=2*(w@c)/den;lam=linalg.cho_solve(cf,jy,check_finite=False)
  jc=2*(y@w)/den-np.einsum('i,qij,j->q',lam,self.A[:,:n,:n],y)
  return val,self.p.coefficient_jacobian.T@jc
 def explicit(self,z):
  V=self.V[:,:self.n];A=self.p.matrix(z);y=linalg.solve(V.T@(A@V),V.T@self.p.f,assume_a='pos');r=self.p.f-A@(V@y)
  return float(r@self.p.riesz.solve(r))
 def errors(self,Z,U,unorm2):
  n=self.n;V=self.V[:,:n];errs=[]
  for start in range(0,len(Z),512):
   z=Z[start:start+512];ar=np.einsum('bq,qij->bij',self.p.coeff(z),self.A[:,:n,:n]);y=np.linalg.solve(ar,np.broadcast_to(self.fr[:n],(len(z),n))[...,None])[...,0]
   E=U[:,start:start+len(z)]-V@y.T;errs.extend(np.sqrt(np.maximum(np.sum(E*(self.p.X@E),axis=0),0)))
  return np.array(errs)

class Pool:
 def __init__(self,rb,Z):
  self.rb=rb;self.Z=Z;self.c=rb.p.coeff(Z);self.n=0
  self.A=np.empty((len(Z),rb.capacity,rb.capacity))
 def values(self):
  rb=self.rb;n=rb.n;q=rb.p.d
  while self.n<n:
   k=self.n;av=self.c@rb.A[:,:k+1,k];self.A[:,:k+1,k]=av;self.A[:,k,:k+1]=av
   self.n+=1
  y=np.linalg.solve(self.A[:,:n,:n],np.broadcast_to(rb.fr[:n],(len(self.Z),n))[...,None])[...,0]
  values=[]
  for start in range(0,len(y),256):
   x=(y[start:start+256,:,None]*self.c[start:start+256,None,:]).reshape(-1,n*q);res=rb.R[:,0][None,:]-x@rb.R[:,1:].T;values.extend(np.log(np.maximum(np.sum(res*res,axis=1),rb.p.ff*1e-28)/rb.p.ff))
  return np.array(values)
