"""Fixed diffusion with a common rational amplitude multiplying linear forcing.

theta(z)=z/(1+delta-mean(z)), z in [-1,1]^d, delta>0.
Exact modal solution coefficients are weights*theta, with energy-orthogonal modes.
This is a genuinely nonlinear forcing family, not a changed error indicator.
"""
import numpy as np

class CollectiveDiffusion:
    def __init__(self,d,rho=.9,delta=.1):
        self.d=d;self.rho=rho;self.delta=delta
        self.weights=rho**np.arange(d)/np.sqrt(d)

    def theta(self,Z):
        Z=np.asarray(Z)
        return Z/(1+self.delta-Z.mean(axis=-1,keepdims=True))

    def solution(self,Z):return self.theta(Z)*self.weights

    def matrix(self,Q):
        B=np.diag(self.weights)-Q@(Q.T*self.weights)
        return B.T@B

    def objective(self,H,Z,gradient=False):
        Z=np.asarray(Z);a=1+self.delta-Z.mean(axis=-1)
        hz=Z@H;val=np.sum(Z*hz,axis=-1)/a**2
        if not gradient:return val
        return val,2*hz/a[...,None]**2+2*(val/(self.d*a))[...,None]

    def search(self,Q,z,alpha=.6,T=80):
        H=self.matrix(Q);ell=max(np.linalg.norm(H,'fro'),1e-30)
        best=z.copy();bestval=-np.inf
        for t in range(T+1):
            val,g=self.objective(H,z,True)
            if val>bestval:best=z.copy();bestval=float(val)
            if t<T:
                g/=ell;z=np.clip(z+alpha*g/np.sqrt(1+g@g),-1,1)
        return best,bestval

    def append(self,Q,z):
        w=self.solution(z)
        for _ in range(2):w-=Q@(Q.T@w)
        size=np.linalg.norm(w)
        if size<1e-13:raise ValueError('Dependent snapshot')
        return np.column_stack((Q,w/size))
