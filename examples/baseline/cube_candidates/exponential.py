"""Analytic diffusion with positive, exponentially parameterized modal forcing.

On (0,1), psi_j=sqrt(2)*sin(j*pi*x)/(j*pi) are H1_0 orthonormal.
The forcing is sum_j weights_j*exp(kappa*A_j z)*(-psi_j''), A=Hadamard/d.
The exact solution has precisely these modal coefficients; no HF solve.
"""
import numpy as np
from scipy.linalg import hadamard


class ExponentialDiffusion:
    def __init__(self,d,rho=.9,kappa=4.):
        self.d=d;self.rho=rho;self.kappa=kappa
        self.A=hadamard(d).astype(float)/d
        self.weights=rho**np.arange(d)

    def theta(self,Z):
        return np.exp(self.kappa*np.asarray(Z)@self.A.T)

    def solution(self,Z):
        return self.theta(Z)*self.weights

    def matrix(self,Q):
        B=np.diag(self.weights)-Q@(Q.T*self.weights)
        return B.T@B

    def objective(self,H,Z,gradient=False):
        th=self.theta(Z);ht=th@H
        val=np.sum(th*ht,axis=-1)
        if not gradient:return val
        return val,2*self.kappa*(th*ht)@self.A

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
        length=np.linalg.norm(w)
        if length<1e-13:raise ValueError('Dependent snapshot')
        return np.column_stack((Q,w/length))

    def corners(self):
        return self.A*self.d
