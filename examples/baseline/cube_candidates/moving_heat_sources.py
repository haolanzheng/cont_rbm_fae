"""Fixed-power moving triangular heat sources for a Poisson cube benchmark.

-u''=d^{-1} sum_j phi_h(x-s_j), u(0)=u(1)=0,
phi_h(t)=(1-|t|/h)_+/h, s_j=h+(1-2h)z_j, z in [0,1]^d.
Energy-orthonormal sine coordinates are analytic. Their truncation is the
only spatial approximation. The exact solution set lies in the convex hull
of single-source solutions and contains all single-source solutions.
Consequently its worst projection error equals a one-dimensional supremum.
"""
import numpy as np

class MovingHeatSources:
    def __init__(self,d,width=.1,modes=128):
        self.d=int(d);self.width=float(width);self.modes=int(modes)
        self.freq=np.pi*np.arange(1,modes+1)
        self.weights=np.sqrt(2)*np.sinc(np.arange(1,modes+1)*width/2)**2/self.freq
        self.derivative_weights=self.weights*self.freq*(1-2*width)
    def single(self,z):
        s=self.width+(1-2*self.width)*np.asarray(z)
        return np.sin(s[...,None]*self.freq)*self.weights
    def solution(self,z):return self.single(z).mean(axis=-2)
    def objective(self,z,q,gradient=False):
        z=np.asarray(z);u=self.solution(z);r=u-(u@q)@q.T
        value=np.sum(r*r,axis=-1)
        if not gradient:return value
        s=self.width+(1-2*self.width)*z
        jac=np.cos(s[...,None]*self.freq)*self.derivative_weights/self.d
        grad=2*np.sum(jac*r[...,None,:],axis=-1)
        return value,grad
    def normalizer(self,q):
        s=self.width+(1-2*self.width)*np.linspace(0,1,257)
        jac=np.cos(s[:,None]*self.freq)*self.derivative_weights
        r=jac-(jac@q)@q.T
        return max(float(np.max(np.sum(r*r,axis=1))/self.d),1e-16)
    def search(self,q,z,steps=80,alpha=.6):
        z=np.array(z,copy=True);ell=self.normalizer(q);best=z.copy();bestval=-np.inf
        for t in range(steps+1):
            val,g=self.objective(z,q,True)
            if val>bestval:bestval=float(val);best=z.copy()
            if t<steps:
                g=g/ell;z=np.clip(z+alpha*g/np.sqrt(1+np.sum(g*g)),0,1)
        return best,bestval
    def append(self,q,z):
        u=self.solution(z);r=u.copy()
        for _ in range(2):r-=q@(q.T@r)
        norm=np.linalg.norm(r)
        if norm<1e-13:return q
        return np.column_stack((q,r/norm))
