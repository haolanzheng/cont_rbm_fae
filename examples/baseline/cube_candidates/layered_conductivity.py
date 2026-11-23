"""Exact piecewise-constant-conductivity diffusion on (0,1).

Equal cells, a_i=1+(contrast-1)z_i, z in [0,1]^d, u(0)=0,u(1)=1.
Coordinates are cell derivatives / sqrt(d). The lift is u=x, and Q spans
zero-boundary corrections (zero-sum derivative coordinates). Galerkin uses
Q.T diag(a) Q, NOT reference-norm orthogonal projection.
"""
import numpy as np

class LayeredConductivity:
    def __init__(self,d,contrast):
        self.d=int(d);self.contrast=float(contrast);self.span=self.contrast-1
        self.lift=np.ones(self.d)/np.sqrt(self.d)

    def coefficients(self,z):return 1+self.span*np.asarray(z)

    def solution(self,z):
        r=1/self.coefficients(z)
        return np.sqrt(self.d)*r/np.sum(r,axis=-1,keepdims=True)

    def reduced(self,q,z):
        a=self.coefficients(z);batch=a.shape[:-1]
        if q.shape[1]==0:return np.broadcast_to(self.lift,a.shape),None
        A=np.einsum('...i,ij,ik->...jk',a,q,q,optimize=True)
        b=-(a*self.lift)@q
        c=np.linalg.solve(A,b[...,None])[...,0]
        return self.lift+c@q.T,A

    def objective(self,q,z,gradient=False,kind='residual'):
        a=self.coefficients(z);t,A=self.reduced(q,z)
        if kind=='residual':
            r=a*t;r=r-r.mean(axis=-1,keepdims=True)
            value=np.sum(r*r,axis=-1)
            if not gradient:return value
            if q.shape[1]:
                lam=np.linalg.solve(A,((a*r)@q)[...,None])[...,0]
                g=2*self.span*t*(r-lam@q.T)
            else:g=2*self.span*t*r
        elif kind=='error':
            u=self.solution(z);e=u-t;value=np.sum(e*e,axis=-1)
            if not gradient:return value
            inverse=1/a;total=np.sum(inverse,axis=-1,keepdims=True)
            g=inverse**2/total*(np.sum(e*u,axis=-1,keepdims=True)-np.sqrt(self.d)*e)
            if q.shape[1]:
                lam=np.linalg.solve(A,(e@q)[...,None])[...,0]
                g+=t*(lam@q.T)
            g*=2*self.span
        else:raise ValueError(kind)
        return value,g

    def normalizer(self,q,kind='residual'):
        # Frobenius norm of the quadratic residual/error Gram matrix from
        # linearization at z=(1/2,...,1/2), held fixed throughout each search.
        ell=self.span**2/self.d*np.sqrt(max(1,self.d-1-q.shape[1]))
        if kind=='error':ell/=((1+self.contrast)/2)**2
        return ell

    def search(self,q,initial,steps=80,alpha=.6,kind='residual'):
        z=np.asarray(initial,dtype=float).copy();best=z.copy()
        best_value=self.objective(q,z,kind=kind);ell=self.normalizer(q,kind)
        for _ in range(steps):
            _,g=self.objective(q,z,True,kind);g/=ell
            z=np.clip(z+alpha*g/np.sqrt(1+np.sum(g*g,axis=-1,keepdims=True)),0,1)
            value=self.objective(q,z,kind=kind);improve=value>best_value
            best=np.where(improve[...,None],z,best);best_value=np.maximum(best_value,value)
        return best,best_value

    def append(self,q,z):
        v=self.solution(z)-self.lift
        for _ in range(2):v-=q@(q.T@v);v-=v.mean()
        size=np.linalg.norm(v)
        if size<1e-12:raise ValueError('Dependent snapshot')
        return np.column_stack((q,v/size))

    def values(self,q,z,kind='residual',chunk=2048):
        return np.concatenate([np.atleast_1d(self.objective(q,z[i:i+chunk],kind=kind)) for i in range(0,len(z),chunk)])
