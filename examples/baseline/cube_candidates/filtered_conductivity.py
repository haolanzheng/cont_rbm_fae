"""Helmholtz-filtered coefficient diffusion; exact filter, quadrature solutions.

z in [0,1]^d defines cellwise material controls. b-ell^2 b''=z,
b'(0)=b'(1)=0, a=1+(contrast-1)b. The solution of -(a u')'=0,
u(0)=0,u(1)=1 is evaluated by integrated reciprocal conductivity.
Midpoint quadrature discretizes H1 products; no HF linear solve is used.
"""
import numpy as np
from .layered_conductivity import LayeredConductivity

class FilteredConductivity(LayeredConductivity):
    def __init__(self,parameter_dimension,contrast=10,length=.15,quadrature=256):
        super().__init__(quadrature,contrast)
        self.parameter_dimension=int(parameter_dimension);self.length=float(length)
        x=(np.arange(quadrature)+.5)/quadrature
        left=np.arange(parameter_dimension)/parameter_dimension
        right=left+1/parameter_dimension
        x=x[:,None];left=left[None,:];right=right[None,:];ell=self.length
        # Exact integral of the Neumann Green function over each control cell.
        s=np.minimum(np.maximum(x,left),right)
        self.K=(np.cosh((1-x)/ell)*(np.sinh(s/ell)-np.sinh(left/ell))+
                np.cosh(x/ell)*(np.sinh((1-s)/ell)-np.sinh((1-right)/ell)))/np.sinh(1/ell)
        assert self.K.min()>-1e-13
        assert np.max(abs(self.K.sum(axis=1)-1))<1e-12
        self._cached_q=None

    def coefficients(self,z):return 1+self.span*(np.asarray(z)@self.K.T)

    def reduced(self,q,z):
        a=self.coefficients(z)
        if q.shape[1]==0:return np.broadcast_to(self.lift,a.shape),None
        if q is not self._cached_q:
            self._cached_q=q
            self._A0=q.T@q
            self._Aj=self.span*np.einsum('ij,ik,il->jkl',self.K,q,q,optimize=True)
            self._b0=-self.lift@q
            self._bj=-self.span*(self.lift[:,None]*self.K).T@q
        A=self._A0+np.einsum('...j,jkl->...kl',z,self._Aj,optimize=True)
        b=self._b0+np.asarray(z)@self._bj
        c=np.linalg.solve(A,b[...,None])[...,0]
        return self.lift+c@q.T,A

    def objective(self,q,z,gradient=False,kind='residual'):
        if kind=='relative':
            value,g=self.objective(q,z,True,'error')
            u=self.solution(z);a=self.coefficients(z);r=1/a;total=r.sum(axis=-1,keepdims=True)
            norm=np.sum(u*u,axis=-1)
            dn=(2*self.span*r*r/total*(norm[...,None]-np.sqrt(self.d)*u))@self.K
            return (value/norm,(g*norm[...,None]-value[...,None]*dn)/norm[...,None]**2) if gradient else value/norm
        result=super().objective(q,z,gradient,kind)
        if gradient:
            value,g=result
            return value,g@self.K
        return result

    def normalizer(self,q,kind='residual'):
        B=self.K-self.K.mean(axis=0,keepdims=True)
        B=B-q@(q.T@B)
        gram=self.span**2/self.d*(B.T@B)
        ell=max(float(np.linalg.norm(gram)),1e-14*self.span**2)
        if kind!='residual':ell/=((1+self.contrast)/2)**2
        return ell
