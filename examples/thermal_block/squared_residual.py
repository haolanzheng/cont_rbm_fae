"""Squared-residual search with the baseline's direct projected update.

J = r^T X^{-1} r / ell_n. By default ell_n is the Frobenius norm
of B_n^T X^{-1} B_n, with B_n=[f,-A_1 v_1,...,-A_d v_n]. It is fixed
during each search. The optional source mode reproduces the preliminary
fixed-source-normalizer experiment; none uses the unscaled squared residual.
No logarithm, Armijo backtracking, or displacement stopping is used.
Historical log runners remain unchanged for reproducibility.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import time
import numpy as np
from scipy import linalg
from .model import RB
from .correlation_prior import sample

class SquaredResidualRB(RB):
    def __init__(self,p,capacity,normalization='gram'):
        super().__init__(p,capacity)
        self.normalization=normalization
        self.normalizer=1. if normalization=='none' else p.ff
    def add(self,u):
        super().add(u)
        if self.normalization=='gram':
            # H=B^T X^-1 B, B=[f,-A_1 v_1,...,-A_d v_n].
            # Sign changes of columns do not change its Frobenius norm.
            self.normalizer=float(np.linalg.norm(self.R.T@self.R,'fro'))
        elif self.normalization not in ('source','none'):raise ValueError(self.normalization)
    def fg(self,z,gradient=True):
        n=self.n;q=self.p.d;c=self.p.coeff(z)
        A=np.einsum('q,qij->ij',c,self.A[:,:n,:n])
        cf=linalg.cho_factor(A,check_finite=False)
        y=linalg.cho_solve(cf,self.fr[:n],check_finite=False)
        res=self.R[:,0]-self.R[:,1:]@(y[:,None]*c).ravel()
        value=float(res@res/self.normalizer)
        if not gradient:return value
        w=(-self.R[:,1:].T@res).reshape(n,q)
        jy=2*(w@c)/self.normalizer
        lam=linalg.cho_solve(cf,jy,check_finite=False)
        jc=2*(y@w)/self.normalizer-np.einsum('i,qij,j->q',lam,self.A[:,:n,:n],y)
        return value,self.p.coefficient_jacobian.T@jc

def search(rb,initial,steps=80,alpha=.6):
    z=initial.copy();best=z.copy();bestv=-np.inf;old=None;minimum_increment=0.
    first_grad=None;last_grad=None
    for t in range(steps+1):
        if t==steps:value=rb.fg(z,False)
        else:value,g=rb.fg(z)
        if value>bestv:bestv=value;best=z.copy()
        if old is not None:minimum_increment=min(minimum_increment,value-old)
        old=value
        if t==steps:break
        gn=float(np.linalg.norm(g))
        if first_grad is None:first_grad=gn
        last_grad=gn
        eta=alpha/np.sqrt(1+g@g)
        z=np.clip(z+eta*g,-1,1)
    return best,dict(iterations=steps,indicator_evaluations=steps+1,objective=bestv,normalizer=rb.normalizer,
                     initial_gradient_norm=first_grad,last_gradient_norm=last_grad,
                     displacement=float(np.linalg.norm(best-initial)),min_increment=minimum_increment)

def train(p,seed,steps,L=None,N=24,normalization='gram'):
    start=time.perf_counter();rng=np.random.default_rng(seed)
    rb=SquaredResidualRB(p,N,normalization);rb.add(p.fom(np.zeros(p.d)));rounds=[]
    for n in range(1,N):
        initial=rng.uniform(-1,1,p.d) if L is None else sample(rng,1,L)[0]
        z,stat=search(rb,initial,steps)
        try:rb.add(p.fom(z))
        except RuntimeError:
            rounds.append(dict(n=n,z=z.tolist(),initial=initial.tolist(),stopped='dependent snapshot',**stat));break
        rounds.append(dict(n=n,z=z.tolist(),initial=initial.tolist(),**stat))
    return rb,time.perf_counter()-start,rounds

def validate(p,normalization='gram'):
    rng=np.random.default_rng(327);rb=SquaredResidualRB(p,4,normalization)
    for z in rng.uniform(-1,1,(4,p.d)):rb.add(p.fom(z))
    z=rng.uniform(-.6,.6,p.d);value,g=rb.fg(z);eps=1e-5
    fd=np.array([(rb.fg(z+eps*np.eye(p.d)[j],False)-rb.fg(z-eps*np.eye(p.d)[j],False))/(2*eps) for j in range(p.d)])
    error=float(np.linalg.norm(g-fd)/max(np.linalg.norm(g),1e-12))
    assert error<1e-5,error
    direct=rb.explicit(z)/rb.normalizer
    assert abs(value-direct)<1e-9*max(value,1e-12)
    logvalue,loggradient=RB.fg(rb,z)
    assert np.allclose(g,value*loggradient,rtol=1e-10,atol=1e-12)
    assert np.isclose(value,np.exp(logvalue)*p.ff/rb.normalizer,rtol=1e-12)
    # The first projected step agrees with the explicitly written update.
    expected=np.clip(z+.6*g/np.sqrt(1+g@g),-1,1)
    selected,stat=search(rb,z,1)
    expectedbest=expected if rb.fg(expected,False)>value else z
    step_error=float(np.max(np.abs(selected-expectedbest)))
    assert step_error<1e-14,step_error
    return dict(gradient_relative_error=error,objective=value,explicit_objective=direct,one_step_verified=True,one_step_roundoff=step_error)
