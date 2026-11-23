"""Numerical enclosures for box quadratic maxima and Kolmogorov widths.

The SDP dual is evaluated with an explicit eigenvalue feasibility correction.
Bounds are mathematical bounds evaluated in float64, not interval certificates.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .box.search import scores

def vertex_bounds(H,exact_dimension=16):
    """Max of PSD quadratic is attained at a vertex. Return lower/upper bounds."""
    d=len(H);upper=float(np.abs(H).sum())
    if d<=exact_dimension:
        best=-np.inf;zbest=None
        for st in range(0,1<<(d-1),4096):
            ids=np.arange(st,min(st+4096,1<<(d-1)),dtype=np.int64)
            Z=np.ones((len(ids),d));Z[:,1:]=2*((ids[:,None]>>np.arange(d-1))&1)-1
            v=scores(H,Z);j=v.argmax()
            if v[j]>best:best=float(v[j]);zbest=Z[j].copy()
        return best,best,zbest
    # Every returned vertex gives a rigorous mathematical lower bound.
    rng=np.random.default_rng(6283+d);Z=np.sign(rng.standard_normal((128,d)));Z[0]=1
    _,vec=np.linalg.eigh(H);Z[1]=np.where(vec[:,-1]>=0,1.,-1.)
    for _ in range(50):
        changes=0
        for i in range(d):
            h=Z@H[:,i]-H[i,i]*Z[:,i];new=np.where(h>=0,1.,-1.)
            changes+=np.count_nonzero(new!=Z[:,i]);Z[:,i]=new
        if not changes:break
    v=scores(H,Z);i=v.argmax()
    return float(v[i]),upper,Z[i]




def maximum(H):
    H=(H+H.T)/2
    lo,hi,s=vertex_bounds(H)
    if len(H)<=16 or hi-lo<1e-10*max(hi,1e-300):return float(lo),float(hi)
    scale=np.linalg.norm(H,2);A=H/scale;d=len(H)
    # At a maximizing sign, diag(s*(Hs))-H may already certify optimality.
    y=s*(A@s);gap=np.linalg.eigvalsh(np.diag(y)-A)[0]
    upper=min(hi,scale*(y.sum()+d*max(0.,-gap)))
    if upper-lo<1e-7*lo:return float(lo),float(upper)
    y=np.abs(A).sum(1)+1
    for mu in [1.,.2,.04,.008,.0016,.00032,.000064,.0000128,.00000256,.000000512]:
        for _ in range(45):
            Q=np.diag(y)-A
            factor=cho_factor(Q,lower=True)
            inv=cho_solve(factor,np.eye(d));g=1-mu*np.diag(inv)
            direction=np.linalg.solve(mu*inv*inv,-g)
            decrement=-g@direction
            if decrement<1e-10:break
            f=y.sum()-2*mu*np.log(np.diag(factor[0])).sum();step=1.
            for _ in range(50):
                trial=y+step*direction
                try:
                    L=np.linalg.cholesky(np.diag(trial)-A)
                    value=trial.sum()-2*mu*np.log(np.diag(L)).sum()
                    if value<=f+.01*step*(g@direction):break
                except np.linalg.LinAlgError:pass
                step*=.5
            else:break
            y=trial
    defect=max(0.,-np.linalg.eigvalsh(np.diag(y)-A)[0])
    upper=min(upper,float(scale*(y.sum()+d*(defect+1e-12))))
    return float(lo),float(max(lo,upper))


def width(G,N):
    """Lower: Rademacher covariance. Upper: a concrete mode subspace.

    For every rank-N orthogonal projector, average squared error on vertices
    is at least sum_{j>N} lambda_j(G). The first N snapshots supply an upper
    bound, using max_z z^T R z <= sum_ij |R_ij|.
    """
    values=np.linalg.eigvalsh(G)[::-1]
    lower=values[N:].sum()
    R=G-G[:,:N]@np.linalg.solve(G[:N,:N],G[:N,:])
    upper=np.abs(R[N:,N:]).sum()
    return float(np.sqrt(max(0,lower))),float(np.sqrt(max(lower,upper)))
