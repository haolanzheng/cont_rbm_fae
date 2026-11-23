"""Consistent FEM baseline: solve AS=B, retain the reference-Laplacian residual.

Energy coordinates avoid ill-conditioning from the prescribed spectral decay.
The search algorithms and parameter coordinates are unchanged.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from .continuous_fem import FEMProblem


def build_model(d):
    p=FEMProblem.build(n_grid=40,d=d,contrast=.25,spectrum_ratio=.78,
                       element_order=3,analytic_snapshots=False)
    spectrum=.78**np.arange(d)
    S0=p.S/spectrum;B0=p.B/spectrum
    Gn=S0.T@(p.A@S0);Gn=(Gn+Gn.T)/2
    U=cholesky(Gn,lower=False)
    Hn=B0.T@p.solve_X(B0);Hn=(Hn+Hn.T)/2
    left=solve_triangular(U.T,Hn,lower=True)
    K=solve_triangular(U.T,left.T,lower=True).T;K=(K+K.T)/2
    C=U*spectrum
    defect=p.A@p.S-p.B
    equation_defect=float(np.linalg.norm(defect)/np.linalg.norm(p.B))
    # Check the residual factor in the original FE coordinates.
    rng=np.random.default_rng(42);Q=np.linalg.qr(rng.normal(size=(d,min(3,d-1))))[0]
    E=C-Q@(Q.T@C)
    V=S0@solve_triangular(U,Q,lower=False)
    residual=p.B-p.A@V@(V.T@p.B)
    direct=residual.T@p.solve_X(residual)
    H=E.T@K@E
    check=float(np.linalg.norm(H-direct)/np.linalg.norm(direct))
    assert equation_defect<1e-10 and check<1e-10
    return dict(C=C,K=K,G0=C.T@C,H0=C.T@K@C,solution_snapshots=p.S,load_matrix=p.B,
                widths=np.linalg.svd(C,compute_uv=False),reference_dofs=p.dofs,
                quadrature_order=p.quadrature_order,relative_discrete_equation_defect=equation_defect,
                residual_factor_check=check,normalized_gram=Gn)


class EnergyROM:
    """Galerkin projection in A-orthonormal coordinates, with Euclidean Q."""
    def __init__(self,C,K):
        self.C=C;self.K=K;self.Q=np.empty((len(C),0))
    def error_factor(self):
        return self.C-self.Q@(self.Q.T@self.C)
    def matrix(self):
        E=self.error_factor();H=E.T@self.K@E
        return (H+H.T)/2
    def append(self,z):
        q=self.C@z
        for _ in range(2):q-=self.Q@(self.Q.T@q)
        norm=float(np.linalg.norm(q))
        if norm<=1e-14:raise RuntimeError('Selected snapshot does not enrich the ROM')
        self.Q=np.column_stack((self.Q,q/norm))
        return norm
