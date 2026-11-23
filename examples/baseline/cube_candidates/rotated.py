"""Linear diffusion with Hadamard-mixed forcing and decreasing modal energy.

psi_j are energy-orthonormal sine modes.  The exact solution coefficients
are diag(rho**arange(d)) @ R @ z / sqrt(d), where R is orthogonal Hadamard.
The parameter domain remains [-1,1]^d; this changes the forcing family.
"""
import numpy as np
from examples.baseline.cube_paired.model import hadamard, projected_search


class RotatedDiffusion:
    def __init__(self, d, rho=.78):
        self.d = int(d)
        self.rho = float(rho)
        self.R = hadamard(d)/np.sqrt(d)
        self.weights = rho**np.arange(d)
        self.B = self.weights[:,None]*self.R/np.sqrt(d)
        self.G = self.B.T@self.B

    def solution(self, Z):
        return np.asarray(Z)@self.B.T

    def matrix(self, Q):
        B = self.B-Q@(Q.T@self.B)
        H = B.T@B
        return (H+H.T)/2

    def append(self, Q, z):
        w = self.solution(z).copy()
        for _ in range(2):
            w -= Q@(Q.T@w)
        size = np.linalg.norm(w)
        if size < 1e-13:
            return Q, 0.
        return np.column_stack((Q,w/size)),float(size)

    def search(self,Q,z,alpha=.6,T=80):
        H=self.matrix(Q)
        best,stats=projected_search(H,z,alpha=alpha,T=T,checkpoints=())
        return best,stats

    def modal_vertices(self):
        return self.R*np.sqrt(self.d)


def verify():
    model=RotatedDiffusion(16,.78)
    expected=np.diag(model.weights)
    err=np.max(np.abs(model.solution(model.modal_vertices())-expected))
    Q=np.eye(16)[:,:8]
    M=model.matrix(Q)
    vmax=np.max(np.einsum('bi,ij,bj->b',model.modal_vertices(),M,model.modal_vertices()))
    if err>1e-13 or abs(vmax-model.rho**16)>1e-13:
        raise AssertionError('Rotated forcing identities failed')
    return dict(modal_vertex_solution_error=float(err),spectral_reference_error=float(abs(vmax-model.rho**16)))
