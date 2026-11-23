"""Nonlinear forcing for fixed diffusion: reciprocal ridges plus linear modes.

With energy-orthonormal Dirichlet diffusion modes psi_j, prescribe
u(z)=sum_j theta_j(z) psi_j and f(z)=sum_j theta_j(z) L psi_j.
The first d coefficients are beta*z/sqrt(d); the remaining R coefficients
are epsilon/(1+epsilon-v_j.z/d), for orthogonal Hadamard sign rows v_j.
This makes the Galerkin residual dual norm equal the energy error exactly.
"""
import numpy as np
from scipy.linalg import hadamard


class RidgeBackground:
    def __init__(self, d, ridges=4, epsilon=.1, beta=.05):
        self.d = int(d)
        self.ridges = int(ridges)
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        if self.d < 1 or self.d & (self.d-1) or not 1 < self.ridges <= self.d:
            raise ValueError('Use a power-of-two d and 1<R<=d')
        if self.epsilon <= 0 or self.beta <= 0:
            raise ValueError('epsilon and beta must be positive')
        self.directions = hadamard(self.d).astype(float)[:self.ridges]
        self.state_dimension = self.d+self.ridges
        self.center_jacobian = np.vstack((self.beta*np.eye(self.d)/np.sqrt(self.d),
            self.epsilon/(1+self.epsilon)**2*self.directions/self.d))

    def solution(self, z):
        z = np.asarray(z, dtype=float)
        s = z @ self.directions.T/self.d
        denominator = 1+self.epsilon-s
        if np.any(denominator <= 0):
            raise ValueError('Parameter outside the positive-denominator region')
        return np.concatenate((self.beta*z/np.sqrt(self.d), self.epsilon/denominator), axis=-1)

    def append_basis(self, Q, z, tolerance=1e-12):
        u = self.solution(z)
        v = u.copy()
        for _ in range(2):
            v -= Q@(Q.T@v)
        size = np.linalg.norm(v)
        if size <= tolerance*np.linalg.norm(u):
            return Q.copy()
        return np.column_stack((Q,v/size))

    def initial_basis(self):
        return self.append_basis(np.empty((self.state_dimension,0)),np.zeros(self.d))

    def normalizer(self, Q):
        D=self.center_jacobian-Q@(Q.T@self.center_jacobian)
        return max(float(np.linalg.norm(D.T@D,'fro')),1e-16)

    def objective(self,z,Q,gradient=False):
        z=np.asarray(z,dtype=float)
        theta=self.solution(z)
        residual=theta-(theta@Q)@Q.T
        values=np.sum(residual*residual,axis=-1)
        if not gradient:
            return values
        s=z@self.directions.T/self.d
        hp=self.epsilon/(1+self.epsilon-s)**2
        grad=2*self.beta/np.sqrt(self.d)*residual[...,:self.d]
        grad+=2*((residual[...,self.d:]*hp)@self.directions)/self.d
        return values,grad

    def errors_from_solutions(self,theta,Q):
        residual=theta-(theta@Q)@Q.T
        return np.linalg.norm(residual,axis=-1)

    def global_error_upper_bound(self,Q):
        """Analytic supremum bound, evaluated in ordinary float64.

        The linear forcing part has norm <= beta. Every reciprocal
        coefficient belongs to [epsilon/(2+epsilon),1]. The projection
        residual norm is convex in these R coefficients, so its maximum
        over their enclosing box occurs at one of its 2**R vertices.
        This bound does not assume the actual ridge coordinates can reach
        all those vertices simultaneously.
        """
        if self.ridges>20:
            raise ValueError('The explicit box-vertex bound is intended for small R')
        bits=(np.arange(2**self.ridges,dtype=np.uint64)[:,None]
              >>np.arange(self.ridges,dtype=np.uint64))&1
        h=np.where(bits,1.,self.epsilon/(2+self.epsilon))
        theta=np.column_stack((np.zeros((len(h),self.d)),h))
        return float(self.beta+self.errors_from_solutions(theta,Q).max())
