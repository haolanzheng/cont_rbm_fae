"""Analytic diffusion with an odd reciprocal modal forcing map.

Q is normalized Hadamard, a=Qz, theta_j=a_j/(1-kappa*a_j**2/d).
For 0<=kappa<1 and z in the cube the map is smooth, with bounded
denominators. At kappa=0 this is exactly the rotated linear example.
The original squared energy-error objective and clipped update are used.
"""
import numpy as np
from scipy.linalg import hadamard
from .exponential import ExponentialDiffusion


class ReciprocalDiffusion(ExponentialDiffusion):
    def __init__(self,d,rho=.78,kappa=.8):
        self.d=d;self.rho=rho;self.kappa=kappa
        self.A=hadamard(d).astype(float)/np.sqrt(d)
        self.weights=rho**np.arange(d)

    def theta(self,Z):
        a=np.asarray(Z)@self.A.T
        return a/(1-self.kappa*a*a/self.d)

    def objective(self,H,Z,gradient=False):
        a=np.asarray(Z)@self.A.T
        v=self.kappa*a*a/self.d
        th=a/(1-v);ht=th@H
        val=np.sum(th*ht,axis=-1)
        if not gradient:return val
        return val,2*((1+v)/(1-v)**2*ht)@self.A

    def corners(self):
        return self.A*np.sqrt(self.d)
