"""Unit-L2-load Poisson problem on the unit square with a parameter cube.

Modes are 2 sin(k pi x) sin(l pi y), 1 <= k,l <= m. The (1,1)
coefficient is fixed to one before normalizing the load. Exact solution
coordinates in the energy-orthonormal basis have weights 1/sqrt(lambda).
"""
import numpy as np
from .normalized_load import NormalizedLoad


class NormalizedLoad2D(NormalizedLoad):
    def __init__(self, m):
        if m < 2:
            raise ValueError('At least two modes per spatial direction required')
        self.m = int(m)
        self.modes = np.array(sorted(
            [(k, l) for k in range(1, m+1) for l in range(1, m+1)],
            key=lambda pair: (pair[0]**2+pair[1]**2, pair)))
        self.eigenvalues = np.pi**2 * np.sum(self.modes**2, axis=1)
        self.d = len(self.modes)-1
        self.weights = 1/np.sqrt(self.eigenvalues)
