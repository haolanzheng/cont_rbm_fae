"""Fixed-pool energy errors with rank-one updates and stable refreshes.

C maps parameters to orthonormal energy coordinates. New basis vectors q
must be orthonormal in those coordinates. Scores omit the roundwise common
normalizer, which does not affect selection. Refreshes prevent cancellation
when the remaining error is many orders below the initial energy.
"""
import numpy as np


class CachedEnergyPool:
    def __init__(self, parameters, C, refresh_ratio=1e-6):
        self.parameters = parameters
        self.residuals = parameters @ C.T
        self.scores = np.einsum('ij,ij->i', self.residuals, self.residuals)
        self.reference_max = float(self.scores.max())
        self.refresh_ratio = refresh_ratio
        self.pending = []
        self.refresh_count = 0

    def select(self):
        return self.parameters[int(np.argmax(self.scores))].copy()

    def update(self, q):
        self.scores -= (self.residuals @ q)**2
        self.pending.append(q.copy())
        if float(self.scores.max()) <= self.refresh_ratio * self.reference_max:
            self.refresh()

    def refresh(self):
        if self.pending:
            Q = np.column_stack(self.pending)
            # Reorthogonalization limits loss of accuracy at very small errors.
            for _ in range(2):
                self.residuals -= (self.residuals @ Q) @ Q.T
            self.pending.clear()
        self.scores = np.einsum('ij,ij->i', self.residuals, self.residuals)
        self.reference_max = float(self.scores.max())
        self.refresh_count += 1
