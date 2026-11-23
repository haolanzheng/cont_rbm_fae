"""Check the energy-coordinate objective against assembled FEM residuals."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,json
from pathlib import Path
from scipy.linalg import cholesky,solve_triangular
from scipy.sparse.linalg import factorized
from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.consistent_fem import EnergyROM
def main():
    p=FEMProblem.build(n_grid=40,d=6,contrast=.25,spectrum_ratio=.78,element_order=3,analytic_snapshots=False)
    root=Path('runs/energy_residual_20261120');a=dict(np.load(root/'model_d6.npz'));C=a['C']
    S0=p.S/(.78**np.arange(6));U=cholesky(S0.T@(p.A@S0),lower=False)
    rom=EnergyROM(C,np.eye(6));rng=np.random.default_rng(1293);checks=[];solve=factorized(p.A.tocsc())
    for n in range(1,6):
     rom.append(rng.normal(size=6));V=S0@solve_triangular(U,rom.Q,lower=False)
     R=p.B-p.A@V@(V.T@p.B);E=p.S-V@(V.T@p.B)
     H=rom.matrix();HA=R.T@solve(R);G=E.T@(p.A@E)
     checks.append(dict(n=n,residual_relative=float(np.linalg.norm(H-HA)/np.linalg.norm(HA)),error_relative=float(np.linalg.norm(H-G)/np.linalg.norm(G))))
    assert max(max(x['residual_relative'],x['error_relative']) for x in checks)<1e-9
    (root/'energy_identity_validation.json').write_text(json.dumps(checks,indent=2));print(checks)

if __name__=='__main__':main()
