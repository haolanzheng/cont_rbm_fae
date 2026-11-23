"""Replace only the HF diffusion solve by nodal sampling of its analytic solution."""
import numpy as np
from .continuous_fem import FEMProblem

def build_model(d):
    p=FEMProblem.build(n_grid=40,d=d,contrast=.25,spectrum_ratio=.78,element_order=3,analytic_snapshots=True)
    A,B,S=p.A,p.B,p.S
    error=B-A@S
    W_B=p.solve_X(B);W_error=p.solve_X(error)
    cross=S.T@B
    data={'G0':p.solution_gram,'H0':B.T@W_B,'cross':cross,
          'residual_error_cross':B.T@W_error,'residual_error_gram':error.T@W_error,
          'widths':p.singular_values,'sampled_snapshots':S,'load_matrix':B,
          'quadrature_order':p.quadrature_order,'reference_dofs':p.dofs,
          'sampled_solution_energy_error':p.projection_error,
          'relative_discrete_equation_defect':float(np.linalg.norm(error)/np.linalg.norm(B)),
          'relative_cross_difference':float(np.linalg.norm(cross-p.solution_gram)/np.linalg.norm(p.solution_gram))}
    # Verify the coefficient formula against the unchanged full Galerkin equations.
    from .geometry_cpu import CoordinateROM,rom_options
    rom=CoordinateROM(data['G0'],data['H0'],**rom_options(data))
    for z in np.random.default_rng(42).normal(size=(min(3,d-1),d)):rom.append(z)
    V=S@rom.Q;R=B-A@V@(V.T@B);E=S-V@(V.T@B)
    H,G=rom.matrices();Href=R.T@p.solve_X(R);Gref=E.T@(A@E)
    data['residual_matrix_check']=float(np.linalg.norm(H-Href)/np.linalg.norm(Href))
    data['error_matrix_check']=float(np.linalg.norm(G-Gref)/np.linalg.norm(Gref))
    assert data['residual_matrix_check']<1e-10 and data['error_matrix_check']<1e-10
    return data
