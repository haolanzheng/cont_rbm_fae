"""Check residual contributions and grid sensitivity for the displayed dimensions.

Uses the same sampled analytic snapshots and original discrete residual norm.
The finer grid is a validation calculation only, not a change to experiments.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
from .run_analytic_baseline import OUT,DIMS,model
from .continuous_fem import FEMProblem
from .geometry_cpu import rom_options,ball_normalized_probability_lower
from .run_extended_comparison_cpu import frozen_space,sphere_probability_lower

def main():
    fine=FEMProblem.build(n_grid=60,d=max(DIMS),analytic_snapshots=True)
    E=fine.B-fine.A@fine.S
    WB=fine.solve_X(fine.B);WE=fine.solve_X(E)
    arrays={'G0':fine.solution_gram,'H0':fine.B.T@WB,'cross':fine.S.T@fine.B,
            'residual_error_cross':fine.B.T@WE,'residual_error_gram':E.T@WE}
    rows=[]
    for d in DIMS:
        a=model(d);H,Q=frozen_space(a['G0'],a['H0'],d//2,rom_kwargs=rom_options(a))
        M=Q@(Q.T@a['cross']);P=np.eye(d)-M
        contribution=M.T@a['residual_error_gram']@M
        mixed=P.T@a['residual_error_cross']@M;mixed=mixed+mixed.T
        eig=la.eigvalsh(H);top=eig[-1]
        f={k:v[:d,:d] for k,v in arrays.items()}
        Hfine,Qfine=frozen_space(f['G0'],f['H0'],d//2,rom_kwargs=rom_options(f))
        ef=la.eigvalsh(Hfine)
        row={'d':d,'rank':d//2,'grid':40,'check_grid':60,
             'lambda1':float(top),'lambda2_over_lambda1':float(eig[-2]/top),
             'interpolation_contribution_over_lambda1':float(la.norm(contribution,2)/top),
             'mixed_contribution_over_lambda1':float(la.norm(mixed,2)/top),
             'relative_lambda1_change_on_refinement':float(abs(ef[-1]-top)/top),
             'fine_lambda2_over_lambda1':float(ef[-2]/ef[-1]),
             'sphere_bound_coarse':float(sphere_probability_lower(H,.9,100)[0]),
             'sphere_bound_fine':float(sphere_probability_lower(Hfine,.9,100)[0]),
             'ball_bound_coarse':float(ball_normalized_probability_lower(H,.9,100)[0]),
             'ball_bound_fine':float(ball_normalized_probability_lower(Hfine,.9,100)[0])}
        assert row['lambda2_over_lambda1']<.81 and row['fine_lambda2_over_lambda1']<.81
        rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'dimension_validation.json').write_text(json.dumps({'production_grid':40,'validation_grid':60,
      'note':'Numerical grid-sensitivity check, not a certified continuum-error bound. Recomputed exact-indicator reference spaces on each grid.',
      'rows':rows},indent=2)+'\n')
    print('VALIDATION COMPLETE',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
