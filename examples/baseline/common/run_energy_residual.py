"""Repeat the displayed baseline experiments with the energy-dual residual.

The cached consistent FEM model is unchanged. In A-orthonormal solution
coordinates, R.T A^{-1} R = E.T E, so the residual metric is the identity.
Run from the repository root with the single-thread CPU environment.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
from . import run_consistent_fem as run
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'runs/consistent_fem_20261113'
OUT=ROOT/'runs/energy_residual_20261120'

def setup():
    OUT.mkdir(exist_ok=True)
    run.OUT=OUT
    # Reuse precisely the same HF solutions, coordinates and evaluation points.
    for d in run.DIMS:
        target=OUT/f'model_d{d}.npz'
        if not target.exists():
            source=OLD/target.name
            with np.load(source) as data: a=dict(data)
            assert float(a['relative_discrete_equation_defect'])<1e-10
            a['K']=np.eye(d);a['H0']=a['C'].T@a['C']
            a['residual_metric']=np.array('diffusion_energy_dual')
            np.savez_compressed(target,**a)
        for domain in ('sphere','ball','cube'):
            name=f'test_{domain}_d{d}.npz'
            if (OLD/name).exists() and not (OUT/name).exists():shutil.copy2(OLD/name,OUT/name)
    audit={'residual_metric':'A^{-1}', 'coordinate_metric':'identity',
           'hf_cache':str(OLD),'test_sets':'unchanged from the displayed experiments',
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'provenance.json').write_text(json.dumps(audit,indent=2))
    return run

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    setup()
    if args.validate:
        from . import validate_energy_residual as validation
        validation.main()
    else:
        run.main()
        path=OUT/'config.json';cfg=json.loads(path.read_text());cfg['residual_norm']='R.T A^{-1} R; K=I in energy coordinates';path.write_text(json.dumps(cfg,indent=2))

if __name__=='__main__':main()
