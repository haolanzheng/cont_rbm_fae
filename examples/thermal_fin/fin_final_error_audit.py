"""Recompute final worst-test errors directly in the full finite-element space."""
import json,time
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from examples.thermal_fin.fin_local_study import Fin

METHODS=['pool2000','pool4000','resamp2000','local','lbfgs5','lbfgs10','lbfgs20','lbfgs100','adam100']

def main():
    # Avoid interfering with isolated timing repetitions.
    while not Path('fin_isolated_timings.json').exists() or len(json.loads(Path('fin_isolated_timings.json').read_text()))<12:
        time.sleep(2)
    results=[]
    with threadpool_limits(limits=1):
        for d,n in [(11,50),(21,80),(33,100)]:
            p=Fin(d);out=Path(f'fin_d{d}');z=np.load(out/'test_meta.npz')['z']
            for method in METHODS:
                for seed in [11,22,33]:
                    info=json.loads((out/f'{method}_{seed}.json').read_text());mm=info['metrics'][str(n)]
                    V=np.load(out/f'{method}_{seed}_basis.npz')['V'];zz=z[mm['worst_index']]
                    A=p.matrix_from_coeff(p.coeff(zz));u=p.fom(zz)
                    y=linalg.solve(V.T@(A@V),V.T@p.f,assume_a='pos');e=u-V@y
                    direct=float(np.sqrt((e@(p.X@e))/(u@(p.X@u))))
                    delta=abs(direct-mm['max'])
                    ortho=float(np.linalg.norm(V.T@(p.X@V)-np.eye(n),ord=np.inf))
                    assert delta<1e-8 and ortho<1e-8
                    results.append({'d':d,'rank':n,'method':method,'seed':seed,'direct_max_point_error':direct,
                                    'difference_from_reduced_formula':delta,'orthogonality_inf_norm':ortho})
    data={'runs_checked':len(results),'maximum_error_difference':max(r['difference_from_reduced_formula'] for r in results),
          'maximum_orthogonality_defect':max(r['orthogonality_inf_norm'] for r in results),'records':results}
    Path('fin_final_error_audit.json').write_text(json.dumps(data,indent=2));print(json.dumps({k:v for k,v in data.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
