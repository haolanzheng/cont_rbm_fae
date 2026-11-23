"""Check an analytic bound for every unenumerated cube vertex.

Let A=(I-QQ.T)diag(weights), e=||A 1||, s=||A||_2.  At a vertex
with k negative coordinates, error <= (e+2 sqrt(k) s)/(delta+2k/d).
The run enumerates all k<=3 vertices (all vertices at d16). If the bound
for k>=4 is below the observed maximum, the maximum is attained in the
enumerated subset. This is floating-point validation with a conservative
roundoff allowance, not interval arithmetic.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from .collective import CollectiveDiffusion

OUT=Path(__file__).resolve().parents[3]/'runs/cube_candidate_screen_20261104'
def main():
    data=json.loads((OUT/'collective_refine_results.json').read_text());cert=[]
    for row in data:
        if row['delta']!=.01:continue
        d=row['d'];rho=row['rho'];delta=row['delta']
        Q=np.load(OUT/f"collective_refine_delta{delta}_r{rho}_d{d}_{row['method']}_s{row['seed']}.npz")['Q']
        m=CollectiveDiffusion(d,rho,delta)
        A=np.diag(m.weights)-Q@(Q.T*m.weights)
        allowance=1e-10*np.max(m.weights)
        e=float(np.linalg.norm(A@np.ones(d)))+np.sqrt(d)*allowance
        s=float(np.linalg.norm(A,2))+allowance
        upper=0. if d==16 else max((e+2*np.sqrt(k)*s)/(delta+2*k/d) for k in range(4,d+1))
        passed=upper<row['E_augmented']-np.sqrt(d)*allowance/delta
        cert.append(dict(d=d,rho=rho,delta=delta,seed=row['seed'],method=row['method'],E=row['E_augmented'],unexamined_vertex_upper_bound=upper,passed=bool(passed),enumeration='all vertices' if d==16 else 'all vertices with k<=3 negative coordinates'))
    (OUT/'collective_vertex_certificates.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(dict(total=len(cert),passed=sum(x['passed'] for x in cert),max_upper_ratio=max(x['unexamined_vertex_upper_bound']/x['E'] for x in cert)))
    if not all(x['passed'] for x in cert):raise AssertionError('Certificate failed')

if __name__=='__main__':main()
