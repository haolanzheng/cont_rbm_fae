import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
import scipy.linalg as la
import json
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Independently rebuild Galerkin spaces using Cholesky and QR.')
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[4]/'runs/alignment_baseline_20261111/piecewise_forcing')
root=parser.parse_args().results
out=[]
for d,N in [(128,32),(128,48),(256,48),(256,64),(256,128)]:
 dat=np.load(root/f'piecewise_extended_d{d}.npz');h=1/d;c=(np.arange(d)+.5)/d
 # Independent construction from the analytic Dirichlet Green kernel.
 G=h*h*(np.minimum.outer(c,c)-np.outer(c,c)-h*np.eye(d)/6)
 B=la.cholesky(G,lower=False)
 trial=dat['test'];assert np.max(np.abs(trial))<=1
 for seed in range(1000,1004):
  stats={}
  for meth in ['gd80','gd320','pool10000']:
   Z=dat[f'z_s{seed}_{meth}'][:N].T
   Q,_=la.qr(B@Z,mode='economic')
   R=B-Q@(Q.T@B);H=R.T@R
   ev=np.linalg.eigvalsh(H);lb=float(np.linalg.norm(trial@R.T,axis=1).max());ub=float(np.sqrt(d*ev[-1]))
   stats[meth]={'lower':lb,'upper':ub,'normalized_lower':np.sqrt(12)*lb,'normalized_upper':np.sqrt(12)*ub,'orth_defect':float(np.linalg.norm(Q.T@Q-np.eye(N),ord=2)),'H_psd_min':float(ev[0])}
  stats['strict_gap_gd80']=stats['pool10000']['lower']-stats['gd80']['upper'];stats['strict_gap_gd320']=stats['pool10000']['lower']-stats['gd320']['upper']
  out.append({'d':d,'N':N,'seed':seed,'results':stats})
(root/'piecewise_independent_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
