import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from examples.baseline.cube_candidates.filtered_conductivity import FilteredConductivity
P=Path(__file__).resolve().parents[3]/'runs/filtered_conductivity_20261111'

def main():
    convergence=[];frozen=[]
    for d in [32,64,128]:
        tag=f'd{d}_c10_ell0.2';test=np.load(P/f'{tag}_test.npz')['parameters']
        rng=np.random.default_rng(881+d)
        points=[test[rng.choice(len(test),512,replace=False)]]
        coarse=FilteredConductivity(d,10,.2,256)
        for method in ['gd80','pool10000']:
            data=np.load(P/f'{tag}_{method}_s1000.npz')
            for N in [16,32]:
                q=data['Q'][:,:N-1];err=coarse.values(q,test,kind='relative',chunk=256)
                points.append(test[np.argsort(err)[-12:]])
        common=np.vstack(points)
        for method in ['gd80','pool10000']:
            selected=np.load(P/f'{tag}_{method}_s1000.npz')['selected']
            for m in [256,512,1024]:
                model=FilteredConductivity(d,10,.2,m);q=np.empty((m,0))
                for n,z in enumerate(selected,1):
                    q=model.append(q,z)
                    if n+1 in [16,32]:
                        err=model.values(q,common,kind='relative',chunk=256)
                        row=dict(d=d,method=method,N=n+1,quadrature=m,E_relative=float(np.sqrt(err.max())))
                        convergence.append(row);print('quadrature',row,flush=True)
        pool=np.random.default_rng(9981).uniform(0,1,(10000,d));initial=np.random.default_rng(4321).uniform(0,1,(64,d))
        for N in [8,16]:
            q=np.load(P/f'{tag}_gd80_s1000.npz')['Q'][:,:N-1]
            ref=float(coarse.values(q,pool,chunk=512).max())
            _,values=coarse.search(q,initial,steps=80)
            row=dict(d=d,N=N,starts=64,pool_size=10000,median_GD_J_over_pool_max=float(np.median(values)/ref),fraction_GD_above_pool_max=float(np.mean(values>ref)),min_GD_J_over_pool_max=float(values.min()/ref))
            frozen.append(row);print('frozen',row,flush=True)
        (P/'audit.json').write_text(json.dumps(dict(quadrature=convergence,frozen=frozen),indent=2)+'\n')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
