"""Independent-pool, fixed-space direction check for natural source variants."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from natural_variants import model,OUT,SEEDS
from run import append,ascend


def main():
    rows=[]
    for d in [64,144]:
        m=int(np.sqrt(d));n=d//4
        for variant,basis in [('original','block'),('bilinear_source','hat')]:
            B,G,_=model(m,basis);saved=np.load(OUT/f'd{d}_{variant}_snapshots.npz')
            for seed in SEEDS:
                Q=np.zeros((d,0))
                for z in saved[f'z_s{seed}_M10000'][:n]:Q,_=append(B,Q,z)
                R=B-Q@(Q.T@B);H=R.T@R;largest=np.linalg.eigvalsh(H)[-1]
                pool=np.random.default_rng(982000+d+seed).uniform(-1,1,(100000,d))
                scores=np.sum((pool@R.T)**2,axis=1)
                start=np.random.default_rng(983000+d+seed).uniform(-1,1,d)
                candidates={'gd80':ascend(H,start,80)}
                for M in [10000,100000]:candidates[f'pool{M}']=pool[np.argmax(scores[:M])]
                row=dict(d=d,n=n,variant=variant,seed=seed,metrics={})
                for name,z in candidates.items():
                    hz=H@z;J=z@hz
                    row['metrics'][name]=dict(J=float(J),normalized_mean_error_reduction=float(hz@hz/(J*largest)))
                rows.append(row)
            subset=[r for r in rows if r['d']==d and r['variant']==variant]
            print(d,variant,{name:float(np.median([r['metrics'][name]['normalized_mean_error_reduction'] for r in subset])) for name in candidates},flush=True)
    (OUT/'fixed_space_directions.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
