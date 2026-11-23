"""Common-test-set final errors from saved CPU greedy histories; no retraining."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_info
from examples.baseline.geometry_cpu import CoordinateROM,uniform_parameters
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/geometry_cpu_20261012'
OUT=ROOT/'runs/final_test_errors_20261020'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=json.loads((SOURCE/'results.json').read_text())['runs']
    records=[];summary=[];checks={'max_orthogonality_defect':0.,'max_relative_worst_error_difference':0.}
    for d in [8,16,32,64]:
        data=np.load(SOURCE/f'fem_d{d}.npz');G0,H0=data['G0'],data['H0'];N=d//2
        for domain in ['sphere','ball']:
            test_seed=202610180+100*d+(domain=='ball')
            Z=uniform_parameters(np.random.default_rng(test_seed),16384,d,domain)
            if d==32:
                saved=np.load(ROOT/f'runs/slides_comparison_20261018/fixed_final_spaces_{domain}_d32.npz')
                assert np.array_equal(Z,saved['test_parameters'])
            denominator=np.einsum('ij,ij->i',Z@G0,Z)
            assert np.min(denominator)>0
            np.savez_compressed(OUT/f'test_{domain}_d{d}.npz',test_parameters=Z,test_seed=test_seed)
            methods=['exp_rgd100' if domain=='sphere' else 'relaxed_pg100','fixed_pool256','fixed_pool4096','fixed_pool10000']
            for method in methods:
                selected=[r for r in runs if r['domain']==domain and r['dimension']==d and r['method']==method]
                assert len(selected)==8
                values=[]
                for r in selected:
                    rom=CoordinateROM(G0,H0)
                    for step in r['rounds'][:N]:rom.append(step['parameter'])
                    P=np.eye(d)-rom.Q@(rom.Q.T@G0)
                    residual=Z@P.T
                    numerator=np.einsum('ij,ij->i',residual@G0,residual)
                    assert np.min(numerator)>-1e-12
                    relative_squared=np.maximum(numerator,0)/denominator
                    assert np.max(relative_squared)<1+1e-8
                    error=float(np.sqrt(np.mean(relative_squared)));values.append(error)
                    _,G=rom.matrices();E=float(np.sqrt(max(np.linalg.eigvalsh(G)[-1],0)))
                    diff=abs(E-r['errors'][N])/r['errors'][N]
                    assert diff<1e-7
                    checks['max_relative_worst_error_difference']=max(checks['max_relative_worst_error_difference'],diff)
                    checks['max_orthogonality_defect']=max(checks['max_orthogonality_defect'],rom.orthogonality_defect())
                    records.append({'domain':domain,'d':d,'N':N,'method':method,'training_seed':r['seed'],
                        'test_seed':int(test_seed),'test_count':len(Z),'rms_relative_energy_error':error})
                summary.append({'domain':domain,'d':d,'N':N,'method':method,'median_rms_relative_energy_error':float(np.median(values))})
        print('Completed d=',d,flush=True)
    previous=json.loads((ROOT/'runs/test_error_curves_20261020/curves.json').read_text())
    aliases={'exp_rgd100':'gd','relaxed_pg100':'gd','fixed_pool256':'pool256','fixed_pool4096':'pool4096','fixed_pool10000':'pool10000'}
    for r in summary:
        if r['d']==32:
            old=next(x for x in previous if x['domain']==r['domain'] and x['method']==aliases[r['method']])
            assert abs(r['median_rms_relative_energy_error']-old['median_over_training_seeds'][-1])<1e-12
    meta={'dimensions':[8,16,32,64],'final_N':'d/2','test_count_per_domain_dimension':16384,
        'metric':'sqrt(mean_z(||S z-u_N(z)||_A^2/||S z||_A^2))','aggregation':'median of per-model test RMS over eight training seeds',
        'source':'runs/geometry_cpu_20261012/results.json and fem_d*.npz',
        'test_sets':'Independent of training; shared across all methods and training seeds within each domain and dimension. At d=32 identical to previously saved test sets.',
        'device':'local CPU','threadpools':threadpool_info(),'checks':checks}
    for name,obj in [('per_seed_errors',records),('summary',summary),('metadata',meta)]:
        (OUT/f'{name}.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks))
    for domain in ['sphere','ball']:
        for d in [8,16,32,64]:
            print(domain,d,[round(r['median_rms_relative_energy_error'],9) for r in summary if r['domain']==domain and r['d']==d])

if __name__=='__main__':main()
