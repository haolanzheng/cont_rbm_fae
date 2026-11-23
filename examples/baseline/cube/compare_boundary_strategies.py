"""Same-start frozen-space and complete-greedy comparisons of box updates."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from .boundary_strategies import METHODS,search
from .search import search as previous_search,scores
from ..geometry_cpu import CoordinateROM,rom_options
from ..common.box_bounds import maximum
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'runs/analytic_hf_a06_t40_20261022'
CUBE=ROOT/'runs/cube_slides_a06_t80_20261102'


def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def validate():
    H=np.array([[1.,.5],[.5,1.]])
    z=np.array([1.,-1.]);a,_=search(H,z,'clip');b,_=search(H,z,'clip_coordinates')
    assert np.array_equal(a,z) and np.isclose(b@H@b,3.)
    rng=np.random.default_rng(42);B=rng.normal(size=(5,5));H=B.T@B;z=rng.uniform(-1,1,5)
    old,_=previous_search(H,z,alpha=.6,T=80);new,_=search(H,z,'clip');assert np.allclose(old,new,atol=1e-11)
    for method in METHODS:
        b,c=search(H,z,method);assert np.max(np.abs(b))<=1+1e-12 and b@H@b>=z@H@z-1e-10
        assert c['min_increment']>=-1e-10
    return {'clip_matches_existing':True,'boundary_local_maximum_flip_test':True,'all_methods_feasible_and_best_return_nondecreasing':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    cfg={'dimensions':[6,8,16,32,48,64],'seeds':list(range(1000,1008)),'frozen_starts':64,'alpha':.6,'updates':80,'relaxation':.5,'step_multipliers':[.25,.5,1,2,4,8,16],'coordinate_sweeps':10,'methods':METHODS,'cpu':platform.platform(),'threads':1,'metric':'fixed vertex-test maximum / exact analytic width','timing':'search, residual matrix, enrichment; excludes test evaluation','frozen_space':'prefix N-1 of saved cube clipping run seed1000, same H for every strategy','fairness':'Same gradient-iteration cap, not equal cost; H actions and coordinate scores logged separately.'}
    dump(out/'config.json',cfg);dump(out/'validation.json',validate())
    records=[];frozen=[]
    if (out/'greedy.json').exists():records=json.loads((out/'greedy.json').read_text())
    if (out/'frozen.json').exists():frozen=json.loads((out/'frozen.json').read_text())
    done={(r['d'],r['seed'],r['method']) for r in records};fdone={(r['d'],r['start'],r['method']) for r in frozen}
    for d in cfg['dimensions']:
        a=np.load(SOURCE/f'model_d{d}.npz');G0,H0=a['G0'],a['H0'];opts=rom_options(a);N=d//2;w=np.sqrt(sum(.78**(2*j) for j in range(N,d)))
        Z=np.load(CUBE/f'test_vertices_d{d}.npz')['parameters']
        rom=CoordinateROM(G0,H0,**opts);Q=np.load(CUBE/f'basis_d{d}_gd_s1000.npz')['Q'];rom.Q=Q[:,:N-1];H,_=rom.matrices();ml,mu=maximum(H)
        starts=np.random.default_rng(202611020+d).uniform(-1,1,(cfg['frozen_starts'],d));np.savez_compressed(out/f'frozen_d{d}.npz',H=H,starts=starts,maximum_interval=[ml,mu])
        order=np.random.default_rng(825+d)
        for i,z0 in enumerate(starts):
            for method in order.permutation(METHODS):
                method=str(method)
                if (d,i,method) in fdone:continue
                begin=time.perf_counter();z,stats=search(H,z0,method);ms=1000*(time.perf_counter()-begin);value=float(z@H@z)
                frozen.append({'d':d,'start':i,'method':method,'value':value,'quality_interval':np.sqrt([value/mu,value/ml]).tolist(),'wall_ms':ms,'stats':stats})
        dump(out/'frozen.json',frozen);print('FROZEN',d,'done',flush=True)
        for seed in cfg['seeds']:
            for method in order.permutation(METHODS):
                method=str(method)
                if (d,seed,method) in done:continue
                rng=np.random.default_rng(seed);rom=CoordinateROM(G0,H0,**opts);rounds=[];elapsed=0
                for n in range(N):
                    z0=rng.uniform(-1,1,d);begin=time.perf_counter();H,_=rom.matrices();z,stats=search(H,z0,method);rom.append(z);elapsed+=time.perf_counter()-begin
                    rounds.append({'n':n,'parameter':z.tolist(),'stats':stats})
                _,G=rom.matrices();err=float(np.sqrt(max(0,scores(G,Z).max())))
                records.append({'d':d,'N':N,'seed':seed,'method':method,'test_max_error':err,'ratio':err/w,'wall_ms':elapsed*1000,'h_actions':sum(r['stats']['h_actions'] for r in rounds),'coordinate_endpoint_scores':sum(r['stats']['coordinate_endpoint_scores'] for r in rounds),'rounds':rounds,'orthogonality_defect':rom.orthogonality_defect()})
                np.savez_compressed(out/f'basis_d{d}_{method}_s{seed}.npz',Q=rom.Q)
            dump(out/'greedy.json',records);print('GREEDY',d,seed,'done',flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
