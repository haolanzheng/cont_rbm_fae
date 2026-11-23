"""Reevaluate saved bases on one shared vertex test set."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
from ..geometry_cpu import CoordinateROM,rom_options
from .search import scores
from .test_set import make_vertex_test_set


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();out=args.run
    data=json.loads((out/'results.json').read_text());meta=json.loads((out/'config.json').read_text());cfg=data['config']
    backup=out/'results_before_vertex_only.json'
    if not backup.exists():backup.write_bytes((out/'results.json').read_bytes())
    summary={}
    for d in cfg['dimensions']:
        model=np.load(Path(meta['model_cache'])/f'model_d{d}.npz');G0=model['G0'];H0=model['H0']
        vertices,exhaustive=make_vertex_test_set(d,65536,cfg['test_seed']);Z=vertices
        np.savez_compressed(out/f'test_vertices_d{d}.npz',parameters=vertices,exhaustive_up_to_sign=exhaustive)
        summary[str(d)]={'vertex_count':len(vertices),'exhaustive_up_to_sign':exhaustive}
        assert len(np.unique(vertices,axis=0))==len(vertices) and np.all(vertices[:,0]==1)
        for r in [x for x in data['runs'] if x['d']==d]:
            basis=np.load(out/f"basis_d{d}_{r['method']}_s{r['seed']}.npz")['Q'];rom=CoordinateROM(G0,H0,**rom_options(model));curve=[]
            for n in range(r['N']+1):
                rom.Q=basis[:,:n];_,G=rom.matrices();sq=np.maximum(scores(G,Z),0)
                curve.append(float(np.sqrt(sq.max())))
            maximum=float(np.sqrt(max(0,scores(G,vertices).max())))
            r['test_max_error_by_N']=curve
            r.pop('test_error_by_N',None);r['test_max_error']=maximum
            r['test_max_error_ratio']=maximum/r['exact_analytic_width'];r['test_set']='vertices_max';r['test_count']=len(Z);r['vertex_test_count']=len(vertices)
            r.pop('test_max_error_by_stratum_by_N',None)
        print('Evaluated',d,summary[str(d)],flush=True)
    cfg.pop('test_boundary_count',None);cfg['test_vertex_count']=65536;cfg['test_sampling']='vertices_max';cfg['test_count']=65536
    data['test_sampling']={'maximum':'Distinct uniform vertices with z_1=+1; exhaustive when 2^(d-1)<=65536','counts':summary,'initialization':'Training initialization and random pools remain uniform random; unchanged.'}
    (out/'results.json').write_text(json.dumps(data,indent=2)+'\n');(out/'test_sampling.json').write_text(json.dumps(data['test_sampling'],indent=2)+'\n')
if __name__=='__main__':main()
