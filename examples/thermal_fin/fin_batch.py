"""Run the agreed smaller-rank fin suite; independent single-thread jobs."""
import os,sys,json,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

def job(item):
    d,n,m,s=item;out=Path(f'fin_d{d}');target=out/f'{m}_{s}.json'
    if target.exists() and json.loads(target.read_text())['rank']==n:return ('existing',item)
    if not (out/'test_meta.npz').exists():
        raise RuntimeError(f'Missing {out}/test_meta.npz: first run fin_local_study.py --task test --d {d} --out {out}')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with open(out/f'{m}_{s}.log','w') as log:
        p=subprocess.run([sys.executable,'-m','examples.thermal_fin.fin_local_study','--task','run','--d',str(d),'--rank',str(n),'--method',m,'--seed',str(s),'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
    return ('ok' if p.returncode==0 else 'FAILED',item)

if __name__=='__main__':
    methods=['pool2000','pool4000','local','lbfgs5','lbfgs10','lbfgs20','lbfgs100','adam100','resamp2000']
    jobs=[(d,n,m,s) for d,n in [(11,50),(21,80),(33,100)] for s in [11,22,33] for m in methods]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(job,x) for x in jobs]):print(f.result(),flush=True)
