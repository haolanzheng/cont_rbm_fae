"""Paired searches on the SAME H and initial candidates, independently of enrichment."""
from .compare_optimizers import *


def main():
    output=OUTPUT/'frozen.json'
    records=json.loads(output.read_text()) if output.exists() else []
    done={(r['d'],r['seed'],r['method'],r['initialization']) for r in records}
    for d in [4,8,16,32,64,128]:
        a=np.load(SOURCE/f'model_d{d}.npz');rom=EnergyROM(a['C'],a['K'])
        rom.Q=np.load(SOURCE/f'basis_cube_d{d}_gd_s1000.npz')['Q'][:,:d//2-1]
        H=rom.matrix()
        for seed in range(1000,1008):
            random=np.random.default_rng(seed+20261115).uniform(-1,1,d)
            pool=np.random.default_rng(seed+1000003).uniform(-1,1,(256,d))
            for init in INITS:
                for method in ('gd','sqp','reflective'):
                    if (d,seed,method,init) in done:continue
                    start=time.perf_counter();z0=random.copy()
                    if init=='pool256':z0=pool[np.argmax(np.sum((pool@H)*pool,axis=1))].copy()
                    z,stats=optimize(H,z0,method);ms=1000*(time.perf_counter()-start)
                    records.append(dict(d=d,n=d//2-1,seed=seed,method=method,initialization=init,wall_ms=ms,initial=z0.tolist(),parameter=z.tolist(),**stats))
            dump(output,records)
        print('FROZEN COMPLETE',d,flush=True)
if __name__=='__main__':main()
