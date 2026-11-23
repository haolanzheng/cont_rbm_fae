"""Post-training audit only: witnesses that stationary local maxima are nonglobal.

Witness points are never returned to the training selector. No FOM snapshots
are obtained at the witnesses. Every positive witness is checked using direct
full-space residual/Riesz calculations, without a full-order state solve.
"""
import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from examples.thermal_fin.fin_local_study import Fin,RB,PoolCache,kkt

def audit(d,seed):
    out=Path(f'fin_d{d}');f=out/f'local_{seed}.json'
    while not f.exists():time.sleep(1)
    info=json.loads(f.read_text());n=info['rank'];V=np.load(out/f'local_{seed}_basis.npz')['V'];p=Fin(d);rb=RB(p,n)
    rng=np.random.default_rng(925011+d)
    pts=np.vstack([rng.uniform(-1,1,(4000,d)),rng.choice([-1.,1.],(1024,d))]);pc=PoolCache(rb,pts);records=[]
    start=time.perf_counter()
    for i,old in enumerate(info['records']):
        rb.add(V[:,i]);values=pc.values();ii=int(values.argmax());zz=np.array(old['z']);localval=rb.explicit(zz);wval=rb.explicit(pts[ii])
        r={'rank_before':i+1,'local_indicator_squared_direct':localval,'witness_indicator_squared_direct':wval,
           'witness_ratio':float(wval/localval),'proven_nonglobal_numerically':bool(wval>localval*(1+1e-5)),
           'witness_z':pts[ii].tolist(),'local_max_check':old['numerical_local_max'],'iterations':old['iterations'],
           'rechecked_projected_gradient':kkt(zz,rb.fg(zz)[1])}
        records.append(r)
    data={'d':d,'seed':seed,'rank':n,'witness_count':len(pts),'records':records,'local_max_count':sum(r['local_max_check'] for r in records),
          'nonglobal_count':sum(r['proven_nonglobal_numerically'] for r in records),
          'nonglobal_local_max_count':sum(r['proven_nonglobal_numerically'] and r['local_max_check'] for r in records),'audit_seconds':time.perf_counter()-start}
    (out/f'local_{seed}_witness.json').write_text(json.dumps(data,indent=2));print('AUDIT',d,seed,data['nonglobal_local_max_count'],'/',len(records),'local',data['local_max_count'],flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        for d in [11,21,33]:
            for seed in [11,22,33]:audit(d,seed)
