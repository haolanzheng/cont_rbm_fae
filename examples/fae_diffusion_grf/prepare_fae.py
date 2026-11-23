"""Generate additional coefficient inputs only; fit a train-only POD control."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from run import MaternSampler, coefficient


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    sampler = MaternSampler(128, 1.0, 0.2)
    fields = {}
    seeds = dict(train=20261123, validation=20261124, test=20261125)
    for name, count in [('train', 2048), ('validation', 256), ('test', 256)]:
        fields[name] = coefficient(sampler.draw(np.random.default_rng(seeds[name]), count)).astype(np.float32)
        np.save(out/f'{name}.npy', fields[name])
        print('generated', name, fields[name].shape, flush=True)
    w = np.ones(129); w[[0,-1]] = 0.5
    w = np.outer(w,w).ravel()/128**2
    flat = fields['train'].reshape(2048,-1).astype(np.float64)
    mean = np.mean(flat@w)
    std = np.sqrt(np.mean(((flat-mean)**2)@w))
    center = flat.mean(axis=0)
    x = (flat-center)*np.sqrt(w)
    vals, vec = eigh(x@x.T, subset_by_index=[2048-256,2047])
    vals, vec = vals[::-1], vec[:,::-1]
    basis = x.T@vec/np.sqrt(np.maximum(vals, 1e-20))[None,:]
    stats = {}
    # Architecture/rank selection uses validation only. Test is stored unopened.
    for name in ['train','validation']:
        raw=fields[name].reshape(len(fields[name]),-1).astype(np.float64)
        v=(raw-center)*np.sqrt(w)
        coord=v@basis
        norm2=(raw*raw)@w
        stats[name]={}
        for rank in [32,64,96,128,192,256]:
            pred=(coord[:,:rank]@basis[:,:rank].T)/np.sqrt(w)+center
            err=np.sqrt(((pred-raw)**2@w)/norm2)
            stats[name][rank]={'mean':float(err.mean()),'p95':float(np.quantile(err,.95))}
    np.savez_compressed(out/'pod.npz', center=center.astype(np.float32), basis=basis.astype(np.float32), weights=w)
    metadata={'nu':1.,'length':.2,'grid_nodes':129,'splits':{k:len(v) for k,v in fields.items()},
              'seeds':seeds,'mean':float(mean),'std':float(std),'POD_relative_L2':stats,
              'normalization':'one scalar training mean/std; no per-field normalization',
              'input':'a=exp(log(10)*tanh(g)); no PDE solves for additional FAE data',
              'test_policy':'test not used for POD rank selection'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(4): main(a.output)
