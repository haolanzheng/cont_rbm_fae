"""End-to-end float32 MPS throughput, including the small CPU factorization."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from torch_model import TorchFAE
from fae_common import DEFAULT_CONFIG,load_config,data_metadata,grid,quadrature,write_json

p=argparse.ArgumentParser();p.add_argument('--batches',type=int,nargs='+',default=[16,32,64])
p.add_argument('--data',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=6);args=p.parse_args()
torch.set_num_threads(4)
c=load_config(args.config);t=c['training'];meta=data_metadata(args.data,c)
args.output.parent.mkdir(parents=True,exist_ok=True)
raw=np.load(args.data/'train.npy',mmap_mode='r');n=raw.shape[1];q=n*n
data=(np.asarray(raw)-meta['mean'])/meta['std'];xy=grid(n);w=quadrature(n)
results=[]
for batch in args.batches:
    print('START',batch,flush=True)
    model=TorchFAE(args.initial,meta['mean'],meta['std'],c).to('mps')
    optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],betas=(t['beta1'],t['beta2']),eps=t['epsilon'],foreach=False)
    rng=np.random.default_rng(t['seed']);durations=[]
    try:
        for i in range(args.steps+3):
            torch.mps.synchronize();start=time.perf_counter()
            ids=rng.choice(len(raw),batch,replace=False);idx=rng.choice(q,t['decoder_queries'],replace=True,p=w/w.sum())
            u=torch.tensor(data[ids],device='mps',dtype=torch.float32)
            target=u.reshape(batch,-1)[:,torch.tensor(idx,device='mps')]
            points=torch.tensor(xy[idx],device='mps',dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred,z=model(u,points);loss=.5*(pred-target).square().mean()+t['latent_penalty']*z.square().sum(1).mean()
            loss.backward();gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],foreach=False)
            optimizer.step();value=float(loss.detach().cpu());gn=float(gnorm.detach().cpu())
            torch.mps.synchronize();elapsed=time.perf_counter()-start
            if not np.isfinite([value,gn]).all():raise RuntimeError('nonfinite update')
            if i>=3:durations.append(elapsed)
        row={'batch':batch,'step_seconds':float(np.mean(durations)),
             'samples_per_second':float(batch/np.mean(durations)),
             'seconds_per_epoch':float(len(raw)/batch*np.mean(durations)),
             'hours_configured_epochs_training_only':float(t['epochs']*len(raw)/batch*np.mean(durations)/3600),
             'metal_allocated_GiB':torch.mps.driver_allocated_memory()/1024**3,
             'loss_after_trial':value,'steps_measured':len(durations),'dtype':'float32'}
    except RuntimeError as error:
        row={'batch':batch,'error':str(error)}
    results.append(row);print(json.dumps(row),flush=True)
    write_json(args.output,results)
    del optimizer,model;gc.collect();torch.mps.empty_cache()
