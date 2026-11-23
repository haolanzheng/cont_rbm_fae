"""Local float32 FAE training with the shared JAX/Torch experiment contract."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch_model import TorchFAE
from fae_common import DEFAULT_CONFIG,load_config,data_metadata,grid,quadrature,learning_rate,stats,write_json,sha256


def save_checkpoint(path,state):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);tmp.replace(path)


@torch.no_grad()
def evaluate(model,raw,c,device,keep=6):
    model.eval();n=c['data']['grid_nodes'];q=n*n;t=c['training']
    w=quadrature(n);xy=grid(n);errors=[];predictions=[];latents=[]
    for start in range(0,len(raw),t['evaluation_batch_size']):
        batch=np.array(raw[start:start+t['evaluation_batch_size']],dtype=np.float32)
        encoded=model.encode(torch.tensor((batch-model.mean)/model.std,device=device,dtype=torch.float32))
        predicted=np.empty((len(batch),q),dtype=np.float32)
        chunk=t['decoder_evaluation_chunk']
        for first in range(0,q,chunk):
            points=torch.tensor(xy[first:first+chunk],device=device,dtype=torch.float32)
            predicted[:,first:first+len(points)]=model.decode(encoded,points).cpu().numpy()
        flat=batch.reshape(len(batch),q)
        errors.extend(np.sqrt(((predicted-flat)**2@w)/(flat*flat@w)))
        latents.extend(encoded.cpu().numpy())
        if start<keep:predictions.extend(predicted[:keep-start])
    model.train()
    return {'relative_L2':stats(errors),'per_sample':np.asarray(errors).tolist()},np.asarray(predictions).reshape(-1,n,n),np.asarray(latents)


def plot_results(out,history,reference,predicted):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/continuous_greedy_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    fig,ax=plt.subplots(figsize=(5,3.3),constrained_layout=True)
    rows=[r for r in history if 'validation' in r]
    ax.semilogy([r['epoch'] for r in rows],[r['validation']['mean'] for r in rows],'-o',ms=3)
    ax.set(xlabel='Epoch',ylabel='Validation mean relative $L^2$ error');ax.grid(alpha=.2)
    fig.savefig(out/'validation_curve.pdf');fig.savefig(out/'validation_curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(8,7.2),constrained_layout=True)
    for i in range(3):
        panels=[reference[i],predicted[i],np.abs(reference[i]-predicted[i])]
        for j,field in enumerate(panels):
            im=axes[i,j].imshow(field,origin='lower',extent=(0,1,0,1),cmap='viridis',
                                 vmin=0 if j==2 else .1,vmax=None if j==2 else 10)
            if i==0:axes[i,j].set_title(['Input $a$','FAE reconstruction','Absolute error'][j])
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);fig.colorbar(im,ax=axes[i,j],shrink=.85)
    fig.savefig(out/'test_reconstruction.pdf');fig.savefig(out/'test_reconstruction.png',dpi=160);plt.close(fig)


def main(args):
    torch.set_num_threads(args.cpu_threads)
    torch.set_default_dtype(torch.float32)
    c=load_config(args.config);t=c['training'];meta=data_metadata(args.data,c)
    device=torch.device(args.device)
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable.')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:raise RuntimeError('Existing run: pass --resume or use a new directory.')
    initial_sha=sha256(args.initial)
    run={'backend':'torch','device':str(device),'torch_version':torch.__version__,'dtype':'float32',
         'shared_config':c,'initial_sha256':initial_sha,'normalization':{'mean':meta['mean'],'std':meta['std']},
         'data_sha256':{k:sha256(args.data/f'{k}.npy') for k in ('train','validation','test')},
         'small_linear_algebra_device':'cpu' if device.type=='mps' else str(device)}
    if args.resume and json.loads((out/'run.json').read_text())!=run:
        raise RuntimeError('Resume settings/data differ from the saved run.')
    write_json(out/'run.json',run)
    model=TorchFAE(args.initial,meta['mean'],meta['std'],c).to(device)
    assert all(p.dtype==torch.float32 for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],
                               betas=(t['beta1'],t['beta2']),eps=t['epsilon'],weight_decay=0,foreach=False)
    raw=np.load(args.data/'train.npy',mmap_mode='r');rawval=np.load(args.data/'validation.npy',mmap_mode='r')
    data=(np.asarray(raw)-meta['mean'])/meta['std']
    n=c['data']['grid_nodes'];q=n*n;xy=grid(n);w=quadrature(n);batch=t['batch_size']
    if len(raw)%batch:raise ValueError('The shared schedule requires complete training batches.')
    total_steps=t['epochs']*(len(raw)//batch)
    rng=np.random.default_rng(t['seed']);torch.manual_seed(t['seed'])
    history=[];start_epoch=0;global_step=0;best=float('inf');best_epoch=0;elapsed_before=0.
    if args.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch'];global_step=state['global_step'];history=state['history']
        rng.bit_generator.state=state['rng_state'];best=state['best'];best_epoch=state['best_epoch']
        elapsed_before=state['elapsed_seconds']
    print('START',json.dumps({'config':c,'parameters':sum(p.numel() for p in model.parameters()),
                            'device':str(device),'torch':torch.__version__,'start_epoch':start_epoch}),flush=True)
    started=time.perf_counter();epoch=start_epoch
    final_epoch=min(t['epochs'],args.stop_after_epochs or t['epochs'])
    for epoch in range(start_epoch+1,final_epoch+1):
        epoch_start=time.perf_counter();order=rng.permutation(len(raw));losses=[];recs=[]
        model.train()
        for start in range(0,len(order),batch):
            ids=order[start:start+batch];idx=rng.choice(q,t['decoder_queries'],replace=True,p=w/w.sum())
            normalized=data[ids]
            u=torch.tensor(normalized,device=device,dtype=torch.float32)
            target=torch.tensor(normalized.reshape(batch,q)[:,idx],device=device,dtype=torch.float32)
            points=torch.tensor(xy[idx],device=device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred,z=model(u,points)
            rec=.5*(pred-target).square().mean();loss=rec+t['latent_penalty']*z.square().sum(1).mean()
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],foreach=False)
            lr=learning_rate(global_step,total_steps,c)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();global_step+=1
            values=[float(loss.detach().cpu()),float(rec.detach().cpu()),float(norm.detach().cpu())]
            if not np.isfinite(values).all():raise RuntimeError(f'Nonfinite update at epoch {epoch}')
            losses.append(values[0]);recs.append(values[1])
        row={'epoch':epoch,'updates':global_step,'loss':float(np.mean(losses)),
             'reconstruction':float(np.mean(recs)),'learning_rate':lr,'training_seconds':time.perf_counter()-epoch_start}
        if epoch==1 or epoch%t['evaluation_interval_epochs']==0 or epoch==t['epochs']:
            evstart=time.perf_counter();metric,_,_=evaluate(model,rawval,c,device,keep=0)
            row['validation']=metric['relative_L2'];row['validation_seconds']=time.perf_counter()-evstart
            if metric['relative_L2']['mean']<best:
                best=metric['relative_L2']['mean'];best_epoch=epoch
                save_checkpoint(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'validation':metric,'run':run})
        row['elapsed_seconds']=elapsed_before+time.perf_counter()-started
        if device.type=='mps':row['metal_allocated_GiB']=torch.mps.driver_allocated_memory()/1024**3
        history.append(row);write_json(out/'history.json',history)
        save_checkpoint(out/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                         'epoch':epoch,'global_step':global_step,'rng_state':rng.bit_generator.state,'history':history,
                         'best':best,'best_epoch':best_epoch,'elapsed_seconds':row['elapsed_seconds'],'run':run})
        write_json(out/'status.json',{'complete':False,'epoch':epoch,'best_epoch':best_epoch,'best_validation':best})
        print('EPOCH',json.dumps(row),flush=True)
    if epoch<t['epochs']:
        write_json(out/'status.json',{'complete':False,'stopped_after_epochs':epoch,'best_epoch':best_epoch,'best_validation':best})
        return
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    test=np.load(args.data/'test.npy',mmap_mode='r')
    testmetric,pred,z=evaluate(model,test,c,device)
    valmetric,_,_=evaluate(model,rawval,c,device,keep=0)
    trainmetric,_,_=evaluate(model,raw[:256],c,device,keep=0)
    write_json(out/'metrics.json',{'best_epoch':best_epoch,'validation':valmetric,'test':testmetric,
                                  'train_subset':trainmetric,'elapsed_seconds':elapsed_before+time.perf_counter()-started})
    np.savez_compressed(out/'test_preview.npz',reference=np.asarray(test[:6]),reconstructed=pred,latent=z)
    plot_results(out,history,np.asarray(test[:6]),pred)
    write_json(out/'status.json',{'complete':True,'epoch':epoch,'best_epoch':best_epoch,'best_validation':best})
    print('COMPLETE',json.dumps({'best_epoch':best_epoch,'test':testmetric['relative_L2']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='mps')
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epochs',type=int,default=0)
    main(p.parse_args())
