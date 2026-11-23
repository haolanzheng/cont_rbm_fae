"""Train the existing native UNO/coordinate-MLP FAE on bounded Matérn inputs.

Shared encoder/decoder classes are imported unchanged from contgreedy.fae.
Only the scalar output transform enforces the conductivity interval. Query
subsampling is an unbiased estimate of the full-grid trapezoidal L2 loss.
"""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, serialization
from flax.traverse_util import unflatten_dict
from fae_common import DEFAULT_CONFIG,load_config,data_metadata,sha256,grid as common_grid,quadrature as common_weights
from contgreedy.fae.core import Autoencoder, Decoder
from contgreedy.fae.decoder import CoordinateEncoding, ConcatResidualDecoder
from contgreedy.fae.uno import UNOEncoder


def write_json(path, value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def coords(n):
    return common_grid(n)


def weights(n):
    return common_weights(n)


class OrthogonalIntegral(nn.Module):
    """Same pooling as examples.hyperelasticity.fae.OrthogonalIntegral."""
    latent_dim: int
    hidden: int=64
    epsilon: float=1e-3

    @nn.compact
    def __call__(self,u,x):
        h=nn.gelu(nn.Dense(self.hidden)(x[0]));h=nn.gelu(nn.Dense(self.hidden)(h))
        kernel=nn.Dense(self.latent_dim*u.shape[-1])(h).reshape(u.shape[1],self.latent_dim,u.shape[-1])
        n=int(round(u.shape[1]**.5));w=jnp.asarray(weights(n))
        gram=jnp.einsum('nlc,nmc,n->lm',kernel,kernel,w)
        factor=jnp.linalg.cholesky(gram+self.epsilon*jnp.eye(self.latent_dim))
        whitened=jax.scipy.linalg.solve_triangular(factor,kernel.transpose(1,0,2).reshape(self.latent_dim,-1),lower=True)
        whitened=whitened.reshape(self.latent_dim,u.shape[1],u.shape[-1]).transpose(1,0,2)
        return jnp.einsum('bnc,nlc,n->bl',u,whitened,w)


class BoundedDecoder(Decoder):
    mean: float
    std: float
    width: int=256
    blocks: int=6
    frequencies: int=6
    lower: float=.1
    upper: float=10.

    def _forward(self,z,x,train=False):
        raw=ConcatResidualDecoder(width=self.width,blocks=self.blocks,frequencies=self.frequencies)(z,x,train)
        a=self.lower+(self.upper-self.lower)*jax.nn.sigmoid(raw)
        return (a-self.mean)/self.std


def build_model(latent,mean,std,model_config=None):
    m=load_config()['model'] if model_config is None else model_config
    encoder=UNOEncoder(is_variational=False,latent_dim=latent,uno_channels=tuple(m['encoder_channels']),
                       uno_blocks_per_level=m['encoder_blocks_per_level'],uno_projection_dim=m['pool_channels'],
                       positional_encoding=CoordinateEncoding(m['coordinate_frequencies']),
                       pooling_fn=OrthogonalIntegral(latent,m['kernel_hidden'],m['orthogonal_epsilon']))
    decoder=BoundedDecoder(mean=mean,std=std,width=m['decoder_width'],blocks=m['decoder_blocks'],
                           frequencies=m['coordinate_frequencies'],lower=m['coefficient_lower'],upper=m['coefficient_upper'])
    return Autoencoder(encoder=encoder,decoder=decoder)


def stats(a):
    a=np.asarray(a)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}


def main(args):
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_enable_x64',False)
    shared=load_config(args.config);t=shared['training'];m=shared['model']
    args.latent=m['latent_dimension'];args.batch=t['batch_size'];args.epochs=t['epochs']
    args.queries=t['decoder_queries'];args.seed=t['seed'];args.eval_batch=t['evaluation_batch_size']
    args.eval_every=t['evaluation_interval_epochs']
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.msgpack').exists() and not args.resume:
        raise RuntimeError('Existing run: use --resume or a new output directory.')
    meta=data_metadata(args.data,shared)
    mean,std=meta['mean'],meta['std']
    raw_train=np.load(args.data/'train.npy',mmap_mode='r')
    raw_val=np.load(args.data/'validation.npy',mmap_mode='r')
    if args.preflight: raw_val=raw_val[:8]
    n=raw_train.shape[1];q=n*n
    normalized=(np.asarray(raw_train)-mean)/std
    xy=coords(n);quadrature=weights(n)
    model=build_model(args.latent,mean,std,m)
    key=jax.random.PRNGKey(args.seed)
    xenc=jnp.broadcast_to(xy,(args.batch,q,2))
    initial=model.init(key,jnp.asarray(normalized[:args.batch].reshape(args.batch,q,1)),xenc,
                       jnp.broadcast_to(xy[:args.queries],(args.batch,args.queries,2)))
    with np.load(args.initial,allow_pickle=False) as archive:
        params=unflatten_dict({k:jnp.asarray(archive[k],dtype=jnp.float32) for k in archive.files},sep='/')
    if jax.tree_util.tree_structure(params)!=jax.tree_util.tree_structure(initial['params']):
        raise ValueError('Shared initial weights do not match the JAX architecture.')
    if any(a.shape!=b.shape for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(initial['params']))):
        raise ValueError('Shared initial parameter shapes differ.')
    param_count=sum(v.size for v in jax.tree_util.tree_leaves(params))
    steps_per_epoch=len(raw_train)//args.batch
    if len(raw_train)%args.batch:
        raise ValueError('The shared schedule requires complete training batches.')
    total_steps=steps_per_epoch*args.epochs
    schedule=optax.cosine_decay_schedule(t['learning_rate'],max(total_steps-1,1),
                                       alpha=t['minimum_learning_rate']/t['learning_rate'])
    optimizer=optax.chain(optax.clip_by_global_norm(t['gradient_clip']),
                         optax.adam(schedule,b1=t['beta1'],b2=t['beta2'],eps=t['epsilon']))
    optstate=optimizer.init(params)

    def loss(params,inputs,targets,xquery):
        z=model.encoder.apply({'params':params['encoder']},inputs,xenc)
        reconstructed=model.decoder.apply({'params':params['decoder']},z,xquery)
        rec=.5*jnp.mean((reconstructed-targets)**2)
        return rec+t['latent_penalty']*jnp.mean(jnp.sum(z*z,axis=-1)),rec

    @jax.jit
    def step(params,optstate,inputs,targets,xquery):
        (value,rec),grads=jax.value_and_grad(loss,has_aux=True)(params,inputs,targets,xquery)
        updates,newstate=optimizer.update(grads,optstate,params)
        return optax.apply_updates(params,updates),newstate,value,rec,optax.global_norm(grads)

    @jax.jit
    def encode(params,inputs):
        return model.encoder.apply({'params':params['encoder']},inputs,
                                   jnp.broadcast_to(xy,(inputs.shape[0],q,2)))

    @jax.jit
    def decode(params,z,points):
        return model.decoder.apply({'params':params['decoder']},z,points)*std+mean

    def evaluate(params,raw,keep=0):
        errors=[];predictions=[];codes=[]
        for first in range(0,len(raw),args.eval_batch):
            batch=np.array(raw[first:first+args.eval_batch],np.float32).reshape(-1,q)
            b=len(batch);z=encode(params,jnp.asarray(((batch-mean)/std)[...,None]))
            prediction=np.empty_like(batch)
            chunk=t['decoder_evaluation_chunk']
            for start in range(0,q,chunk):
                points=xy[start:start+chunk];padded=np.pad(points,((0,chunk-len(points)),(0,0)),mode='edge')
                values=decode(params,z,jnp.broadcast_to(padded,(b,chunk,2)))
                prediction[:,start:start+len(points)]=np.asarray(values)[:,:len(points),0]
            errors.extend(np.sqrt(((prediction-batch)**2@quadrature)/(batch*batch@quadrature)))
            codes.extend(np.asarray(z))
            if first<keep:predictions.extend(prediction[:max(0,keep-first)])
        return {'relative_L2':stats(errors),'per_sample':np.asarray(errors).tolist()},np.asarray(predictions),np.asarray(codes)

    config={'shared_config':shared,'model':m,'initial_sha256':sha256(args.initial),
            'data_sha256':{k:sha256(args.data/f'{k}.npy') for k in ('train','validation','test')},
            'bounded_head':'a=lower+(upper-lower)*sigmoid(raw_decoder); output standardized for loss',
            'batch':args.batch,'queries':args.queries,'epochs':args.epochs,'seed':args.seed,
            'normalization':{'mean':mean,'std':std},'grid_nodes':n,'parameters':param_count,
            'devices':[str(d) for d in jax.devices()],'jax':jax.__version__,
            'query_sampling':'with replacement, probability proportional to trapezoidal weights',
            'validation':f'full {n}x{n} grid; best mean relative L2; test only after selection',
            'preflight':args.preflight,
            'data_metadata':meta,'selection_policy':'validation selects checkpoint; test follows completed training'}
    if args.resume and (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=config:
            raise RuntimeError('Resume requires the same model, data, schedule, and device configuration.')
    write_json(out/'config.json',config)
    print('CONFIG',json.dumps({k:v for k,v in config.items() if k!='data_metadata'}),flush=True)
    rng=np.random.default_rng(args.seed);history=[];best=float('inf');best_epoch=0
    started=time.perf_counter();start_epoch=0;epoch=0
    if args.resume:
        restored=serialization.from_bytes({'params':params,'optstate':optstate,'metadata':''},
                                           (out/'last.msgpack').read_bytes())
        params,optstate=restored['params'],restored['optstate']
        saved=json.loads(restored['metadata']);start_epoch=saved['epoch'];epoch=start_epoch
        rng.bit_generator.state=saved['rng_state'];history=saved['history']
        best,best_epoch=saved['best'],saved['best_epoch']
        started-=history[-1]['elapsed_seconds']
    if args.benchmark_steps:
        durations=[]
        for i in range(args.benchmark_steps+3):
            before=time.perf_counter()
            ids=rng.choice(len(raw_train),args.batch,replace=False)
            idx=rng.choice(q,args.queries,replace=True,p=quadrature/quadrature.sum())
            u=normalized[ids].reshape(args.batch,q,1)
            params,optstate,value,rec,gnorm=step(params,optstate,jnp.asarray(u),jnp.asarray(u[:,idx]),
                                                jnp.asarray(np.broadcast_to(xy[idx],(args.batch,args.queries,2))))
            value,rec,gnorm=map(float,(value,rec,gnorm))
            assert np.isfinite([value,rec,gnorm]).all()
            elapsed=time.perf_counter()-before
            if i>=3:durations.append(elapsed)
        result={'steady_step_seconds_mean':float(np.mean(durations)),
                'steady_step_seconds_median':float(np.median(durations)),
                'training_seconds_per_epoch':float(np.mean(durations)*steps_per_epoch),
                'training_seconds_100_epochs':float(np.mean(durations)*steps_per_epoch*100),
                'includes':'input transfer, gradient, Adam, synchronization; excludes validation/checkpoint IO',
                'measured_steps':len(durations),'batch':args.batch,'latent':args.latent,
                'devices':[str(d) for d in jax.devices()]}
        write_json(out/'benchmark.json',result);print('BENCHMARK',json.dumps(result),flush=True)
        return
    final_epoch=min(args.epochs,args.stop_after_epochs or args.epochs)
    for epoch in range(start_epoch+1,final_epoch+1):
        order=rng.permutation(len(raw_train));losses=[];recs=[]
        for start in range(0,len(order)-args.batch+1,args.batch):
            if args.preflight and start>=3*args.batch: break
            ids=order[start:start+args.batch];idx=rng.choice(q,args.queries,replace=True,p=quadrature/quadrature.sum())
            u=normalized[ids].reshape(args.batch,q,1)
            targets=u[:,idx];xquery=np.broadcast_to(xy[idx],(args.batch,args.queries,2))
            params,optstate,value,rec,gnorm=step(params,optstate,jnp.asarray(u),jnp.asarray(targets),jnp.asarray(xquery))
            value,rec,gnorm=map(float,(value,rec,gnorm))
            if not np.isfinite([value,rec,gnorm]).all():raise RuntimeError(f'Nonfinite training at epoch {epoch}')
            losses.append(value);recs.append(rec)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'reconstruction':float(np.mean(recs)),
             'elapsed_seconds':time.perf_counter()-started}
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            metric,_,_=evaluate(params,raw_val)
            row['validation']=metric['relative_L2']
            if row['validation']['mean']<best:
                best=row['validation']['mean'];best_epoch=epoch
                (out/'best.msgpack').write_bytes(serialization.to_bytes(params))
        history.append(row);write_json(out/'history.json',history)
        metadata=json.dumps({'epoch':epoch,'history':history,'rng_state':rng.bit_generator.state,
                             'best':best,'best_epoch':best_epoch})
        tmp=out/'last.msgpack.tmp'
        tmp.write_bytes(serialization.to_bytes({'params':params,'optstate':optstate,'metadata':metadata}))
        tmp.replace(out/'last.msgpack')
        write_json(out/'status.json',{'complete':False,'epoch':epoch,'best_epoch':best_epoch,'best_validation':best})
        print('EPOCH',json.dumps(row),flush=True)
        if args.preflight:
            break
    if epoch<args.epochs and not args.preflight:
        write_json(out/'status.json',{'complete':False,'stopped_after_epochs':epoch,
                                     'best_epoch':best_epoch,'best_validation':best})
        return
    params=serialization.from_bytes(params,(out/'best.msgpack').read_bytes())
    valmetric,pred,z=evaluate(params,raw_val,keep=6)
    np.savez_compressed(out/'validation_preview.npz',reference=np.asarray(raw_val[:6]),
                        reconstructed=pred.reshape(-1,n,n),latent=z)
    trainmetric,_,_=evaluate(params,raw_train[:8 if args.preflight else 256])
    metrics={'best_epoch':best_epoch,'validation':valmetric,'train_subset':trainmetric,
             'test':'not evaluated: architecture selection uses validation',
             'elapsed_seconds':time.perf_counter()-started}
    if args.final_test and not args.preflight:
        raw_test=np.load(args.data/'test.npy',mmap_mode='r')
        metrics['test'],pred,z=evaluate(params,raw_test,keep=6)
        np.savez_compressed(out/'test_preview.npz',reference=np.asarray(raw_test[:6]),
                            reconstructed=pred.reshape(-1,n,n),latent=z)
    write_json(out/'metrics.json',metrics)
    write_json(out/'status.json',{'complete':not args.preflight,'preflight':args.preflight,
                                 'epoch':epoch,'best_epoch':best_epoch,'best_validation':best})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--benchmark-steps',type=int,default=0)
    p.add_argument('--stop-after-epochs',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--final-test',action='store_true')
    main(p.parse_args())
