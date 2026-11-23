"""Export float32 forward/gradient/Adam references for backend checks."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict,unflatten_dict
from train_jax import build_model
from fae_common import DEFAULT_CONFIG,load_config,data_metadata,grid,sha256,write_json,learning_rate

def main(args):
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_enable_x64',False)
    c=load_config(args.config);t=c['training'];meta=data_metadata(args.data,c)
    m=build_model(c['model']['latent_dimension'],meta['mean'],meta['std'],c['model'])
    with np.load(args.initial) as a:params=unflatten_dict({k:jnp.array(a[k]) for k in a.files},sep='/')
    fields=np.load(args.data/'train.npy',mmap_mode='r');args.output.mkdir(parents=True,exist_ok=True)
    for n,b in [(17,2),(c['data']['grid_nodes'],1)]:
        stride=(fields.shape[1]-1)//(n-1)
        inputs=((np.array(fields[:b,::stride,::stride])-meta['mean'])/meta['std']).astype(np.float32)
        x=np.broadcast_to(grid(n),(b,n*n,2)).copy()
        query=np.broadcast_to(np.random.default_rng(31).uniform(-1,1,(32,2)).astype(np.float32),(b,32,2)).copy()
        target=np.random.default_rng(32).normal(size=(b,32,1)).astype(np.float32)
        def loss(params,u):
            z=m.encoder.apply({'params':params['encoder']},u.reshape(b,n*n,1),jnp.array(x))
            pred=m.decoder.apply({'params':params['decoder']},z,jnp.array(query))
            value=.5*jnp.mean((pred-jnp.array(target))**2)+t['latent_penalty']*jnp.mean(jnp.sum(z*z,axis=1))
            return value,(pred,z)
        (value,(pred,z)),(grads,du)=jax.jit(jax.value_and_grad(loss,argnums=(0,1),has_aux=True))(params,jnp.array(inputs))
        opt=optax.chain(optax.clip_by_global_norm(t['gradient_clip']),
                        optax.adam(t['learning_rate'],b1=t['beta1'],b2=t['beta2'],eps=t['epsilon']))
        update,_=opt.update(grads,opt.init(params),params);after=optax.apply_updates(params,update)
        np.savez(args.output/f'jax_reference_{n}.npz',inputs=inputs,query=query,targets=target,
                 prediction=np.asarray(pred),latent=np.asarray(z),loss=np.asarray(value),input_gradient=np.asarray(du),
                 **{'grad/'+k:np.asarray(v) for k,v in flatten_dict(grads,sep='/').items()},
                 **{'updated/'+k:np.asarray(v) for k,v in flatten_dict(after,sep='/').items()})
        print(n,float(value),flush=True)
    steps=t['epochs']*(c['data']['train_size']//t['batch_size'])
    schedule=optax.cosine_decay_schedule(t['learning_rate'],max(steps-1,1),
                                        alpha=t['minimum_learning_rate']/t['learning_rate'])
    indices=np.arange(steps)
    shared_rates=np.array([learning_rate(int(i),steps,c) for i in indices])
    jax_rates=np.asarray(jax.vmap(schedule)(jnp.asarray(indices)))
    schedule_error=float(np.max(np.abs(shared_rates-jax_rates)))
    assert np.allclose(shared_rates,jax_rates,rtol=2e-5,atol=1e-10)
    write_json(args.output/'reference_metadata.json',{'shared_config':c,'initial_sha256':sha256(args.initial),
               'normalization':{'mean':meta['mean'],'std':meta['std']},'jax':jax.__version__,
               'maximum_learning_rate_absolute_error':schedule_error})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
