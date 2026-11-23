"""Generate one native Flax parameter archive for both training backends."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from train_jax import build_model
from fae_common import DEFAULT_CONFIG,load_config,data_metadata,grid,sha256,write_json

def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.output.suffix!='.npz':raise ValueError('Initial parameter archive must have suffix .npz.')
    jax.config.update('jax_enable_x64',False)
    c=load_config(args.config);meta=data_metadata(args.data,c)
    n=c['data']['grid_nodes'];xy=jnp.asarray(grid(n))[None]
    model=build_model(c['model']['latent_dimension'],meta['mean'],meta['std'],c['model'])
    # Initial parameter shapes depend on channel/latent dimensions, not fields.
    values=model.init(jax.random.PRNGKey(c['training']['seed']),
                      jnp.zeros((1,n*n,1),dtype=jnp.float32),xy,xy[:,:32])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:np.asarray(v,dtype=np.float32) for k,v in flatten_dict(values['params'],sep='/').items()}
    np.savez(args.output,**arrays)
    write_json(args.output.with_suffix('.json'),{'shared_config':c,'jax':jax.__version__,
        'parameter_count':sum(v.size for v in arrays.values()),'sha256':sha256(args.output)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
