"""Controlled GH200 decoder-only memory comparison, with parameter AND latent VJPs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
from contgreedy.fae.model import init_model, decode_coordinates

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.reference_root / "src"))
    from pod_decoder2 import PODClassifierDecoder
    from functional_autoencoders.positional_encodings import RandomFourierEncoding
    if jax.default_backend() != "gpu":
        raise RuntimeError("GH200 required")
    n,b=100,64
    xy0=jnp.zeros((1,4,2),jnp.float32)
    z0=jnp.zeros((1,128),jnp.float32)
    results=[]
    cases=[("darcy_pod_w128_l3",128,3),("conditional_w128_l3",128,3),
           ("conditional_w256_l6",256,6)]
    for name,width,blocks in cases:
        if name.startswith("darcy"):
            decoder=PODClassifierDecoder(out_dim=1,n_regions=2,n_basis=128,M=1,
                basis_hidden_dim=width,basis_n_layers=blocks,
                coefs_hidden_dim=128,coefs_n_layers=3,
                positional_encoding=RandomFourierEncoding(n_features=128,in_dim=2,
                                                          scale=3.0,learnable=False))
            params=jax.jit(decoder.init)(jax.random.PRNGKey(1),z0,xy0)["params"]
            def loss(params,z,xy,target):
                y,pi,logits=decoder.apply({"params":params},z,xy,train=True)
                return .5*jnp.mean((y[...,0]-target)**2)
        else:
            cfg=dict(latent_dimension=128,width=4,modes=2,pool_channels=16,
                     kernel_hidden=64,coordinate_frequencies=6,
                     decoder_kind="conditional_mlp",decoder_width=width,decoder_blocks=blocks)
            all_params=jax.jit(lambda k:init_model(k,cfg))(jax.random.PRNGKey(1))
            params={k:v for k,v in all_params.items()
                    if k.startswith("decoder.") or k.startswith("position_encoding.")}
            def loss(params,z,xy,target):
                y=decode_coordinates(params,z,xy[0],cfg)
                return .5*jnp.mean((y-target)**2)
        start=time.perf_counter()
        fn=jax.jit(jax.value_and_grad(loss,argnums=(0,1)))
        compiled=fn.lower(params,jax.ShapeDtypeStruct((b,128),jnp.float32),
                          jax.ShapeDtypeStruct((b,n*n,2),jnp.float32),
                          jax.ShapeDtypeStruct((b,n*n),jnp.float32)).compile()
        m=compiled.memory_analysis()
        row=dict(case=name,batch=b,resolution=n,
                 parameters=sum(v.size for v in jax.tree_util.tree_leaves(params)),
                 temp_GiB=m.temp_size_in_bytes/2**30,
                 total_GiB=(m.temp_size_in_bytes+m.argument_size_in_bytes+
                            m.output_size_in_bytes-m.alias_size_in_bytes)/2**30,
                 compile_seconds=time.perf_counter()-start)
        results.append(row)
        print("DECODER_MEMORY",json.dumps(row),flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(
            scope="Decoder only; matched MSE; gradients w.r.t. params and z; no encoder/optimizer.",
            reference_source_sha256=hashlib.sha256((a.reference_root/"src/pod_decoder2.py").read_bytes()).hexdigest(),
            rows=results),indent=2))
        del compiled,fn,params
        jax.clear_caches()
if __name__=="__main__":
    main()
