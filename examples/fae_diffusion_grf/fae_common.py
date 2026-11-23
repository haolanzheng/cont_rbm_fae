"""Configuration and data conventions shared by JAX/GPU and Torch/MPS."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

DEFAULT_CONFIG=Path(__file__).parent/'configs/fae_128.json'

def load_config(path=DEFAULT_CONFIG):
    c=json.loads(Path(path).read_text())
    if c['numerics']['dtype']!='float32':raise ValueError('This experiment uses float32.')
    if c['training']['weight_decay']!=0:raise ValueError('This paired implementation uses Adam without weight decay.')
    return c

def learning_rate(step,total_steps,c):
    t=c['training'];f=min(step/max(total_steps-1,1),1.)
    return t['minimum_learning_rate']+.5*(t['learning_rate']-t['minimum_learning_rate'])*(1+math.cos(math.pi*f))

def grid(n):
    x=np.linspace(-1,1,n,dtype=np.float32);xx,yy=np.meshgrid(x,x)
    return np.stack((xx,yy),-1).reshape(-1,2)

def quadrature(n):
    w=np.ones(n,dtype=np.float32);w[[0,-1]]=.5
    return np.outer(w,w).ravel()/(n-1)**2

def stats(v):
    v=np.asarray(v)
    return dict(mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=float(v.max()))

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def data_metadata(directory,c):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text())
    for name in ('train','validation','test'):
        a=np.load(directory/f'{name}.npy',mmap_mode='r')
        if a.shape!=(c['data'][f'{name}_size'],c['data']['grid_nodes'],c['data']['grid_nodes']):
            raise ValueError(f'Unexpected {name} shape: {a.shape}')
    return meta

def expected_parameter_shapes(c):
    """Flax-layout shapes for the shared architecture; catches silent drift."""
    m=c['model'];shapes={};channels=m['encoder_channels'];pool=m['pool_channels'];latent=m['latent_dimension']
    hidden=m['kernel_hidden'];features=4+4*m['coordinate_frequencies'];width=m['decoder_width']
    def dense(prefix,ins,outs,bias=True):
        shapes[prefix+'/kernel']=(ins,outs)
        if bias:shapes[prefix+'/bias']=(outs,)
    def conv(prefix,ins,outs):
        shapes[prefix+'/kernel']=(3,3,ins,outs);shapes[prefix+'/bias']=(outs,)
    def norm(prefix,ch):
        shapes[prefix+'/scale']=(ch,);shapes[prefix+'/bias']=(ch,)
    def stage(prefix,ins,outs):
        conv(prefix+'/Conv_0',ins,outs)
        for j in range(m['encoder_blocks_per_level']):
            block=prefix+f'/UNOResidualBlock_{j}'
            for i in range(2):norm(block+f'/GroupNorm_{i}',outs);conv(block+f'/Conv_{i}',outs,outs)
    for i,ch in enumerate(channels):stage(f'encoder/UNO_0/down_{i}',3 if i==0 else channels[i-1],ch)
    for i in reversed(range(len(channels)-1)):stage(f'encoder/UNO_0/up_{i}',channels[i+1]+channels[i],channels[i])
    norm('encoder/UNO_0/GroupNorm_0',channels[0]);conv('encoder/UNO_0/Conv_0',channels[0],pool)
    dense('encoder/token_skip_proj',1,pool,False);shapes['encoder/token_skip_alpha']=()
    dense('encoder/pooling_fn/Dense_0',features,hidden);dense('encoder/pooling_fn/Dense_1',hidden,hidden)
    dense('encoder/pooling_fn/Dense_2',hidden,latent*pool);dense('encoder/Dense_0',latent,latent,False)
    prefix='decoder/ConcatResidualDecoder_0';dense(prefix+'/ResidualMLP_0/Dense_0',latent+features,width)
    for i in range(m['decoder_blocks']):
        for j in range(2):dense(prefix+f'/ResidualMLP_0/ResidualMLPBlock_{i}/Dense_{j}',width,width)
    dense(prefix+'/Dense_0',width,1)
    return shapes
