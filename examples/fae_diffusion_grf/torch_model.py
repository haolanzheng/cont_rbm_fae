"""Float32 PyTorch equivalent of the native Flax FAE used in this pilot.

Flax parameter paths are retained for explicit weight/gradient comparisons.
The original [32,64,128] convolutional U-shaped encoder, orthogonal integral
pooling, and width-256/six-block coordinate decoder are unchanged.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fae_common import grid,quadrature,load_config,expected_parameter_shapes


def position(x,frequencies=6):
    xx,yy=x[...,0],x[...,1]
    base=torch.stack((xx,yy,1-xx.square(),1-yy.square()),-1)
    f=torch.pi*2.**torch.arange(frequencies,device=x.device,dtype=x.dtype)
    return torch.cat((base,torch.sin(xx[...,None]*f),torch.cos(xx[...,None]*f),
                      torch.sin(yy[...,None]*f),torch.cos(yy[...,None]*f)),-1)


def resize_matrix(old,new):
    """Half-pixel, antialiased linear interpolation, as in jax.image.resize."""
    scale=old/new
    center=(np.arange(new,dtype=np.float32)+.5)*scale-.5
    w=np.maximum(0,1-np.abs(center[:,None]-np.arange(old,dtype=np.float32)[None,:])/max(scale,1))
    return w/np.maximum(w.sum(axis=1,keepdims=True),1e-10)


class TorchFAE(nn.Module):
    def __init__(self,initial,mean,std,config=None):
        super().__init__()
        self.mean=float(mean);self.std=float(std)
        self.config=load_config() if config is None else config
        self.model_config=self.config['model']
        if self.model_config['encoder_blocks_per_level']!=1:
            raise ValueError('The current paired implementation uses one residual block per encoder level.')
        if isinstance(initial,(str,Path)):
            with np.load(initial,allow_pickle=False) as archive: initial={k:archive[k] for k in archive.files}
        expected=expected_parameter_shapes(self.config)
        if {k:tuple(v.shape) for k,v in initial.items()}!=expected:
            raise ValueError('Initial parameter names/shapes do not match the shared model configuration.')
        self.flax_keys=tuple(initial)
        params={}
        for key,value in initial.items():
            value=np.asarray(value,dtype=np.float32)
            if key.endswith('/kernel'):
                if value.ndim==4:value=value.transpose(3,2,0,1)
                elif value.ndim==2:value=value.T
            params[key.replace('/','__')]=nn.Parameter(torch.tensor(np.array(value,copy=True)))
        self.p=nn.ParameterDict(params)
        self.latent=self.get('encoder/Dense_0/kernel').shape[0]
        if self.latent!=self.model_config['latent_dimension']:raise ValueError('Initial weights and latent configuration differ.')
        self._resize_cache={};self._grid_cache={}

    def get(self,key): return self.p[key.replace('/','__')]

    def dense(self,x,prefix):
        bias=self.p.get((prefix+'/bias').replace('/','__'),None)
        return F.linear(x,self.get(prefix+'/kernel'),bias)

    def conv(self,x,prefix):
        return F.conv2d(x,self.get(prefix+'/kernel'),self.get(prefix+'/bias'),padding=1)

    def norm(self,x,prefix):
        groups=next(g for g in range(min(8,x.shape[1]),0,-1) if x.shape[1]%g==0)
        return F.group_norm(x,groups,self.get(prefix+'/scale'),self.get(prefix+'/bias'),eps=1e-6)

    def stage(self,x,prefix):
        x=self.conv(x,prefix+'/Conv_0');block=prefix+'/UNOResidualBlock_0'
        h=self.conv(F.gelu(self.norm(x,block+'/GroupNorm_0'),approximate='tanh'),block+'/Conv_0')
        h=self.conv(F.gelu(self.norm(h,block+'/GroupNorm_1'),approximate='tanh'),block+'/Conv_1')
        return x+h

    def resize(self,x,n):
        old=x.shape[-1];key=(old,n,str(x.device))
        if key not in self._resize_cache:
            self._resize_cache[key]=torch.tensor(resize_matrix(old,n),dtype=x.dtype,device=x.device)
        w=self._resize_cache[key]
        return torch.matmul(w,torch.matmul(x,w.T))

    def geometry(self,n,device):
        key=(n,str(device))
        if key not in self._grid_cache:
            xy=torch.tensor(grid(n),dtype=torch.float32,device=device)
            w=torch.tensor(quadrature(n),dtype=torch.float32,device=device)
            self._grid_cache[key]=(xy,w,position(xy,self.model_config['coordinate_frequencies']))
        return self._grid_cache[key]

    def encode(self,field):
        # field is [B,n,n], standardized using training-only constants.
        b,n,_=field.shape;xy,w,pos=self.geometry(n,field.device)
        h=torch.cat((field[...,None],xy.reshape(1,n,n,2).expand(b,-1,-1,-1)),-1).permute(0,3,1,2)
        h=h.contiguous(memory_format=torch.channels_last)
        prefix='encoder/UNO_0';skips=[]
        levels=len(self.model_config['encoder_channels']);channels=self.model_config['pool_channels']
        for i in range(levels):
            h=self.stage(h,f'{prefix}/down_{i}');skips.append(h)
            if i<levels-1:h=self.resize(h,h.shape[-1]//2)
        for i in reversed(range(levels-1)):
            h=self.resize(h,skips[i].shape[-1]);h=torch.cat((h,skips[i]),dim=1)
            h=self.stage(h,f'{prefix}/up_{i}')
        h=self.conv(F.gelu(self.norm(h,prefix+'/GroupNorm_0'),approximate='tanh'),prefix+'/Conv_0')
        feature=h.permute(0,2,3,1).reshape(b,n*n,channels)
        feature=feature+self.get('encoder/token_skip_alpha')*self.dense(field.reshape(b,n*n,1),'encoder/token_skip_proj')
        kernel=F.gelu(self.dense(pos,'encoder/pooling_fn/Dense_0'),approximate='tanh')
        kernel=F.gelu(self.dense(kernel,'encoder/pooling_fn/Dense_1'),approximate='tanh')
        kernel=self.dense(kernel,'encoder/pooling_fn/Dense_2').reshape(n*n,self.latent,channels)
        weighted=kernel*w[:,None,None].sqrt()
        matrix=weighted.permute(1,0,2).reshape(self.latent,-1)
        gram=matrix@matrix.T+self.model_config['orthogonal_epsilon']*torch.eye(self.latent,device=field.device,dtype=torch.float32)
        # Integrate before applying L^{-1}: exact linear-algebra equivalence,
        # avoiding a triangular solve with n*n*16 right-hand sides.
        pooled=(feature*w[None,:,None]).reshape(b,-1)@kernel.permute(1,0,2).reshape(self.latent,-1).T
        if field.device.type=='mps':
            # MPS lacks Cholesky in the installed Torch version. Only the small
            # latent matrix and B*latent RHS cross devices; autograd is retained.
            factor=torch.linalg.cholesky(gram.to('cpu'))
            z=torch.linalg.solve_triangular(factor,pooled.T.to('cpu'),upper=False).T.to(field.device)
        else:
            factor=torch.linalg.cholesky(gram)
            z=torch.linalg.solve_triangular(factor,pooled.T,upper=False).T
        return self.dense(z,'encoder/Dense_0')

    def decode(self,z,xy):
        if xy.ndim==2:xy=xy[None].expand(len(z),-1,-1)
        prefix='decoder/ConcatResidualDecoder_0'
        h=torch.cat((position(xy,self.model_config['coordinate_frequencies']),z[:,None,:].expand(-1,xy.shape[1],-1)),-1)
        h=self.dense(h,prefix+'/ResidualMLP_0/Dense_0')
        for i in range(self.model_config['decoder_blocks']):
            path=f'{prefix}/ResidualMLP_0/ResidualMLPBlock_{i}'
            h=h+self.dense(F.gelu(self.dense(h,path+'/Dense_0'),approximate='tanh'),path+'/Dense_1')
        lower=self.model_config['coefficient_lower'];upper=self.model_config['coefficient_upper']
        return lower+(upper-lower)*torch.sigmoid(self.dense(h,prefix+'/Dense_0')).squeeze(-1)

    def forward(self,field,xy):
        z=self.encode(field)
        return (self.decode(z,xy)-self.mean)/self.std,z

    def flax_arrays(self,grad=False):
        result={}
        for k in self.flax_keys:
            v=self.get(k).grad if grad else self.get(k)
            a=v.detach().cpu().numpy()
            if k.endswith('/kernel'):
                if a.ndim==4:a=a.transpose(2,3,1,0)
                elif a.ndim==2:a=a.T
            result[k]=a
        return result
