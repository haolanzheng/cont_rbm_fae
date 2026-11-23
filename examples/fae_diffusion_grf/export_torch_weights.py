"""Export a trained Torch checkpoint as native Flax-layout model weights.

Optimizer states are not converted. Exact continuation uses each backend's
own --resume checkpoint; this archive is for inference or a new training run.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch_model import TorchFAE
from fae_common import sha256,write_json

def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.output.suffix!='.npz':raise ValueError('Output must have suffix .npz.')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    run=state['run']
    if sha256(args.initial)!=run['initial_sha256']:
        raise ValueError('Use the initial archive recorded by this training run.')
    norm=run['normalization']
    model=TorchFAE(args.initial,norm['mean'],norm['std'],run['shared_config'])
    model.load_state_dict(state['model'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,**model.flax_arrays())
    write_json(args.output.with_suffix('.json'),{'shared_config':run['shared_config'],
        'normalization':norm,'source_checkpoint':str(args.checkpoint),
        'source_checkpoint_sha256':sha256(args.checkpoint),'epoch':state['epoch'],
        'sha256':sha256(args.output),'optimizer_converted':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
