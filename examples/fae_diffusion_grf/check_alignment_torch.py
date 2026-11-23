"""Check shared weights, forward values, derivatives, and an Adam update."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch_model import TorchFAE
from fae_common import DEFAULT_CONFIG,load_config,sha256,write_json

def main(args):
    torch.set_num_threads(4);c=load_config(args.config);t=c['training']
    meta=json.loads((args.reference/'reference_metadata.json').read_text())
    assert meta['shared_config']==c and meta['initial_sha256']==sha256(args.initial)
    norm=meta['normalization'];results=[]
    def error(v,ref):
        v=np.asarray(v);ref=np.asarray(ref)
        return {'relative':float(np.linalg.norm(v-ref)/max(np.linalg.norm(ref),1e-10)),
                'maximum_absolute':float(np.max(np.abs(v-ref)))}
    for n in [17,c['data']['grid_nodes']]:
        model=TorchFAE(args.initial,norm['mean'],norm['std'],c).to(args.device)
        initial={k:v.copy() for k,v in model.flax_arrays().items()}
        opt=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],betas=(t['beta1'],t['beta2']),eps=t['epsilon'],foreach=False)
        with np.load(args.reference/f'jax_reference_{n}.npz') as a:
            x=torch.tensor(a['inputs'],device=args.device,requires_grad=True)
            query=torch.tensor(a['query'],device=args.device);target=torch.tensor(a['targets'][...,0],device=args.device)
            pred,z=model(x,query)
            loss=.5*(pred-target).square().mean()+t['latent_penalty']*z.square().sum(1).mean();loss.backward()
            grads=model.flax_arrays(grad=True)
            row={'grid':n,'device':args.device,'torch':torch.__version__,'parameters':sum(v.numel() for v in model.parameters()),
                 'loss':error(float(loss.detach().cpu()),a['loss']),
                 'prediction':error(pred.detach().cpu().numpy()[...,None],a['prediction']),
                 'latent':error(z.detach().cpu().numpy(),a['latent']),
                 'input_gradient':error(x.grad.cpu().numpy(),a['input_gradient']),
                 'parameter_gradient':error(np.concatenate([grads[k].ravel() for k in model.flax_keys]),
                                            np.concatenate([a['grad/'+k].ravel() for k in model.flax_keys]))}
            torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],foreach=False);opt.step()
            updated=model.flax_arrays()
            row['parameters_after_adam']=error(np.concatenate([updated[k].ravel() for k in model.flax_keys]),
                                     np.concatenate([a['updated/'+k].ravel() for k in model.flax_keys]))
            row['end_to_end_adam_increment']=error(
                np.concatenate([(updated[k]-initial[k]).ravel() for k in model.flax_keys]),
                np.concatenate([(a['updated/'+k]-initial[k]).ravel() for k in model.flax_keys]))
            # Isolate optimizer agreement from rounding in near-zero gradients.
            same=TorchFAE(args.initial,norm['mean'],norm['std'],c).to(args.device)
            same_opt=torch.optim.Adam(same.parameters(),lr=t['learning_rate'],
                                     betas=(t['beta1'],t['beta2']),eps=t['epsilon'],foreach=False)
            for k in same.flax_keys:
                grad=a['grad/'+k]
                if k.endswith('/kernel'):
                    if grad.ndim==4:grad=grad.transpose(3,2,0,1)
                    elif grad.ndim==2:grad=grad.T
                same.get(k).grad=torch.tensor(np.array(grad,copy=True),device=args.device)
            torch.nn.utils.clip_grad_norm_(same.parameters(),t['gradient_clip'],foreach=False)
            same_opt.step();same_after=same.flax_arrays()
            row['adam_increment_identical_gradient']=error(
                np.concatenate([(same_after[k]-initial[k]).ravel() for k in model.flax_keys]),
                np.concatenate([(a['updated/'+k]-initial[k]).ravel() for k in model.flax_keys]))
            results.append(row);print(json.dumps(row),flush=True)
    write_json(args.reference/f'torch_parity_{args.device}.json',results)
    assert all(r['prediction']['relative']<5e-4 and r['parameter_gradient']['relative']<1e-2
               and r['input_gradient']['relative']<1e-2
               and r['adam_increment_identical_gradient']['relative']<1e-4 for r in results)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu')
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);main(p.parse_args())
