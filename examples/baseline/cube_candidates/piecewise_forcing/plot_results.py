import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser(description='Summarize the recorded piecewise forcing experiments.')
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[4]/'runs/alignment_baseline_20261111/piecewise_forcing')
root=parser.parse_args().results
data={d:json.loads((root/f'piecewise_extended_d{d}.json').read_text()) for d in [64,128,256]}
methods=['gd80','gd320','pool256','pool10000']
summary={'curves':[],'same_space':[],'shape':[],'checks':{}}
fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
colors=['#175788','#36a1a1','#db8653','#983c45']
labels=['GD, 80 steps','GD, 320 steps','Pool, 256','Pool, 10000']
for ax,(d,a) in zip(axes,data.items()):
    ns=sorted(set(r['N'] for r in a['full_greedy']))
    for method,col,label in zip(methods,colors,labels):
        med=[]
        for n in ns:
            vals=[r for r in a['full_greedy'] if r['method']==method and r['N']==n]
            med.append(np.median([r['normalized_by_global_maxnorm'] for r in vals]))
            summary['curves'].append({'d':d,'N':n,'method':method,'median_error_normalized':float(med[-1]),
              'median_global_upper_normalized':float(np.median([r['global_error_spectral_upper_normalized'] for r in vals])),
              'median_exact_mean_square_error':float(np.median([r['exact_uniform_mean_square_error'] for r in vals]))})
        ax.plot(ns,100*np.array(med),'o-',color=col,label=label,lw=1.6,ms=4)
    ax.set(title=f'Parameter dimension d = {d}',xlabel='Reduced dimension N',yscale='log')
    ax.grid(alpha=.2);ax.axhline(1,color='gray',ls=':',lw=.8)
    for n in ns:
        ratio=[]
        for s in range(1000,1004):
            gd=next(r for r in a['full_greedy'] if r['method']=='gd320' and r['N']==n and r['seed']==s)
            pool=next(r for r in a['full_greedy'] if r['method']=='pool10000' and r['N']==n and r['seed']==s)
            ratio.append(gd['global_error_spectral_upper']/pool['max_energy_error'])
        summary['checks'][f'd{d}_N{n}_gd320_upper_over_pool10000_observed']=ratio
    for n in sorted(set(r['N_before'] for r in a['same_space'])):
        rows=[r for r in a['same_space'] if r['N_before']==n]
        ratios={}
        for key in ['J','exact_mean_square_gain','fixed_top1pct_mean_square_gain']:
            ratios[key]=[r['candidates']['gd320'][key]/r['candidates']['pool10000'][key] for r in rows]
        summary['same_space'].append({'d':d,'N_before':n,'GD320_over_pool10000':ratios,
          'worst_error_gain_wins':sum(r['candidates']['gd320']['E_after']<r['candidates']['pool10000']['E_after'] for r in rows)})
axes[0].set_ylabel('Test maximum energy error / (1 / sqrt(12)) [%]')
axes[1].legend(fontsize=8)
fig.suptitle('Poisson with independent piecewise constant forcing: 4-seed medians',fontsize=11)
fig.tight_layout();fig.savefig(root/'piecewise_error_curves.png',dpi=180);fig.savefig(root/'piecewise_error_curves.pdf');plt.close(fig)

# Selected parameter shapes; actual greedy selections, no rescaling.
d=256;a=data[d];saved=np.load(root/f'piecewise_extended_d{d}.npz');x=(np.arange(d)+.5)/d
fig,axes=plt.subplots(4,2,figsize=(11,7),sharex=True,sharey=True)
for row,n in enumerate([1,8,32,64]):
    for col,method in enumerate(['gd320','pool10000']):
        z=saved[f'z_s1000_{method}'][n-1]
        axes[row,col].step(x,z,where='mid',color=colors[1 if col==0 else 3],lw=.9)
        diag=next(r for r in a['training'] if r['seed']==1000 and r['method']==method)['selected'][n-1]
        axes[row,col].set_title(f'{method}, round {n}; {diag["sign_changes"]} sign changes',fontsize=9)
        axes[row,col].set_ylim(-1.1,1.1);axes[row,col].grid(alpha=.15)
        summary['shape'].append({'d':d,'seed':1000,'method':method,**diag})
for ax in axes[-1]:ax.set_xlabel('Physical position x')
fig.suptitle('Chosen piecewise constant forcing, d = 256 (one shared seed)',fontsize=11)
fig.tight_layout();fig.savefig(root/'piecewise_selected_shapes.png',dpi=180);fig.savefig(root/'piecewise_selected_shapes.pdf');plt.close(fig)

# Meaningful numerical implementation checks.
checks={}
for d in data:
    a=np.load(root/f'piecewise_extended_d{d}.npz');B=a['B'];G=a['G']
    defects=[np.linalg.norm(a[k].T@a[k]-np.eye(a[k].shape[1]),2) for k in a.files if k.startswith('Q_')]
    checks[f'd{d}']={'gram_factor_relative_defect':float(np.linalg.norm(B.T@B-G)/np.linalg.norm(G)),
      'constant_load_norm_squared':float(np.ones(d)@G@np.ones(d)),
      'max_basis_orthogonality_defect':float(max(defects))}
summary['implementation_checks']=checks
(root/'piecewise_extended_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'artifacts':str(root),'implementation_checks':checks},indent=2))
