"""Continue saved sphere/ball baseline runs to rank d-1 for the meeting deck.

No HF solves or changes to the original experiment. Retains saved selections,
initialization streams, residual matrices, and fixed test sets. Run from the
repository root with PYTHONPATH=src:. and the CPU environment.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/meeting_mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.baseline.geometry_cpu import CoordinateROM,rom_options,uniform_parameters,search,pool_search
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'runs/analytic_hf_a06_t40_20261022'
OUT=ROOT/'runs/meeting_20261113'
FIG=ROOT/'paper/slides_meeting_20261113/figs'
STYLE=[('GD','#215f91','-'),('Pool 256','#c17818','--'),('Pool 4096','#4b8061','-.'),('Pool 10000','#a34d42',':')]
def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    records=json.loads((OLD/'results.json').read_text())['runs']
    model=np.load(OLD/'model_d32.npz'); allruns=[]
    for domain in ('sphere','ball'):
        test=np.load(OLD/f'test_{domain}_d32.npz')['test_parameters']
        methods=['exp_rgd40' if domain=='sphere' else 'normalized_pg40','fixed_pool256','fixed_pool4096','fixed_pool10000']
        for method in methods:
            for seed in range(1000,1008):
                old=next(r for r in records if r['dimension']==32 and r['domain']==domain and r['method']==method and r['seed']==seed)
                rom=CoordinateROM(model['G0'],model['H0'],**rom_options(model))
                rng=np.random.default_rng(seed)
                isgd=method==methods[0]
                pool=None if isgd else uniform_parameters(np.random.default_rng(seed+1_000_003),int(method.replace('fixed_pool','')),32,domain)
                errors=[]; parameters=[];stop=None
                for n in range(32):
                    H,G=rom.matrices()
                    errors.append(float(np.sqrt(max(np.einsum('ij,ij->i',test@G,test).max(),0))))
                    if n==31:break
                    init=uniform_parameters(rng,1,32,domain)[0] if isgd else None
                    if n<len(old['rounds']):
                        z=np.array(old['rounds'][n]['parameter'])
                    elif isgd:
                        z,_=search(H,init,steps=40,domain=domain,alpha=.6,ball_step_rule='normalized')
                    else:z,_=pool_search(H,pool)
                    try:rom.append(z)
                    except RuntimeError as e:stop=str(e);break
                    parameters.append(z.tolist())
                row=dict(domain=domain,method=method,seed=seed,errors=errors,parameters=parameters,stop=stop,orthogonality=rom.orthogonality_defect())
                allruns.append(row)
            print(domain,method,'rank',min(len(r['errors'])-1 for r in allruns if r['domain']==domain and r['method']==method),flush=True)
    (OUT/'extended_curves.json').write_text(json.dumps(allruns,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(10.4,4.1),sharey=True)
    for ax,domain in zip(axs,('sphere','ball')):
        methods=['exp_rgd40' if domain=='sphere' else 'normalized_pg40','fixed_pool256','fixed_pool4096','fixed_pool10000']
        for method,(label,color,ls) in zip(methods,STYLE):
            runs=[r for r in allruns if r['domain']==domain and r['method']==method]
            length=min(len(r['errors']) for r in runs)
            val=np.median([r['errors'][:length] for r in runs],axis=0)
            ax.semilogy(np.arange(length),val,label=label,color=color,ls=ls,lw=2)
        ax.set(title=domain.capitalize()+', $d=32$',xlabel='$N$',xlim=(0,31),xticks=[0,8,16,24,31]);ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Maximum test error');axs[1].legend(frameon=False,fontsize=10)
    fig.tight_layout();fig.savefig(FIG/'test_error_vs_N_d32.pdf');plt.close(fig)
    cube=ROOT/'runs/cube_fixed_clip_N_20261102'
    base=json.loads((cube/'results.json').read_text());comp=json.loads((cube/'comparison_results.json').read_text())
    print('cube base keys',base[0].keys(),flush=True)
    (OUT/'settings.json').write_text(json.dumps(dict(source=str(OLD),dimensions=[32],alpha=.6,T=40,rank=31,test_size=16384,seeds=list(range(1000,1008)),metric='fixed-test maximum A-norm error'),indent=2))
if __name__=='__main__':main()
