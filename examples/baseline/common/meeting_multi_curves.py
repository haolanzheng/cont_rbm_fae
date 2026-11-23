"""Extend saved sphere/ball ROMs to d-1 at d=6,16,32,64.
Retain existing initial points, pools, test sets and original low-d rotation.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/meeting_mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.baseline.geometry_cpu import CoordinateROM,rom_options,uniform_parameters,search,pool_search
from examples.baseline.run_extended_comparison_cpu import tensor_candidates
ROOT=Path(__file__).resolve().parents[3];OLD=ROOT/'runs/analytic_hf_a06_t40_20261022';OUT=ROOT/'runs/meeting_20261113';FIG=ROOT/'paper/slides_meeting_20261113/figs'
STYLE=[('gd','GD, $T=40$','#24557A','-'),('pool256','Pool 256','#C17E3C','--'),('pool4096','Pool 4096','#438273','-.'),('pool10000','Pool 10000','#993F4D',':'),('grid1','Grid $h=1$','#555555','--'),('grid0.5','Grid $h=1/2$','#555555','-.'),('grid0.25','Grid $h=1/4$','#555555',':')]
MARKERS={'gd':'o','pool256':'^','pool4096':'D','pool10000':'s','grid1':'v','grid0.5':'P','grid0.25':'X'}
def main():
 high=json.loads((OLD/'results.json').read_text())['runs'];low=json.loads((OLD/'low_dimension_results.json').read_text())['runs'];rows=[]
 cache=json.loads((OUT/'extended_curves.json').read_text())
 for d in (6,16,32,64):
  model=np.load(OLD/f'model_d{d}.npz')
  G0=model['rotated_G0'] if d==6 else model['G0'];H0=model['rotated_H0'] if d==6 else model['H0'];opts={k:model['rotated_'+k] for k in rom_options(model)} if d==6 else rom_options(model)
  for dom in ('sphere','ball'):
   Z=np.load(OLD/(f'fixed_final_spaces_{dom}_d6.npz' if d==6 else f'test_{dom}_d{d}.npz'))['test_parameters']
   for method,*_ in STYLE:
    if method.startswith('grid') and d!=6:continue
    pool=tensor_candidates(d,float(method[4:]),dom)[0] if method.startswith('grid') else None
    group=[]
    for seed in ([0] if method.startswith('grid') else range(1000,1008)):
     key=('exp_rgd40' if dom=='sphere' else 'normalized_pg40') if method=='gd' else 'fixed_'+method
     if d==32:
      c=next(r for r in cache if (r['domain'],r['method'],r['seed'])==(dom,key,seed));row=dict(c,d=d,method=method);rows.append(row);group.append(row);continue
     r=next(r for r in (low if d==6 else high) if r['domain']==dom and r['dimension']==d and r['method']==(method if d==6 else key) and r['seed']==seed and (d!=6 or r['orientation']=='rotated'))
     rom=CoordinateROM(G0,H0,**opts);rng=np.random.default_rng(seed)
     if method.startswith('pool'):pool=uniform_parameters(np.random.default_rng(seed+1_000_003),int(method[4:]),d,dom)
     errors=[];parameters=[];stop=None
     for n in range(d):
      H,G=rom.matrices();errors.append(float(np.sqrt(max(np.einsum('ij,ij->i',Z@G,Z).max(),0))))
      if n==d-1:break
      init=uniform_parameters(rng,1,d,dom)[0] if method=='gd' else None
      if n<len(r['rounds']):z=np.array(r['rounds'][n]['parameter'])
      elif method=='gd':z,_=search(H,init,steps=40,domain=dom,alpha=.6,ball_step_rule='normalized')
      else:z,_=pool_search(H,pool)
      try:rom.append(z)
      except RuntimeError as e:stop=str(e);break
      parameters.append(z.tolist())
     row=dict(d=d,domain=dom,method=method,seed=seed,errors=errors,parameters=parameters,stop=stop)
     rows.append(row);group.append(row)
    print(d,dom,method,'rank',min(len(x['errors'])-1 for x in group),flush=True)
    (OUT/'multi_dimension_curves.json').write_text(json.dumps(rows,indent=2))
 plot(rows)

def plot(rows):
 plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,4,figsize=(13.2,6.1),sharey='col');handles=[];labels=[]
 for ri,dom in enumerate(('sphere','ball')):
  for ci,d in enumerate((6,16,32,64)):
   ax=axs[ri,ci]
   for method,label,col,ls in STYLE:
    rr=[r for r in rows if (r['d'],r['domain'],r['method'])==(d,dom,method)]
    if not rr:continue
    length=min(len(r['errors']) for r in rr);v=np.median([r['errors'][:length] for r in rr],axis=0)
    line,=ax.semilogy(range(length),v,label=label,color=col,ls=ls,lw=1.7,marker=MARKERS[method],markevery=max(1,length//7),ms=3.7,mfc='white',mew=1)
    if ri==ci==0:handles.append(line);labels.append(label)
   ax.set_title(f'{dom.capitalize()}, $d={d}$',fontsize=11);ax.set_xlabel('$N$',labelpad=1);ax.set_xlim(0,d-1);ax.grid(alpha=.18)
   ax.set_xticks([0,d//2,d-1]);
  axs[ri,0].set_ylabel('Test maximum error')
 fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=10,frameon=False)
 fig.tight_layout(pad=.7,h_pad=1.0,w_pad=.8,rect=(0,.12,1,1))
 for ext in ('pdf','png'):fig.savefig(FIG/f'test_error_vs_N_dimensions.{ext}',dpi=180)
 print('Done',len(rows),'runs',flush=True)
if __name__=='__main__':
 import sys
 if '--plot-only' in sys.argv:plot(json.loads((OUT/'multi_dimension_curves.json').read_text()))
 else:main()
