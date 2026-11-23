"""Reevaluate saved baseline spaces: maximum absolute A-error on fixed tests.

No training. Sphere/ball retain their stored test points; cube uses only its
stored vertices. Each seed's maximum is taken before the median across seeds.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/baseline_maximum_plot')
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..geometry_cpu import CoordinateROM,rom_options
from ..cube.search import scores
STYLES=[('gd','GD, $T=40$','#245E91','-','o'),('pool256','Pool $M=256$','#BC762C','--','^'),('pool4096','Pool $M=4096$','#477A68','-.','D'),('pool10000','Pool $M=10000$','#B44E49',':','s')]


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cube',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);args=p.parse_args()
 data=json.loads((args.source/'results.json').read_text());records=[]
 for domain in ['sphere','ball']:
  for d in data['config']['dimensions']:
   model=np.load(args.source/f'model_d{d}.npz');test=np.load(args.source/f'test_{domain}_d{d}.npz');Z=test['test_parameters'] if 'test_parameters' in test else test['parameters']
   for method,_,_,_,_ in STYLES:
    key=('exp_rgd40' if domain=='sphere' else 'normalized_pg40') if method=='gd' else 'fixed_'+method
    for r in [x for x in data['runs'] if (x['domain'],x['dimension'],x['method'])==(domain,d,key)]:
     rom=CoordinateROM(model['G0'],model['H0'],**rom_options(model));curve=[float(np.sqrt(scores(model['G0'],Z).max()))]
     for step in r['rounds']:
      rom.append(np.array(step['parameter']))
      if d==32:
       _,G=rom.matrices();curve.append(float(np.sqrt(max(0,scores(G,Z).max()))))
     _,G=rom.matrices();final=float(np.sqrt(max(0,scores(G,Z).max())))
     records.append({'domain':domain,'d':d,'method':method,'seed':r['seed'],'maximum_error_by_N':curve if d==32 else None,'maximum_error':final,'test_count':len(Z)})
   print('Evaluated',domain,d,flush=True)
 out=args.source/'maximum_test_errors.json';out.write_text(json.dumps({'metric':'maximum absolute A-energy error on shared fixed test points','runs':records},indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11.2,4.3),sharey=True)
 for domain,ax in zip(['sphere','ball'],axes):
  for method,label,col,ls,mk in STYLES:
   rr=[r for r in records if (r['domain'],r['d'],r['method'])==(domain,32,method)];curve=np.median([r['maximum_error_by_N'] for r in rr],axis=0)
   ax.semilogy(range(17),curve,label=label,color=col,ls=ls,marker=mk,ms=5,markevery=4,lw=2,mfc='white')
  ax.set(title=domain.capitalize(),xlabel='Reduced dimension $N$',xticks=[0,4,8,12,16]);ax.grid(alpha=.16)
 axes[0].set_ylabel('Maximum energy error');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=10,frameon=False);fig.subplots_adjust(left=.09,right=.99,bottom=.25,top=.9,wspace=.15)
 for ext in ['pdf','png']:fig.savefig(args.figures/f'test_error_vs_N_d32.{ext}',dpi=180)
 plt.close(fig)
 cube=json.loads((args.cube/'results.json').read_text());backup=args.cube/'results_before_maximum_curves.json'
 if not backup.exists():backup.write_bytes((args.cube/'results.json').read_bytes())
 model=np.load(args.source/'model_d32.npz');Z=np.load(args.cube/'test_vertices_d32.npz')['parameters']
 for r in cube['runs']:
  if r['d']!=32:continue
  Q=np.load(args.cube/f"basis_d32_{r['method']}_s{r['seed']}.npz")['Q'];rom=CoordinateROM(model['G0'],model['H0'],**rom_options(model));curve=[]
  for n in range(r['N']+1):
   rom.Q=Q[:,:n];_,G=rom.matrices();curve.append(float(np.sqrt(max(0,scores(G,Z).max()))))
  assert np.isclose(curve[-1],r['test_max_error'],rtol=1e-12)
  r['test_max_error_by_N']=curve
 cube['displayed_test_metric']='maximum absolute energy error on shared vertices';cube['config']['test_sampling']='vertices_max'
 (args.cube/'results.json').write_text(json.dumps(cube,indent=2)+'\n')
 print('Cube maximum curves evaluated',flush=True)
 rows=[]
 for domain in ['sphere','ball']:
  for idx,d in enumerate(data['config']['dimensions']):
   v=[np.median([r['maximum_error'] for r in records if (r['domain'],r['d'],r['method'])==(domain,d,m)]) for m,*_ in STYLES]
   rows.append((domain.capitalize() if idx==0 else '')+' & '+str(d)+' & '+' & '.join(f'{x:.5f}' for x in v)+r'\\')
  if domain=='sphere':rows.append(r'\midrule')
 (args.source/'maximum_test_error_table.tex').write_text('\n'.join(rows)+'\n')
if __name__=='__main__':main()
