"""Final test maxima with uniform meshes included at d=4,6; no retraining."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from ..geometry_cpu import CoordinateROM,rom_options,uniform_parameters
from ..cube.search import scores
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'runs/analytic_hf_a06_t40_20261022'
OUT=ROOT/'runs/final_error_tables_with_mesh_20261102'
METHODS=['gd','pool256','pool4096','pool10000','grid1','grid0.5','grid0.25']

def main():
 OUT.mkdir(exist_ok=True)
 high=json.loads((SOURCE/'maximum_test_errors.json').read_text())['runs'];low=json.loads((SOURCE/'low_dimension_results.json').read_text())['runs'];records=high.copy()
 for domain in ['sphere','ball']:
  for d in [4,6]:
   a=np.load(SOURCE/f'model_d{d}.npz');G0=a['rotated_G0'];H0=a['rotated_H0'];opts={k:a['rotated_'+k] for k in rom_options(a)}
   if d==6:Z=np.load(SOURCE/f'fixed_final_spaces_{domain}_d6.npz')['test_parameters']
   else:Z=uniform_parameters(np.random.default_rng(202610180+100*d+(domain=='ball')),16384,d,domain)
   np.savez_compressed(OUT/f'test_{domain}_d{d}.npz',parameters=Z)
   for method in METHODS:
    for r in [r for r in low if (r['domain'],r['dimension'],r['method'],r['orientation'])==(domain,d,method,'rotated')]:
     rom=CoordinateROM(G0,H0,**opts)
     for step in r['rounds'][:d//2]:rom.append(step['parameter'])
     _,G=rom.matrices();value=float(np.sqrt(max(0,scores(G,Z).max())))
     if d==6:
      saved=np.load(SOURCE/f'fixed_final_spaces_{domain}_d6.npz');Q=saved[f"Q_{method}_seed{r['seed']}"];P=np.eye(d)-Q@(Q.T@saved['cross']);check=np.sqrt(scores(P.T@saved['G0']@P,Z).max());assert np.isclose(value,check,rtol=1e-10)
     records.append({'domain':domain,'d':d,'method':method,'seed':r['seed'],'maximum_error':value,'test_count':len(Z),'orientation':'rotated'})
 cube=json.loads((ROOT/'runs/cube_slides_a06_t80_20261102/results.json').read_text())['runs']
 records.extend({'domain':'cube','d':r['d'],'method':r['method'],'seed':r['seed'],'maximum_error':r['test_max_error']} for r in cube)
 (OUT/'records.json').write_text(json.dumps(records,indent=2)+'\n')
 for domains,title,filename in [(['sphere','ball'],'Final reduced models: test errors across dimensions','sphere_ball.tex'),(['cube'],'Cube: final test errors across dimensions','cube.tex')]:
  rows=[]
  for domain in domains:
   for idx,d in enumerate([4,6,8,16,32,48,64]):
    vals=[]
    for method in METHODS:
     r=[x for x in records if (x['domain'],x['d'],x['method'])==(domain,d,method)]
     expected=1 if method.startswith('grid') else 8
     assert len(r)==(expected if d<=6 or not method.startswith('grid') else 0),(domain,d,method,len(r))
     vals.append(f"{np.median([x['maximum_error'] for x in r]):.5f}" if r else '---')
    rows.append(((domain.capitalize() if idx==0 else '')+' & ' if len(domains)>1 else '')+str(d)+' & '+' & '.join(vals)+r'\\')
   if domain=='sphere':rows.append(r'\midrule')
  frame=r'\begin{frame}{'+title+r'''}
\small
$N=d/2$. Test-set maximum $A$-norm error (without division by $d_N$).
\begin{center}
\scriptsize
\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{3pt}
\begin{tabular}{'''+('l' if len(domains)>1 else '')+'rrrrrrrr'+r'''}\toprule
'''+('Domain & ' if len(domains)>1 else '')+r'''$d$ & GD & Pool $256$ & Pool $4096$ & Pool $10000$
& \multicolumn{3}{c}{Uniform mesh}\\
'''+(' & ' if len(domains)>1 else '')+r''' & & & & & $h=1$ & $h=1/2$ & $h=1/4$\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{center}
\footnotesize
Shared test sets; eight-seed medians for GD and pools.
Uniform meshes are tested only at $d=4,6$.
\end{frame}'''
  (OUT/filename).write_text(frame+'\n')
if __name__=='__main__':main()
