"""Combine archived and added dimensions, plot test maximum / exact width."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/baseline_dimension_plot')
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STYLES=[('gd','GD','#245E91','-','o'),('pool256','Pool $M=256$','#BC762C','--','^'),('pool4096','Pool $M=4096$','#477A68','-.','D'),('pool10000','Pool $M=10000$','#B44E49',':','s')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cube',type=Path,required=True);p.add_argument('--sweep',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);args=p.parse_args()
 data=json.loads((args.sweep/'results.json').read_text());records=data['runs'].copy()
 old=json.loads((args.source/'maximum_test_errors.json').read_text())
 for r in old['runs']:
  records.append({'domain':r['domain'],'d':r['d'],'method':r['method'],'seed':r['seed'],'test_max_error':r['maximum_error'],'ratio':r['maximum_error']/.78**(r['d']//2)})
 cube=json.loads((args.cube/'results.json').read_text())
 for r in cube['runs']:
  if r['d'] not in [8,16,32,48,64] or r['method'].startswith('grid'):continue
  records.append({'domain':'cube','d':r['d'],'method':r['method'],'seed':r['seed'],'test_max_error':r['test_max_error'],'ratio':r['test_max_error_ratio']})
 dims=list(range(8,65,4));summary=[]
 for domain in ['sphere','ball','cube']:
  for method,*_ in STYLES:
   for d in dims:
    rr=[r for r in records if (r['domain'],r['d'],r['method'])==(domain,d,method)];assert len(rr)==8,(domain,d,method,len(rr))
    summary.append({'domain':domain,'d':d,'method':method,'median_ratio':float(np.median([r['ratio'] for r in rr])),'median_maximum_error':float(np.median([r['test_max_error'] for r in rr]))})
 (args.sweep/'combined_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 for domains,filename in [(['sphere','ball'],'test_width_ratio_vs_d'),(['cube'],'cube_test_width_ratio_vs_d')]:
  fig,axes=plt.subplots(1,len(domains),figsize=(11.2,4.3) if len(domains)==2 else (9,4.3),squeeze=False)
  for domain,ax in zip(domains,axes[0]):
   for method,label,col,ls,mk in STYLES:
    values=[next(r['median_ratio'] for r in summary if (r['domain'],r['d'],r['method'])==(domain,d,method)) for d in dims]
    label=label+(', $T=80$' if domain=='cube' else ', $T=40$') if method=='gd' else label
    ax.plot(dims,values,label=label,color=col,ls=ls,marker=mk,ms=4.5,lw=2,mfc='white')
   ax.set(xlabel='Parameter dimension $d$',xticks=[8,16,24,32,40,48,56,64],title=domain.capitalize() if len(domains)==2 else None);ax.grid(alpha=.16)
  axes[0,0].set_ylabel('Test maximum / $d_N$');h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4 if len(domains)==2 else 2,fontsize=10,frameon=False);fig.subplots_adjust(left=.09,right=.985,top=.9,bottom=.25,wspace=.22)
  for ext in ['pdf','png']:fig.savefig(args.figures/f'{filename}.{ext}',dpi=180)
  plt.close(fig)
 rows=[]
 for d in [8,16,32,48,64]:
  vals=[next(r['median_maximum_error'] for r in summary if (r['domain'],r['d'],r['method'])==('cube',d,m)) for m,*_ in STYLES]
  rows.append(str(d)+' & '+' & '.join(f'{x:.5f}' for x in vals)+r'\\')
 fragment=r'''\begin{frame}{Cube: final test errors across dimensions}
\small
$N=d/2$.
\begin{center}
\renewcommand{\arraystretch}{1.3}\setlength{\tabcolsep}{6pt}
\begin{tabular}{rrrrr}\toprule
$d$ & GD & Pool $256$ & Pool $4096$ & Pool $10000$\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{center}
\footnotesize
Maximum $A$-norm error on the shared vertex test set; median over eight training seeds.
\end{frame}

\begin{frame}{Cube: test error relative to width versus dimension}
$N=d/2$; $d=8,12,\ldots,64$.
\begin{figure}
\centering
\includegraphics[width=\linewidth,height=0.67\textheight,keepaspectratio]{figs/cube_test_width_ratio_vs_d.pdf}
\end{figure}
\footnotesize
Test-set maximum error divided by exact $d_N$; median over eight training seeds.
\end{frame}
'''
 (args.sweep/'cube_additions.tex').write_text(fragment)
 (args.sweep/'sphere_ball_addition.tex').write_text(r'''\begin{frame}{Test error relative to width versus dimension}
$N=d/2$; $d=8,12,\ldots,64$.
\begin{figure}
\centering
\includegraphics[width=\linewidth,height=0.67\textheight,keepaspectratio]{figs/test_width_ratio_vs_d.pdf}
\end{figure}
\footnotesize
Test-set maximum error divided by exact $d_N$; median over eight training seeds.
\end{frame}
''')
if __name__=='__main__':main()
