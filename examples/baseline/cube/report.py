"""Generate cube figures and a Beamer fragment from saved experiment records."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cube_baseline_matplotlib')
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLES=[('gd','GD, $T=40$','#245E91','-','o'),('pool256','Pool $M=256$','#BC762C','--','^'),('pool4096','Pool $M=4096$','#477A68','-.','D'),('pool10000','Pool $M=10000$','#B44E49',':','s')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);args=p.parse_args()
 data=json.loads((args.run/'results.json').read_text());runs=data['runs'];args.figures.mkdir(parents=True,exist_ok=True)
 steps=data['config']['updates']
 styles=[(m,f'GD, $T={steps}$' if m=='gd' else label,col,ls,mk) for m,label,col,ls,mk in STYLES]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 def select(d,m):return [r for r in runs if r['d']==d and r['method']==m]
 fig,ax=plt.subplots(figsize=(9,4.3))
 for m,label,col,ls,mk in styles:
  r=select(32,m);v=np.median([x['test_max_error_by_N'] for x in r],axis=0)
  ax.semilogy(range(len(v)),v,label=label,color=col,ls=ls,marker=mk,ms=5,lw=2,markevery=2,mfc='white')
 ax.set(xlabel='Reduced dimension $N$',ylabel='Maximum energy error',xticks=range(0,17,2));ax.grid(alpha=.16);ax.legend(frameon=False,ncol=2,fontsize=10)
 fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(args.figures/f'cube_test_error_vs_N_d32.{ext}',dpi=180)
 plt.close(fig)
 dims=[d for d in data['config']['dimensions'] if d>=8];freq=[]
 fig,ax=plt.subplots(figsize=(9,4.3))
 for m,label,col,ls,mk in styles:
  values=[]
  for d in dims:
   r=select(d,m);bounds=np.mean(np.array([x['indicator_quality_interval'] for x in r])>=.9,axis=0)
   freq.append({'d':d,'method':m,'frequency_interval':bounds.tolist(),'runs':len(r)});values.append(bounds)
  values=np.array(values);floor=np.finfo(float).eps
  ax.plot(dims,np.maximum(values[:,0],floor),label=label,color=col,ls=ls,marker=mk,ms=7,lw=2,mfc='white')
  if np.any(values[:,0]!=values[:,1]):ax.fill_between(dims,np.maximum(values[:,0],floor),np.maximum(values[:,1],floor),color=col,alpha=.2)
 ax.set(xlabel='Parameter dimension $d$',ylabel='Empirical attainment frequency',xticks=dims,yscale='log',ylim=(np.finfo(float).eps/3,2),yticks=[1,1e-4,1e-8,1e-12,1e-16]);ax.grid(alpha=.16);ax.legend(frameon=False,ncol=2,fontsize=10,loc='center right')
 fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(args.figures/f'cube_empirical_probability_dimension.{ext}',dpi=180)
 plt.close(fig)
 (args.run/'attainment_frequencies.json').write_text(json.dumps(freq,indent=2)+'\n')
 rows=[]
 for m,label in [('gd',f'GD, $T={steps}$'),('pool256','Pool $256$'),('pool4096','Pool $4096$'),('pool10000','Pool $10000$'),('grid1','Grid $h=1$'),('grid0.5','Grid $h=1/2$'),('grid0.25','Grid $h=1/4$')]:
  cells=[]
  for d in [6,32]:
   r=select(d,m)
   if not r:cells.append('---');continue
   rho=data['config']['rho'];N=d//2;exact_width=np.sqrt(sum(rho**(2*j) for j in range(N,d)))
   ratio=np.median([x['test_max_error_ratio'] for x in r]);ms=np.median([x['wall_ms'] for x in r])
   cells.append(f'{ratio:.4f} / {ms:.2f}')
  rows.append(label+' & '+' & '.join(cells)+r'\\')
 fragment=r'''
% BEGIN CUBE RESULTS (generated from saved cube runs)
\begin{frame}{Cube baseline: gradient descent update}
\small
$\mathcal Z=[-1,1]^d$, with the Euclidean metric.
\[
 J_n(z)=\frac{z^\top H_nz}{\ell},\qquad
 v_t=\frac{2H_nz^{(t)}}{\ell},\qquad
 \eta_t=\frac{\alpha}{\sqrt{1+\|v_t\|_2^2}}.
\]
\[
 z_i^{(t+1)}=\max\{-1,\min\{1,z_i^{(t)}+\eta_t(v_t)_i\}\},
 \qquad i=1,\ldots,d.
\]
The gradient step is clipped to the parameter interval in each coordinate.
$T=80$ gradient updates per greedy round.
\end{frame}


\begin{frame}{Cube: the Kolmogorov $N$-width}
\small
$\mathcal Z=[-1,1]^d$. For $0\le N<d$,
\textbf{Upper bound:} take $V=\operatorname{span}\{\psi_1,\ldots,\psi_N\}$.
\[
 \sup_{z\in\mathcal Z}\inf_{v\in V}\|u(\cdot;z)-v\|_{\mathfrak a}^2
 =\sup_{z\in\mathcal Z}\sum_{j>N}\rho^{2(j-1)}z_j^2
 =\sum_{j>N}\rho^{2(j-1)}.
\]
\textbf{Lower bound:} take any $N$-dimensional $V\subset H_0^1(D)$
and its energy-orthogonal projector $P_V$.
The largest error is at least the average over uniform vertices $z\in\{\pm1\}^d$:
\[
 \sup_{z\in\mathcal Z}\|(I-P_V)u(\cdot;z)\|_{\mathfrak a}^2
 \ge \mathbb E_z\|(I-P_V)u(\cdot;z)\|_{\mathfrak a}^2
 =\sum_{j=1}^d\rho^{2(j-1)}(1-\|P_V\psi_j\|_{\mathfrak a}^2).
\]
By Bessel's inequality, $0\le\|P_V\psi_j\|_{\mathfrak a}^2\le1$ and
$\sum_j\|P_V\psi_j\|_{\mathfrak a}^2\le N$: the captured weighted sum
is at most $\sum_{j=1}^N\rho^{2(j-1)}$.
\begin{block}{Cube}
\centering
$\displaystyle d_N^2=\sum_{j>N}\rho^{2(j-1)}=\rho^{2N}\frac{1-\rho^{2(d-N)}}{1-\rho^2}.$
\end{block}
\end{frame}

\begin{frame}{Cube: greedy error and wall time}
\small
$N=d/2$. Each entry: test-set maximum error / $d_N$, and wall time (ms).
\[
 \frac{\max_{z\in\mathcal Z_{\mathrm{test}}}
       \|\mathbf u(z)-\mathbf u_N(z)\|_A}{d_N}.
\]
\begin{center}
\footnotesize
\renewcommand{\arraystretch}{1.35}\setlength{\tabcolsep}{7pt}
\begin{tabular}{lrr}\toprule
Method & $d=6$, $N=3$ & $d=32$, $N=16$\\\midrule
'''+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{center}
\footnotesize
Eight-seed medians for GD and pools; same timing scope as above.
Exact analytic $d_N$; errors evaluated on the P3 discretization.
Vertex test set: exhaustive up to sign at $d=6$; $65{,}536$ sampled at $d=32$.
\end{frame}

\begin{frame}{Cube: test error versus $N$ at $d=32$}
Test set: $65{,}536$ vertices, shared by all methods and all $N$.
\begin{figure}
\centering
\includegraphics[width=\linewidth,height=0.66\textheight,keepaspectratio]{figs/cube_test_error_vs_N_d32.pdf}
\end{figure}
\footnotesize
Maximum $A$-norm error on the fixed vertex test set; median over eight training seeds.
\end{frame}

\begin{frame}{Cube: empirical probability versus dimension}
\begin{figure}
\centering
\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{figs/cube_empirical_probability_dimension.pdf}
\end{figure}
\footnotesize
Final dimension $N=d/2$; eight runs per method; $\gamma=0.9$. Bands reflect bounds on $\max_{z\in\mathcal Z}J_{N-1}(z)$.
\end{frame}
% END CUBE RESULTS
'''
 extra=args.run/'additional_result_frames.tex'
 if extra.exists():fragment=fragment.replace(r'\begin{frame}{Cube: empirical probability versus dimension}',extra.read_text()+'\n'+r'\begin{frame}{Cube: empirical probability versus dimension}')
 (args.run/'cube_slides.tex').write_text(fragment.replace('$T=80$ gradient',f'$T={steps}$ gradient'))
if __name__=='__main__':main()
