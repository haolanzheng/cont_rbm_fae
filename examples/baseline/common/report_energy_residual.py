"""Refresh only meeting-deck FEM settings, result tables and result figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/consistent_fem_plots')
from pathlib import Path
import json,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FixedLocator,FormatStrFormatter,NullFormatter
from .run_energy_residual import setup
setup()
from .run_consistent_fem import ROOT,OUT,TABLE_DIMS,METHODS,GRIDS
from .plot_dimension_sweep import STYLES
from .meeting_multi_curves import plot as sphere_ball_curves
FIG=OUT/'figs';FIG.mkdir(exist_ok=True);TEX=ROOT/'paper/slides_short/slides.tex'
from . import meeting_multi_curves
meeting_multi_curves.FIG=FIG
STYLE7=[('gd','GD','#24557A','-','o'),('pool256','Pool 256','#C17E3C','--','^'),('pool4096','Pool 4096','#438273','-.','D'),('pool10000','Pool 10000','#993F4D',':','s')]+[('grid1','Grid $h=1$','#555555','--','v'),('grid0.5','Grid $h=1/2$','#555555','-.','P'),('grid0.25','Grid $h=1/4$','#555555',':','X')]
BSTYLE=[('gd','Clipping','#245E91','-','o'),('active_face','Active face','#BC762C','--','^'),('normal_scaled','Adaptive normal updates','#477A68','-.','s')]

def save(fig,name,tight=False):
 for ext in ['pdf']:fig.savefig(FIG/f'{name}.{ext}',dpi=180,**({'bbox_inches':'tight'} if tight else {}))
 plt.close(fig)

def select(rows,dom,d,m):return [r for r in rows if (r['domain'],r['d'],r['method'])==(dom,d,m)]
def median(rows,dom,d,m,key):return float(np.median([r[key] for r in select(rows,dom,d,m)]))

def errors_and_ratios(rows):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 dims=list(range(8,65,4))
 for domains,name in [(['sphere','ball'],'test_width_ratio_vs_d'),(['cube'],'cube_test_width_ratio_vs_d')]:
  fig,axes=plt.subplots(1,len(domains),figsize=(11.2,4.3) if len(domains)==2 else (9,4.3),squeeze=False)
  for dom,ax in zip(domains,axes[0]):
   allvalues=[]
   for m,label,col,ls,mk in STYLES:
    values=[median(rows,dom,d,m,'ratio') for d in dims];allvalues+=values
    if m=='gd':label+=', $T='+('80' if dom=='cube' else '40')+'$'
    ax.plot(dims,values,label=label,color=col,ls=ls,marker=mk,ms=4.5,lw=2,mfc='white')
   ax.set(xlabel='Parameter dimension $d$',xticks=[8,16,24,32,40,48,56,64],title=dom.capitalize() if len(domains)==2 else None);ax.grid(alpha=.16)
   if dom!='cube':ax.axhline(1,color='#777777',lw=.8,ls=':',zorder=0);ax.set_ylim(.96,max(2.25,1.05*max(allvalues)))
  axes[0,0].set_ylabel('$E_N/d_N$' if len(domains)==2 else 'Test maximum / $d_N$')
  fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4 if len(domains)==2 else 2,fontsize=10,frameon=False)
  fig.subplots_adjust(left=.09,right=.985,top=.9,bottom=.25,wspace=.22);save(fig,name)
 sphere_ball_curves([r for r in rows if r['domain']!='cube' and 'errors' in r])
 fig,axs=plt.subplots(2,2,figsize=(10,5.4),sharey=False);handles=[];labels=[]
 for d,ax in zip([6,16,32,64],axs.flat):
  values=[]
  for m,label,col,ls,mk in STYLE7:
   rr=select(rows,'cube',d,m)
   if not rr:continue
   length=min(len(r['errors']) for r in rr);v=np.median([r['errors'][:length] for r in rr],axis=0);values.extend(v)
   line,=ax.semilogy(range(length),v,label=label,color=col,ls=ls,lw=1.7,marker=mk,markevery=max(1,length//7),ms=4,mfc='white',mew=1)
   if d==6:handles.append(line);labels.append(label)
  ax.set_title(f'$d={d}$',pad=3);ax.set_xlabel('$N$',labelpad=1);ax.grid(alpha=.18)
  positive=np.asarray(values);positive=positive[positive>0];lo,hi=positive.min(),positive.max();pad=max(.035,.08*np.log10(hi/lo));ax.set_ylim(lo/10**pad,hi*10**pad)
  if hi/lo<10:
   ticks=MaxNLocator(nbins=5).tick_values(lo,hi);ax.yaxis.set_major_locator(FixedLocator(ticks[(ticks>=lo/10**pad)&(ticks<=hi*10**pad)]));ax.yaxis.set_major_formatter(FormatStrFormatter('%g'));ax.yaxis.set_minor_formatter(NullFormatter())
 for ax in axs[:,0]:ax.set_ylabel('Test maximum error')
 fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9,frameon=False);fig.tight_layout(pad=.7,h_pad=.65,w_pad=1.5,rect=(0,.12,1,1));save(fig,'cube_test_error_vs_N_dimensions')
 fig,axs=plt.subplots(1,3,figsize=(12,4.4))
 for d,ax in zip([6,32,64],axs):
  for m,label,col,ls,mk in BSTYLE:
   v=np.median([r['errors'][:d//2+1] for r in select(rows,'cube',d,m)],axis=0)
   ax.semilogy(range(len(v)),v,label=label,color=col,ls=ls,lw=2.3,marker=mk,ms=5,mfc='white',markevery=max(1,(d//2)//4))
  ax.set(title=f'$d={d}$',xlabel='Reduced dimension $N$',xticks=([0,1,2,3] if d==6 else list(range(0,d//2+1,d//8))));ax.grid(alpha=.16)
  if d==6:
   lo,hi=ax.get_ylim();ticks=MaxNLocator(nbins=5).tick_values(lo,hi);ax.yaxis.set_major_locator(FixedLocator(ticks[(ticks>=lo)&(ticks<=hi)]));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_minor_formatter(NullFormatter())
 axs[0].set_ylabel('Maximum energy error');fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(left=.075,right=.99,top=.88,bottom=.24,wspace=.3);save(fig,'cube_boundary_updates_error_vs_N',True)

def probability_figures():
 records=json.loads((OUT/'probability_theory.json').read_text());dims=[8,16,32,48,64]
 plt.rcParams.update({'font.size':12,'axes.titlesize':14,'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8','savefig.facecolor':'#FAFAF8','axes.edgecolor':'#68727A','text.color':'#263039','axes.labelcolor':'#263039'})
 for name,key,xkey,xlabel,ylabel in [('step_size_probability_bounds','step_lower','alpha',r'Step scale $\alpha$',r'Analytical lower bound for $q_{40}$'),('iteration_probability_bounds','time_lower','T',r'Number of updates $T$',r'Analytical lower bound for $q_T$')]:
  fig,axes=plt.subplots(1,2,figsize=(11.6,3.9))
  for d,col,ls,mk in zip(dims,['#27647B','#B1813D','#527B61','#A55359','#64717D'],['-','--','-.',':',(0,(5,1,1,1))],['o','^','D','s','v']):
   for dom,ax in zip(['sphere','ball'],axes):
    r=next(r for r in records if (r['domain'],r['d'])==(dom,d));ax.plot(r[xkey],r[key],color=col,ls=ls,lw=2.4,label=f'd = {d}',marker=mk,ms=5,mfc='white',mew=1.1,markevery=max(1,len(r[xkey])//8))
  for dom,ax in zip(['Sphere','Ball'],axes):
   ax.set(title=dom,ylim=(0,1.025),xlabel=xlabel);ax.set_xlim(left=0);ax.grid(alpha=.17)
   if xkey=='T':ax.set_xlim(0,100);ax.axvline(40,color='#68727A',ls=':',lw=1,alpha=.65)
  axes[0].set_ylabel(ylabel);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.52,-.05));fig.subplots_adjust(wspace=.2,bottom=.18);save(fig,name,True)
 fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
 for dom,ax in zip(['sphere','ball'],axes):
  rr=[next(r for r in records if (r['domain'],r['d'])==(dom,d)) for d in dims]
  ax.plot(dims,[r['lower'] for r in rr],color='#245E91',lw=2.8,ls='--',marker='s',mfc='white',ms=6,label=r'GD: lower bound for $q_{40}(\tau)$')
  for m,c,ls,mk in [(256,'#BC762C',':','^'),(4096,'#477A68','-.','D'),(10000,'#B44E49','-','v')]:ax.plot(dims,[r['pool_upper'][str(m)] for r in rr],color=c,lw=2,ls=ls,marker=mk,mfc='white',ms=6,label=rf'Pool: upper bound for $P_{{{m}}}(\tau)$')
  ax.set(xticks=dims,title=dom.capitalize(),xlabel=r'Parameter dimension $d$',yscale='log');ax.set_ylim(top=1.5);ax.grid(alpha=.16)
 axes[0].set_ylabel('Probability bound');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,0),fontsize=11);fig.subplots_adjust(left=.085,right=.985,top=.91,bottom=.27,wspace=.2);save(fig,'probability_bounds_dimension',True)
 records=json.loads((OUT/'probability_empirical.json').read_text());floor=np.finfo(float).eps
 for domains,name in [(['sphere','ball'],'empirical_probability_dimension'),(['cube'],'cube_empirical_probability_dimension')]:
  fig,axes=plt.subplots(1,len(domains),figsize=(11.6,4.8) if len(domains)==2 else (9,4.3),sharey=True,squeeze=False)
  for dom,ax in zip(domains,axes[0]):
   for m,label,c,ls,mk in STYLES:
    if m=='gd':label='GD, $T='+('80' if dom=='cube' else '40')+'$'
    lows=[];highs=[]
    for d in dims:
     rr=[r for r in records if (r['domain'],r['d'],r['method'])==(dom,d,m)];assert len(rr)==8
     lows.append(np.mean([r['hit_lower'] for r in rr]));highs.append(np.mean([r['hit_upper'] for r in rr]))
    ax.plot(dims,np.maximum(lows,floor),color=c,lw=2.3,ls=ls,marker=mk,ms=7,mfc='white',mew=1.5,label=label)
    if lows!=highs:ax.fill_between(dims,np.maximum(lows,floor),np.maximum(highs,floor),color=c,alpha=.2)
   ax.set(title=dom.capitalize() if dom!='cube' else None,xlabel=r'Parameter dimension $d$',xticks=dims,yscale='log',ylim=(floor/3,2),yticks=[1,1e-4,1e-8,1e-12,1e-16]);ax.grid(alpha=.16)
  axes[0,0].set_ylabel('Empirical attainment frequency');fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.01),fontsize=11);fig.subplots_adjust(left=.10,right=.985,top=.91,bottom=.25,wspace=.20);save(fig,name,True)

def tables(rows):
 source=TEX.read_text();backup=OUT/'slides_before_update.tex'
 if not backup.exists():backup.write_text(source)
 times=json.loads((OUT/'timings.json').read_text())
 for title,domains,timing in [('Sphere and ball: final test errors across dimensions',['sphere','ball'],False),('Sphere and ball: wall time',['sphere','ball'],True),('Cube: final test errors across dimensions',['cube'],False),('Cube: wall time',['cube'],True)]:
  a=source.index('\\begin{frame}{'+title+'}');b=source.index('\\end{frame}',a);f=source[a:b]
  lines=f.splitlines();new=[];domain=domains[0];multi=len(domains)>1
  for line in lines:
   cells=line.removesuffix(r'\\').split('&');di=1 if multi else 0
   if len(cells)>di+1 and cells[di].strip().isdigit():
    d=int(cells[di]);domain=cells[0].strip().lower() if multi and cells[0].strip() else domain
    vals=[]
    for m in METHODS+GRIDS:
     rr=select(rows,domain,d,m)
     vals.append(None if not rr else (next(t['wall_ms'] for t in times if (t['domain'],t['d'],t['method'])==(domain,d,m)) if timing else float(np.median([r['test_max_error'] for r in rr]))))
    minimum=min(v for v in vals if v is not None);formatted=[]
    for v in vals:
     if v is None:formatted.append('---');continue
     if timing:label=f'{v:.2f}'
     elif v>=1e-4:label=f'{v:.5f}'
     else:
      mant,e=f'{v:.2e}'.split('e');label=mant+r'\!\times\!10^{'+str(int(e))+'}'
     if not timing and v==minimum:label=r'\mathbf{'+label+'}'
     formatted.append('$'+label+'$')
    new.append((' & '.join(cells[:di+1]))+' & '+' & '.join(formatted)+r'\\')
   elif r'$\dagger$' in line:
    if timing:new.append('Shared HF assembly and solves are excluded.')
   else:new.append(line)
  source=source[:a]+'\n'.join(new)+'\n'+source[b:]
 a=source.index(r'\begin{frame}{Cube: final errors for three boundary updates}');b=source.index(r'\end{frame}',a);f=source[a:b]
 for d in [6,8,16,32,48,64]:
  vals=[median(rows,'cube',d,m,'test_max_error') for m,*_ in BSTYLE];best=min(vals)
  newline=f'{d} & {d//2} & '+' & '.join((r'\textbf{'+f'{v:.5f}'+'}') if v==best else f'{v:.5f}' for v in vals)+r'\\'
  f=re.sub(r'^'+str(d)+r' &.*$',lambda m:newline,f,flags=re.M)
 source=source[:a]+f+source[b:]
 source=source.replace(r'\textbf{HF discretization.} Evaluate the analytic solution at the original P3 nodes $x_i$:',r'\textbf{HF discretization.} Solve the P3 finite-element equations:')
 source=source.replace(r'\mathbf u^{\rm HF}(z)=Sz,\qquad S_{ij}=\rho^{j-1}\psi_j(x_i).',r'AS=B,\qquad \mathbf u^{\rm HF}(z)=Sz.')
 source=source.replace('Analytic snapshots at $14{,}884$ P3 nodes; same CPU, float64, one thread.','P3 finite elements with $14{,}884$ degrees of freedom; same CPU, float64, one thread.')
 source=source.replace('Curves stop at the largest $N$ reached by all seeds.','')
 TEX.write_text(source)

def main():
 rows=json.loads((OUT/'results.json').read_text());assert all(r['rank']>=r['target_rank'] for r in rows)
 before=OUT/'figures_before';before.mkdir(exist_ok=True)
 for p in FIG.glob('*.pdf'):
  if not (before/p.name).exists():shutil.copy2(p,before/p.name)
 errors_and_ratios(rows);probability_figures()
 for folder in ['slides_short','slides_meeting_20261113']:
  global TEX
  TEX=ROOT/'paper'/folder/'slides.tex';tables(rows)
  for p in FIG.glob('*.pdf'):shutil.copy2(p,TEX.parent/'figs'/p.name)
 for p in FIG.glob('*.pdf'):
  target=ROOT/'paper/writeup/figs'/p.name
  if target.exists():shutil.copy2(p,target)
 print('Meeting deck tables and figures refreshed from consistent FEM runs.',flush=True)

if __name__=='__main__':main()
