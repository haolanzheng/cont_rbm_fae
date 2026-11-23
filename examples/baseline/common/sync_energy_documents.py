"""Synchronize existing writeup/notebook result blocks from energy-dual runs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,re
import numpy as np
from .run_energy_residual import ROOT,OUT

def tabular(source,start):
    a=source.index(r'\begin{tabular}',start);b=source.index(r'\end{tabular}',a)+len(r'\end{tabular}')
    return a,b,source[a:b]

def replace_table(source,label,table):
    pos=source.index(r'\label{'+label+'}');start=source.rfind(r'\begin{table}',0,pos)
    a,b,_=tabular(source,start);return source[:a]+table+source[b:]

def main():
    rows=json.loads((OUT/'results.json').read_text());timings=json.loads((OUT/'timings.json').read_text())
    def med(dom,d,m,key='test_max_error'):
        return float(np.median([r[key] for r in rows if (r['domain'],r['d'],r['method'])==(dom,d,m)]))
    def fmt(v):
        return f'{v:.5f}' if v>=1e-4 else (lambda a,e:a+r'\times10^{'+str(int(e))+'}')(*f'{v:.2e}'.split('e'))
    slides=(ROOT/'paper/slides_short/slides.tex').read_text()
    p=ROOT/'paper/writeup/writeup.tex';s=p.read_text()
    for label,title in [('tab:sphere-time','Sphere and ball: wall time'),('tab:cube-time','Cube: wall time'),('tab:boundary','Cube: final errors for three boundary updates')]:
        _,_,table=tabular(slides,slides.index(r'\begin{frame}{'+title+'}'))
        s=replace_table(s,label,table)
    a=s.index('At $d=16$, $N=8$, the median maximum test errors');b=s.index('\n\n',a)
    s=s[:a]+('At $d=16$, $N=8$, the median maximum test errors for GD and the $10{,}000$-point pool are '
        f"${fmt(med('sphere',16,'gd'))}$ and ${fmt(med('sphere',16,'pool10000'))}$ on the sphere, and "
        f"${fmt(med('ball',16,'gd'))}$ and ${fmt(med('ball',16,'pool10000'))}$ on the ball. "
        'At $d=64$, $N=32$, the corresponding errors are '
        f"${fmt(med('sphere',64,'gd'))}$ and ${fmt(med('sphere',64,'pool10000'))}$ on the sphere, and "
        f"${fmt(med('ball',64,'gd'))}$ and ${fmt(med('ball',64,'pool10000'))}$ on the ball.")+s[b:]
    a=s.index('On the cube, ',s.index(r'\subsubsection{Cube}'));b=s.index('\n\n',a)
    s=s[:a]+('On the cube, at $d=16$, $N=8$, clipping gives a median maximum test error of '
        f"${fmt(med('cube',16,'gd'))}$, compared with ${fmt(med('cube',16,'pool4096'))}$ for the $4096$-point pool. "
        'At $d=64$, $N=32$, the corresponding values are '
        f"${fmt(med('cube',64,'gd'))}$ and ${fmt(med('cube',64,'pool4096'))}$. "
        r'The comparison of the three boundary updates is reported in \cref{tab:boundary}.')+s[b:]
    p.write_text(s)
    # Retain the notebook's existing relative-error diagnostic, recomputed.
    rms=[];curves=[]
    for dom in ['sphere','ball']:
        for d in [8,16,32,48,64]:
            a=dict(np.load(OUT/f'model_d{d}.npz'));C=a['C'];Z=np.load(OUT/f'test_{dom}_d{d}.npz')['parameters']
            F=Z@C.T;den=np.sum(F*F,axis=1)
            for row in [r for r in rows if (r['domain'],r['d'])==(dom,d) and r['method'] in ['gd','pool256','pool4096','pool10000']]:
                Q=np.load(OUT/f"basis_{dom}_d{d}_{row['method']}_s{row['seed']}.npz")['Q']
                ns=range(d) if d==32 else [d//2];values=[]
                for n in ns:
                    E=C-Q[:,:n]@(Q[:,:n].T@C);res=Z@E.T
                    values.append(float(np.sqrt(np.mean(np.sum(res*res,axis=1)/den))))
                rms.append(dict(domain=dom,d=d,method=row['method'],seed=row['seed'],value=values[d//2] if d==32 else values[0]))
                if d==32:curves.append(dict(domain=dom,method=row['method'],seed=row['seed'],values=values))
    (OUT/'relative_error_checks.json').write_text(json.dumps(rms,indent=2))
    p=ROOT/'paper/continuous_greedy_manifold_attainment.tex';s=p.read_text()
    pos=s.index(r'\label{tab:baseline-error-time}');start=s.rfind(r'\begin{table}',0,pos);a,b,t=tabular(s,start)
    methods=[('GD, $T=40$','gd'),('Pool $256$','pool256'),('Pool $4096$','pool4096'),('Pool $10000$','pool10000'),('Grid $h=1$','grid1'),('Grid $h=1/2$','grid0.5'),('Grid $h=1/4$','grid0.25')]
    for label,m in methods:
        cells=[]
        for d,dom in [(6,'sphere'),(6,'ball'),(32,'sphere'),(32,'ball')]:
            rr=[r for r in rows if (r['domain'],r['d'],r['method'])==(dom,d,m)]
            if not rr:cells.append('---');continue
            tm=next(x['wall_ms'] for x in timings if (x['domain'],x['d'],x['method'])==(dom,d,m))
            cells.append(f'{med(dom,d,m,"ratio"):.4f} / {tm:.2f}')
        t=re.sub('^'+re.escape(label)+r' &.*$',lambda _:label+' & '+' & '.join(cells)+r'\\',t,flags=re.M)
    s=s[:a]+t+s[b:]
    start=s.index(r'\begin{tabular}{lrrrrr}',s.index(r'\label{fig:baseline-common-test}'));a,b,t=tabular(s,start)
    dom='sphere';lines=[]
    for line in t.splitlines():
        cells=line.removesuffix(r'\\').split('&')
        if len(cells)==6 and cells[1].strip().isdigit():
            if cells[0].strip():dom=cells[0].strip().lower()
            d=int(cells[1]);values=[float(np.median([r['value'] for r in rms if (r['domain'],r['d'],r['method'])==(dom,d,m)])) for m in ['gd','pool256','pool4096','pool10000']]
            line=cells[0]+' & '+str(d)+' & '+' & '.join('$'+fmt(v)+'$' for v in values)+r'\\'
        lines.append(line)
    s=s[:a]+'\n'.join(lines)+s[b:];s=s.replace('\\caption{$E_N/d_N$ / wall time in milliseconds, $N=d/2$.','\\caption{$E_N/d_N$ / wall time in milliseconds, $N=d/2$, with the analytic width $d_N=\\rho^N$.')
    p.write_text(s)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .meeting_multi_curves import STYLE,MARKERS
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.1),sharey=True)
    for dom,ax in zip(['sphere','ball'],axes):
        for m,label,col,ls in STYLE[:4]:
            values=np.median([r['values'] for r in curves if (r['domain'],r['method'])==(dom,m)],axis=0)
            ax.semilogy(range(len(values)),values,color=col,ls=ls,marker=MARKERS[m],markevery=4,mfc='white',ms=4,label=label)
        ax.set(title=dom.capitalize()+', $d=32$',xlabel='$N$');ax.grid(alpha=.18)
    axes[0].set_ylabel('Root mean square relative energy error')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(ROOT/'paper/slides_short/figs/test_error_vs_N_d32.pdf');plt.close(fig)
    print('Updated writeup and notebook results, including the notebook relative-error diagnostic.')

if __name__=='__main__':main()
