"""Render the existing baseline figures from the analytic-solution rerun."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/analytic_baseline_matplotlib')
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import betaincc
from .geometry_cpu import ball_normalized_probability_lower
from .run_extended_comparison_cpu import sphere_probability_lower
from .run_analytic_baseline import DIMS,OUT,FIG,ROOT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'axes.labelsize':12,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#FAFAF8',
 'axes.facecolor':'#FAFAF8','savefig.facecolor':'#FAFAF8','axes.edgecolor':'#68727A',
 'text.color':'#263039','axes.labelcolor':'#263039','pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    for ext in ['pdf','png']:
        path=FIG/f'{name}.{ext}'
        fig.savefig(path,dpi=180,bbox_inches='tight')
        if name in {'probability_bounds_dimension','step_size_probability_bounds','iteration_probability_bounds'}:
            shutil.copy2(path,ROOT/'paper/slides_short/figs'/path.name)
    plt.close(fig)
def main():
    rows=json.loads((OUT/'probability_curves.json').read_text())
    theory_rows=rows
    displayed=[r for r in theory_rows if r['dimension'] in DIMS and r['rank']==r['dimension']//2]
    assert {r['dimension'] for r in displayed}==set(DIMS)
    assert all(r['lambda2_ratio']<r['gamma']**2 for r in displayed), 'Displayed pool bounds require the stated spectral gap'
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
    for dom,ax,method in zip(['sphere','ball'],axes,['exp_rgd','normalized_pg']):
        rr=sorted([r for r in theory_rows if r['domain']==dom and r['dimension'] in DIMS and r['rank']==r['dimension']//2],key=lambda r:r['dimension'])
        dims=[r['dimension'] for r in rr];x=np.asarray(dims)
        ax.plot(x,[r['methods'][method]['snapshots'][-1]['analytic_lower'] for r in rr],color='#245E91',lw=2.8,ls='--',marker='s',mfc='white',ms=6,label=r'GD: lower bound for $q_{40}(\tau)$')
        for m,c,ls,mk in [(256,'#BC762C',':','^'),(4096,'#477A68','-.','D'),(10000,'#B44E49','-','v')]:
            ps=np.array([r['direct_analytic_upper'] for r in rr]);upper=np.ones_like(ps);mask=ps<1
            upper[mask]=-np.expm1(m*np.log1p(-ps[mask]))
            ax.plot(x,upper,color=c,lw=2,ls=ls,marker=mk,mfc='white',ms=6,label=rf'Pool: upper bound for $P_{{{m}}}(\tau)$')
        ax.set_xticks(x,dims);ax.set_title(dom.capitalize());ax.set_xlabel(r'Parameter dimension $d$')
        ax.set_yscale('log');ax.set_ylim(top=1.5);ax.grid(alpha=.16,which='major')
    axes[0].set_ylabel('Probability bound');h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,0),fontsize=11)
    fig.subplots_adjust(left=.085,right=.985,top=.91,bottom=.27,wspace=.2);save(fig,'probability_bounds_dimension')
    from .plot_greedy_attainment import render
    render(OUT,FIG)
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.9));records=[]
    for d,color,style in zip(DIMS,['#27647B','#B1813D','#527B61','#A55359','#64717D'],['-','--','-.',':',(0,(5,1,1,1))]):
        for dom,ax in zip(['sphere','ball'],axes):
            H=np.load(OUT/f'reference_d{d}_n{d//2}.npz')['H'];sigma=np.linalg.eigvalsh(H)[-1]/np.linalg.norm(H,'fro')
            step=np.linspace(0,1,301)
            if dom=='sphere':
                c=1+2*step*sigma*(1-.9**2)/np.sqrt(1+sigma**2)*(1-step**2*sigma**2/(6*(1+sigma**2))-step*.9**2*sigma);b=(d-1)/2
            else:
                eta=step/np.sqrt(1+4*.9**2*sigma**2)
                c=(1+2*eta*sigma)/np.sqrt(1+4*.9**2*eta*sigma*(1+eta*sigma));b=(d+1)/2
            q=betaincc(.5,b,.9**2*c**(-80))
            ax.plot(step,q,color=color,ls=style,lw=2.4,label=f'd = {d}')
            records.append({'domain':dom,'d':d,'rank':d//2,'sigma':float(sigma),'steps':step.tolist(),'lower_bound':q.tolist()})
    for ax,dom in zip(axes,['Sphere','Ball']):
        ax.set_title(dom);ax.set_ylim(0,1.025);ax.set_xlim(left=0);ax.grid(alpha=.17);ax.set_xlabel(r'Step scale $\alpha$')
    axes[0].set_ylabel(r'Analytical lower bound for $q_{40}$');h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.52,-.05))
    fig.subplots_adjust(wspace=.2,bottom=.18);save(fig,'step_size_probability_bounds')
    (OUT/'step_size_bounds.json').write_text(json.dumps(records,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.9));records=[]
    times=np.arange(101)
    for d,color,style in zip(DIMS,['#27647B','#B1813D','#527B61','#A55359','#64717D'],['-','--','-.',':',(0,(5,1,1,1))]):
        H=np.load(OUT/f'reference_d{d}_n{d//2}.npz')['H']
        for dom,ax in zip(['sphere','ball'],axes):
            lower,certificate=(sphere_probability_lower if dom=='sphere' else ball_normalized_probability_lower)(H,.9,times,.6)
            ax.plot(times,lower,color=color,ls=style,lw=2.4,label=f'd = {d}')
            records.append({'domain':dom,'d':d,'rank':d//2,'alpha':.6,'T':times.tolist(),'lower_bound':np.asarray(lower).tolist()})
    for ax,dom in zip(axes,['Sphere','Ball']):
        ax.set_title(dom);ax.set_ylim(0,1.025);ax.set_xlim(0,100);ax.grid(alpha=.17);ax.set_xlabel(r'Number of updates $T$')
        ax.axvline(40,color='#68727A',ls=':',lw=1,alpha=.65)
    axes[0].set_ylabel(r'Analytical lower bound for $q_T$');h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.52,-.05))
    fig.subplots_adjust(wspace=.2,bottom=.18);save(fig,'iteration_probability_bounds')
    (OUT/'iteration_bounds.json').write_text(json.dumps(records,indent=2)+'\n')
    # Full greedy curves retained with the run, without adding slides.
    high=json.loads((OUT/'results.json').read_text())
    for dom in ['sphere','ball']:
        fig,axes=plt.subplots(3,2,figsize=(11.5,11),constrained_layout=True)
        for d,ax in zip(DIMS,axes.flat):
            x=np.arange(d//2+1);ax.semilogy(x,np.load(OUT/f'model_d{d}.npz')['widths'][:len(x)],'k--',label=r'$d_N$')
            for key,label in [('exact_indicator','Exact indicator'),('exp_rgd40' if dom=='sphere' else 'normalized_pg40','GD'),('fixed_pool256','Pool 256'),('fixed_pool4096','Pool 4096'),('fixed_pool10000','Pool 10000'),('fresh_pool41','Fresh 41')]:
                vals=np.array([r['errors'] for r in high['runs'] if r['dimension']==d and r['domain']==dom and r['method']==key])
                ax.semilogy(x,np.median(vals,axis=0),label=label);ax.fill_between(x,vals.min(0),vals.max(0),alpha=.08)
            ax.set_title(f'd = {d}');ax.set_xlabel('Reduced dimension N');ax.set_ylabel('Worst-case energy error');ax.grid(alpha=.16)
        axes.flat[-1].set_visible(False)
        axes.flat[0].legend(fontsize=9);save(fig,f'{dom}_greedy')
    for name in ['test_error_vs_N_d32','empirical_probability_dimension']:
        for ext in ['pdf','png']:
            path=FIG/f'{name}.{ext}'
            if path.exists():shutil.copy2(path,ROOT/'paper/slides_short/figs'/path.name)
    print('All figures generated',flush=True)
if __name__=='__main__':main()
