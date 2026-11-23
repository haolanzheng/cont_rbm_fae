"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Summarize temporal diagnostics without redefining strata after observing them."""
    import os
    os.environ['OPENBLAS_NUM_THREADS']='1'
    import json
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.bgk.bgk_temporal import OUT,context,hardness
    from examples.bgk.bgk_decisive import dump,LO,SPAN
    
    par,truth,groups=context();ref=np.load(OUT/'reference.npz');hh=hardness(truth,ref)
    names=['old_pool','old_adam','new_pool','new_adam','old_adam_swapped']
    data={name:json.loads((OUT/f'{name}.json').read_text()) for name in names if (OUT/f'{name}.json').exists()}
    summary=dict(complete={k:v.get('complete',False) for k,v in data.items()},runs={},comparisons={})
    for name,x in data.items():
     E=np.array([d['errors'] for d in x['diagnostics']]);P=np.array([d['projection_errors'] for d in x['diagnostics']]);n=len(E)
     sel=x['selections'];z=(np.array([s['mu'] for s in sel])-LO)/SPAN
     rows=[]
     for i,s in enumerate(sel):
      rows.append(dict(rank=s['rank'],mu=s['mu'],boundary_distance=float(np.min(np.minimum(z[i],1-z[i]))),
        initial_percentile=float(np.mean(hh['initial'][:256]<=s['initial_error'])),
        pod3_percentile=float(np.mean(hh['pod3'][:256]<=s['pod3_error'])),
        current_error_percentile=float(np.mean(E[i,:256]<=s['pre_state_error']))))
     windows={}
     loggain=np.mean(np.log(E[:-1,:256]/E[1:,:256]),axis=1)
     for window,start,end in [('early',0,8),('late',8,14),('extended',14,len(rows))]:
      rr=rows[start:end]
      if rr:
       windows[window]={k:float(np.mean([r[k] for r in rr])) for k in ['initial_percentile','pod3_percentile','current_error_percentile','boundary_distance']}
       windows[window]['count']=len(rr);windows[window]['boundary_count']=sum(r['boundary_distance']<1e-8 for r in rr)
       jj=list(range(start,min(end,len(loggain))))
       if jj:
        total=float(sum(loggain[j] for j in jj))
        windows[window]['uniform_mean_log_error_reduction']=total
        for proxy in ['initial','pod3']:
         for q in [.8,.9]:
          hard=float(sum(loggain[j] for j in jj if rows[j][proxy+'_percentile']>=q))
          windows[window][f'{proxy}_q{q}_hard_log_gain_share']=hard/total if total!=0 else None
     strata={}
     for proxy in hh:
      for q in [.8,.9]:
       mask=hh[proxy][:256]>=np.quantile(hh[proxy][:256],q)
       for label,which in [('hard',mask),('remaining',~mask)]:
        a=E[:,:256][:,which];p=P[:,:256][:,which]
        strata[f'{proxy}_q{q}_{label}']=dict(count=int(sum(which)),max=a.max(axis=1).tolist(),mean=a.mean(axis=1).tolist(),
         projection_max=p.max(axis=1).tolist(),step_mean_log10_gain=np.mean(np.log10(a[:-1]/a[1:]),axis=1).tolist())
     summary['runs'][name]=dict(ranks=[d['rank'] for d in x['diagnostics']],selections=rows,windows=windows,strata=strata,
      metrics={g:{metric:[d[g][metric] for d in x['diagnostics']] for metric in ['max','mean','p95']} for g in groups},
      all_max=[d['all']['max'] for d in x['diagnostics']],projection_uniform_max=P[:,:256].max(axis=1).tolist(),
      solver_failures=sum(d['inner_stats']['failures'] for d in x['diagnostics']),stopped=x.get('stopped'),
      step_uniform_fraction_halved=np.mean(E[1:,:256]<.5*E[:-1,:256],axis=1).tolist(),
      step_uniform_median_factor=np.median(E[1:,:256]/E[:-1,:256],axis=1).tolist(),
      step_uniform_mean_log_error_reduction=loggain.tolist())
    for a,b in [('old_adam','old_pool'),('new_adam','new_pool'),('old_adam_swapped','old_adam')]:
     if a not in data or b not in data:continue
     aa=summary['runs'][a];bb=summary['runs'][b];n=min(len(aa['ranks']),len(bb['ranks']));out={}
     for group in ['typical','old_typical','boundary']:
      for metric in ['max','mean','p95']:
       ratio=np.array(aa['metrics'][group][metric][:n])/bb['metrics'][group][metric][:n]
       stable=[i+1 for i in range(1,n) if np.all(ratio[i:]<1)]
       out[f'{group}_{metric}']=dict(ratios=ratio.tolist(),first_observed_win=next((i+1 for i in range(1,n) if ratio[i]<1),None),
         sustained_win_to_endpoint=stable[0] if stable else None)
     ea=np.array([d['errors'] for d in data[a]['diagnostics'][:n]])[:,:256]
     eb=np.array([d['errors'] for d in data[b]['diagnostics'][:n]])[:,:256]
     out['pointwise_uniform']=dict(fraction_wins=np.mean(ea<eb,axis=1).tolist(),median_ratio=np.median(ea/eb,axis=1).tolist())
     summary['comparisons'][a+'_vs_'+b]=out
    dump(OUT/'BGK_TEMPORAL_SUMMARY.json',summary)
    
    fig,ax=plt.subplots(2,3,figsize=(15,8.6),constrained_layout=True)
    colors={'old_pool':'#777777','old_adam':'#087e8b','new_pool':'#cc7a00','new_adam':'#7353ba','old_adam_swapped':'#d1495b'}
    labels={'old_pool':'Original pool, tight replay','old_adam':'Original Adam, tight replay','new_pool':'Tight pool, rerun','new_adam':'Tight Adam, rerun','old_adam_swapped':'Adam, late block first'}
    for name,r in summary['runs'].items():
     for col,metric in enumerate(['max','mean']):
      ax[0,col].semilogy(r['ranks'],r['metrics']['typical'][metric],label=labels[name],color=colors[name],lw=1.5)
     ax[1,2].semilogy(r['ranks'],r['projection_uniform_max'],label=labels[name],color=colors[name])
    for name in ['old_adam','new_adam']:
     if name not in summary['runs']:continue
     r=summary['runs'][name];s=r['selections'];nn=[v['rank']+1 for v in s]
     ax[0,2].plot(nn,[v['initial_percentile'] for v in s],'-o',ms=3,label=labels[name],color=colors[name])
     ax[1,0].plot(nn,[v['pod3_percentile'] for v in s],'-o',ms=3,label=labels[name],color=colors[name])
    if 'old_adam' in summary['runs']:
     r=summary['runs']['old_adam']
     for label,style in [('hard','-'),('remaining','--')]:
      ss=r['strata']['initial_q0.9_'+label]
      ax[1,1].plot(np.arange(2,len(r['ranks'])+1),ss['step_mean_log10_gain'],style,label=label)
    titles=['Uniform test: maximum error','Uniform test: mean error','Selected point: initial-error percentile',
            'Selected point: independent POD3-error percentile','Original Adam: per-step mean log10 error reduction','Uniform test: best mass-constrained projection error']
    for a,title in zip(ax.flat,titles):
     a.set_title(title,fontsize=10);a.set_xlabel('Number of snapshots N');a.grid(alpha=.2);a.legend(fontsize=7)
    ax[0,0].set_ylabel('Relative state error');ax[0,1].set_ylabel('Relative state error')
    ax[0,2].set_ylim(-.03,1.03);ax[1,0].set_ylim(-.03,1.03)
    fig.savefig(OUT/'BGK_TEMPORAL_RESULTS.png',dpi=170);plt.close(fig)
    if all(k in data and len(data[k]['diagnostics'])>=15 for k in ['old_adam','old_pool']):
     fig,axes=plt.subplots(2,2,figsize=(11.5,8.5),constrained_layout=True)
     fig.suptitle('Pointwise comparison: blue = Adam better; red = pool better; black star = pool worst point',fontsize=11)
     mus=np.array(data['old_adam']['mus'][1:]);ranks=np.arange(2,16)
     sc=axes[0,0].scatter(mus[:,0],mus[:,1],c=ranks,cmap='viridis',s=75,edgecolors='black',linewidths=.4)
     for mu,n in zip(mus,ranks):axes[0,0].annotate(str(n),mu,xytext=(5,5),textcoords='offset points',fontsize=8)
     axes[0,0].set_title('Original Adam: snapshot locations (labels = N)')
     fig.colorbar(sc,ax=axes[0,0],label='Snapshot count after enrichment',shrink=.8)
     for ax,n in zip([axes[0,1],axes[1,0],axes[1,1]],[8,9,15]):
      aa=np.array(data['old_adam']['diagnostics'][n-1]['errors'][:256]);bb=np.array(data['old_pool']['diagnostics'][n-1]['errors'][:256])
      sc=ax.scatter(par[:256,0],par[:256,1],c=np.log10(aa/bb),cmap='RdBu_r',vmin=-2,vmax=2,s=23)
      ax.scatter(*par[int(np.argmax(bb))],marker='*',s=160,c='black',edgecolors='white',linewidths=.6)
      ax.set_title(f'N={n}: max ratio {max(aa)/max(bb):.3f}; Adam wins {np.mean(aa<bb):.1%}')
      fig.colorbar(sc,ax=ax,label='log10(Adam error / pool error)',shrink=.8)
     for ax in axes.flat:
      ax.set_xlim(-1.12,1.12);ax.set_ylim(.7,10.5);ax.set_xlabel('a');ax.set_ylabel('b');ax.grid(alpha=.15)
     fig.savefig(OUT/'BGK_TEMPORAL_MAP.png',dpi=170);plt.close(fig)
    print(json.dumps({'complete':summary['complete'],'windows':{k:v['windows'] for k,v in summary['runs'].items()},
     'crossovers':{k:{m:{kk:vv for kk,vv in d.items() if kk!='ratios'} for m,d in v.items()} for k,v in summary['comparisons'].items()}},indent=2))


if __name__ == "__main__":
    main()
