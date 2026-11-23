"""Summarize saved cube optimizer comparisons without changing any slides."""
from .compare_optimizers import *
import csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/cube_optimizer_report_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'gd':'GD, T=80','sqp':'SQP','reflective':'Reflective trust region','trust':'Interior-point trust region'}
STYLE={'gd':('#24557A','-','o'),'sqp':('#C17E3C','--','^'),'reflective':('#438273','-.','s'),'trust':('#993F4D',':','D')}
DIMS=[4,8,16,32,64,128]


def main():
    rows=json.loads((OUTPUT/'results.json').read_text());old=json.loads((SOURCE/'results.json').read_text())
    complete={(r['d'],r['seed'],r['method'],r['initialization']) for r in rows}
    assert len(complete)==len(rows)
    def select(d,m,i):return [r for r in rows if (r['d'],r['method'],r['initialization'])==(d,m,i)]
    summary=[]
    for d in DIMS:
        for init in INITS:
            for method in METHODS:
                rr=select(d,method,init)
                if not rr:continue
                assert len(rr)==8,(d,method,init,len(rr))
                stats=[s for r in rr for s in r['rounds'][:d//2]]
                summary.append(dict(d=d,N=d//2,method=method,initialization=init,
                    median_error=float(np.median([r['test_max_error'] for r in rr])),
                    error_q25=float(np.quantile([r['test_max_error'] for r in rr],.25)),error_q75=float(np.quantile([r['test_max_error'] for r in rr],.75)),
                    median_ratio=float(np.median([r['ratio'] for r in rr])),median_wall_ms=float(np.median([r['wall_ms'] for r in rr])),
                    median_iterations_per_round=float(np.median([s['nit'] for s in stats])),
                    median_evaluations_per_round=float(np.median([s['nfev']+s['initialization_evaluations'] for s in stats])),
                    solver_failures=sum(s['solver_success'] is False for s in stats),rounds=len(stats)))
    with (OUTPUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    matches=[]
    for r in rows:
        if r['method']=='gd' and r['initialization']=='random':
            original=next(x for x in old if (x['domain'],x['d'],x['seed'],x['method'])==('cube',r['d'],r['seed'],'gd'))
            matches.append(abs(r['test_max_error']/original['test_max_error']-1))
            assert np.isclose(r['test_max_error'],original['test_max_error'],rtol=2e-7,atol=2e-14)
    audit=dict(runs=len(rows),gd_matches_old_max_relative_difference=max(matches),
        max_feasibility_defect=max(s['feasibility_defect'] for r in rows for s in r['rounds']),
        max_orthogonality_defect=max(r['orthogonality_defect'] for r in rows),stopped_runs=[(r['d'],r['method'],r['seed'],r['initialization']) for r in rows if r['stop']],
        solver_failures=[dict(d=r['d'],method=r['method'],init=r['initialization'],seed=r['seed'],n=s['n'],status=s['status']) for r in rows for s in r['rounds'] if s['solver_success'] is False])
    dump(OUTPUT/'audit.json',audit)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    primary=['gd','sqp','reflective']
    for key,ylabel,name in [('median_error','Test maximum energy error','final_error_vs_dimension'),('median_wall_ms','Greedy training time (ms)','wall_time_vs_dimension')]:
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
        for init,ax in zip(INITS,axes):
            for m in primary:
                rr=[next(x for x in summary if (x['d'],x['method'],x['initialization'])==(d,m,init)) for d in DIMS]
                c,ls,mk=STYLE[m];ax.plot(DIMS,[x[key] for x in rr],color=c,ls=ls,marker=mk,label=LABELS[m],mfc='white',lw=2)
            ax.set(xlabel='Parameter dimension d',ylabel=ylabel,xscale='log',yscale='log',title='Random start' if init=='random' else 'Best of 256 pool points',xticks=DIMS)
            ax.set_xticklabels(DIMS);ax.grid(alpha=.15)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(OUTPUT/f'{name}.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
    for init,ax in zip(INITS,axes):
        base=np.array([next(x['median_error'] for x in summary if (x['d'],x['method'],x['initialization'])==(d,'gd',init)) for d in DIMS])
        for m in primary:
            values=np.array([next(x['median_error'] for x in summary if (x['d'],x['method'],x['initialization'])==(d,m,init)) for d in DIMS])/base
            c,ls,mk=STYLE[m];ax.plot(DIMS,values,color=c,ls=ls,marker=mk,label=LABELS[m],mfc='white',lw=2)
        ax.set(xlabel='Parameter dimension d',ylabel='Median test error / GD median test error',xscale='log',title='Random start' if init=='random' else 'Best of 256 pool points',xticks=DIMS)
        ax.set_xticklabels(DIMS);ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(OUTPUT/'relative_error_to_gd.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10.5,9.2))
    for d,axs in zip([16,32,64],axes):
        for init,ax in zip(INITS,axs):
            for m in primary:
                rr=select(d,m,init);v=np.median([r['errors'] for r in rr],axis=0);c,ls,mk=STYLE[m]
                ax.semilogy(range(len(v)),v,label=LABELS[m],color=c,ls=ls,marker=mk,markevery=max(1,d//8),ms=4,mfc='white',lw=1.7)
            ax.axvline(d//2,color='.65',ls=':',lw=1);ax.set(xlabel='Reduced dimension N',ylabel='Test maximum energy error',title=f'd={d}, '+('random start' if init=='random' else 'best of 256'));ax.grid(alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUTPUT/'test_error_vs_N.png',dpi=190);plt.close(fig)
    equal_budget=json.loads((OUTPUT/'config.json').read_text()).get('equal_step_budget')
    lines=['Cube optimizer comparison','N=d/2; eight-seed medians; old test sets, alpha=.6 and GD T=80.',
           'SQP = SciPy SLSQP. Reflective TR = Fides 0.7.8 Coleman-Li / two-dimensional subspace / exact Hessian.',
           'Original paper objectives are replaced by the common normalized baseline J.',
           'Initialization: one uniform point, or best of 256 fixed uniform points reused across rounds.',
           'Time includes initialization, optimization and enrichment; no HF setup or test evaluation.',
           (f'All optimizers have the same {equal_budget}-iteration cap; early stops and actual counts are saved.' if equal_budget else 'Iteration caps differ: GD 80, other solvers up to 500 with stopping criteria. Counts and stopping status are saved.'),
           'No equality of per-iteration cost or global optimality is assumed.','']
    for init in INITS:
        lines += [init,'d | GD error | SQP error | Reflective TR error | GD ms | SQP ms | Reflective TR ms']
        for d in DIMS:
            rr=[next(x for x in summary if (x['d'],x['method'],x['initialization'])==(d,m,init)) for m in primary]
            lines.append(f'{d} | '+' | '.join(f"{r['median_error']:.6g}" for r in rr)+' | '+' | '.join(f"{r['median_wall_ms']:.2f}" for r in rr))
        lines+=['']
    frozen_path=OUTPUT/'frozen.json'
    if frozen_path.exists():
        fr=json.loads(frozen_path.read_text());lines+=['Fixed-space searches: identical H and starts, median J(method)/J(GD).']
        for init in INITS:
            for d in DIMS:
                ratios={m:[r['value']/next(x['value'] for x in fr if (x['d'],x['seed'],x['initialization'],x['method'])==(d,r['seed'],init,'gd')) for r in fr if (r['d'],r['initialization'],r['method'])==(d,init,m)] for m in ['sqp','reflective']}
                lines.append(f"{init} d={d}: SQP {np.median(ratios['sqp']):.5f}, reflective {np.median(ratios['reflective']):.5f}")
    lines+=['','Termination diagnostics at N=d/2:']
    for method in METHODS:
        rr=[r for r in rows if r['method']==method];ss=[x for r in rr for x in r['rounds'][:r['d']//2]]
        failures=sum(x['solver_success'] is False for x in ss)
        lines.append(f'{method}: {failures}/{len(ss)} unsuccessful solver terminations; '+
                     f'projected-gradient infinity norm median={np.median([x["projected_gradient_inf"] for x in ss]):.4g}, max={max(x["projected_gradient_inf"] for x in ss):.4g}.')
    lines+=['Solver success flags do not certify global optimality; failed/capped runs are retained, not discarded.',
            'The projected-gradient stationarity check is saved independently of each solver stopping test.',
            'Equal iteration counts do not imply equal per-iteration cost; time is reported separately.']
    (OUTPUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('AUDIT',audit['runs'],audit['gd_matches_old_max_relative_difference'],'failures',len(audit['solver_failures']))
if __name__=='__main__':main()
