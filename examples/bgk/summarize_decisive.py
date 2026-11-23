"""Summarize predeclared BGK controls without selecting favorable seeds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.bgk.bgk_decisive import OUT,SEEDS,METHODS,CHECKPOINTS,dump

LABELS={'fixed':'Fixed uniform 50','fixed_corners':'Fixed 46 + corners','fresh':'Fresh uniform 50',
 'fresh_corners':'Fresh 46 + corners','pilot_only':'Shared pilot 14','hybrid':'Pilot 14 + Adam 36','original_adam':'Original 5-start Adam'}
ZH={'fixed':"Fixed uniform pool, 50 points",'fixed_corners':"Fixed pool, 46 points + corners",'fresh':"Fresh uniform pool, 50 points per round",
 'fresh_corners':"Fresh pool, 46 points + corners per round",'pilot_only':"Common 14-point pilot",'hybrid':"14-point pilot + 36 Adam updates",'original_adam':"Original five-start Adam"}
COLORS=dict(zip(METHODS,['#bd8621','#d64f35','#4a93c9','#237b4b','#8f8f8f','#9b42b4']))

def summarize():
 runs={m:[json.loads((OUT/f'{m}_{s}.json').read_text()) for s in SEEDS] for m in METHODS}
 assert all(x['complete'] for rows in runs.values() for x in rows)
 original=json.loads((OUT/'original_adam_901.json').read_text());assert original['complete']
 precision=json.loads((OUT/'precision.json').read_text())
 frozen=[json.loads((OUT/f'frozen_{p}_{s}.json').read_text()) for p in ['pool','adam'] for s in SEEDS]
 summary={'methods':{},'paired':{},'frozen':{},'precision':{}}
 def arr(m,group,metric,rank=15):
  return np.array([next(d for d in r['diagnostics'] if d['rank']==rank)[group][metric] for r in runs[m]])
 for m in METHODS:
  stats={}
  for rank in CHECKPOINTS:
   stats[str(rank)]={}
   for group in ['all','typical','boundary','corner_neighborhood']:
    stats[str(rank)][group]={}
    for metric in ['max','mean','p95']:
     a=arr(m,group,metric,rank);stats[str(rank)][group][metric]=dict(median=float(np.median(a)),min=float(min(a)),max=float(max(a)),values=a.tolist())
  stats['training_cpu_median']=float(np.median([r['training_cpu_seconds'] for r in runs[m]]))
  stats['training_wall_median']=float(np.median([r['training_wall_seconds'] for r in runs[m]]))
  stats['unique_training_solves_median']=float(np.median([sum(x['unique_solves'] for x in r['selections']) for r in runs[m]]))
  stats['inner_failures']=sum(x['inner_stats']['failures'] for r in runs[m] for x in r['selections'])+sum(r['final_inner_stats']['failures'] for r in runs[m])
  summary['methods'][m]=stats
 for a,b in [('fixed_corners','fixed'),('fresh','fixed'),('fresh_corners','fixed_corners'),
             ('hybrid','fresh_corners'),('hybrid','pilot_only'),('hybrid','fixed_corners')]:
  key=a+'/'+b;ratio=arr(a,'all','max')/arr(b,'all','max');rng=np.random.default_rng(30019)
  boot=np.median(ratio[rng.integers(0,len(ratio),(10000,len(ratio)) )],axis=1)
  summary['paired'][key]=dict(ratios=ratio.tolist(),median=float(np.median(ratio)),wins=int(sum(ratio<1)),
    n=len(ratio),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),note='Exploratory paired bootstrap with only six seeds')
 for path in ['pool','adam']:
  rows=[f for f in frozen if f['space']==path]
  summary['frozen'][path]=dict(
    indicator_ratio=[f['hybrid']['value']/f['pool']['value'] for f in rows],
    post_error_ratio=[f['enrichment']['hybrid']['all']['max']/f['enrichment']['fresh_corners']['all']['max'] for f in rows],
    cpu_matched_indicator_ratio=[f['hybrid']['value']/f['cpu_matched']['value'] for f in rows],
    matched_random_counts=[f['cpu_matched']['evaluations'] for f in rows],
    pilot_gain=[f['hybrid']['gain'] for f in rows])
 summary['precision']=dict(max_kernel_residual_relative=max(x['r_relative'] for x in precision['kernel']),
    max_kernel_jacobian_relative=max(x['jac_relative'] for x in precision['kernel']),
    max_solver_state_difference=max(x['state_difference'] for x in precision['solver_comparison']),
    max_gradient_relative=max(x['relative'] for x in precision['gradients']))
 summary['original_adam']=original['diagnostics'][-1]
 reference_ratio=original['diagnostics'][-1]['all']['max']/arr('fresh_corners','all','max')
 summary['original_adam_vs_fresh_corners']=dict(ratios=reference_ratio.tolist(),
   median=float(np.median(reference_ratio)),wins=int(sum(reference_ratio<1)),n=6,
   note='One deterministic Adam trajectory compared with six randomized baseline runs; not six independent Adam trajectories.')
 replay=json.loads((OUT/'snapshot_precision_replay.json').read_text())
 summary['snapshot_precision_replay']={m:{
   'original_snapshot_max':r['original']['all']['max'],
   'tight_snapshot_max':r['tight']['all']['max'],
   'max_truth_change_1e11_to_1e13':max(x['relative_truth_change'] for x in r['truth_1e13_checks'])
  } for m,r in replay.items()}
 dump(OUT/'BGK_DECISIVE_SUMMARY.json',summary)

 fig,ax=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
 for m in METHODS:
  vals=np.array([arr(m,'all','max',n) for n in CHECKPOINTS])
  ax[0,0].plot(CHECKPOINTS,np.median(vals,axis=1),'o-',color=COLORS[m],label=LABELS[m])
  ax[0,0].fill_between(CHECKPOINTS,np.min(vals,axis=1),np.max(vals,axis=1),color=COLORS[m],alpha=.06)
 ax[0,0].plot(CHECKPOINTS,[x['all']['max'] for x in original['diagnostics']],'k*:',label='Original Adam (one run)')
 ax[0,0].set(yscale='log',xlabel='Basis size N',ylabel='Audit maximum relative error',title='Full trajectories: median and seed range')
 ax[0,0].legend(fontsize=8);ax[0,0].grid(alpha=.2)
 for i,m in enumerate(METHODS):
  vals=arr(m,'all','max');ax[0,1].scatter(i+np.linspace(-.15,.15,len(vals)),vals,color=COLORS[m],s=30)
  ax[0,1].plot([i-.25,i+.25],[np.median(vals)]*2,color='black',linewidth=2)
 ax[0,1].scatter(6,original['diagnostics'][-1]['all']['max'],marker='*',s=100,color='black')
 ax[0,1].set(yscale='log',xticks=np.arange(7),xticklabels=['Fixed','Fixed+C','Fresh','Fresh+C','Pilot','Hybrid','Orig. A'],ylabel='N=15 audit maximum',title='Every seed shown; black bars are medians')
 ax[0,1].grid(axis='y',alpha=.2)
 for m in METHODS:
  ax[1,0].scatter(arr(m,'typical','max'),arr(m,'boundary','max'),color=COLORS[m],label=LABELS[m],s=30)
 ax[1,0].scatter(original['diagnostics'][-1]['typical']['max'],original['diagnostics'][-1]['boundary']['max'],marker='*',s=100,color='black')
 ax[1,0].set(xscale='log',yscale='log',xlabel='N=15 typical maximum',ylabel='N=15 boundary maximum',title='Boundary and typical errors')
 ax[1,0].grid(alpha=.2)
 for path,marker,color in [('pool','o','#d18b20'),('adam','s','#8e44ad')]:
  f=summary['frozen'][path]
  ax[1,1].scatter(f['indicator_ratio'],f['post_error_ratio'],marker=marker,color=color,label=f'Frozen {path} history',s=45)
 ax[1,1].axvline(1,color='gray',ls=':');ax[1,1].axhline(1,color='gray',ls=':')
 ax[1,1].set(xscale='log',yscale='log',xlabel='Hybrid / fresh+C selected squared residual',ylabel='Hybrid / fresh+C post-enrichment error',title='Same-space interventions, equal 50-call budget')
 ax[1,1].legend(fontsize=8);ax[1,1].grid(alpha=.2)
 fig.savefig(OUT/'BGK_DECISIVE_RESULTS.png',dpi=170);plt.close(fig)

 lines=["# BGK mechanism analysis: paired comparisons",'',
 "This report uses six prespecified paired seeds, complete greedy trajectories for N=1…15, and one-step interventions in two frozen spaces. The model is a BGK screening surrogate, not the full Boltzmann collision operator.",
 '', "## Main findings",'',
 "**The large gap from the fixed uniform pool cannot be attributed entirely to continuous optimization.** Within this parameter box and N≤15, boundary coverage and refreshed candidates already yield comparable or smaller final errors. Continuous updates improve search near non-corner peaks, but larger indicator values do not consistently improve enrichment.",
 '',
 "1. The median combined maximum error is 7.855e-6 for the fixed uniform pool, 1.403e-6 after adding corners, 1.840e-6 for a fresh uniform pool, and 1.146e-6 for a fresh pool with corners. Gains vary by seed; adding corners is not uniformly beneficial.",
 "2. The 14-point pilot plus 36 Adam updates gives a median final error of 1.459e-6. Against the fresh pool with corners using the same pilot and 50-evaluation budget, it wins in 1/6 seeds. The median paired error ratio is 1.331, about 33% worse. This is the median of paired ratios, not the ratio of two medians.",
 "3. Rerunning the original five-start Adam with corners plus center and 50 evaluations gives 3.458e-6, worse than all six fresh-pool-with-corners runs. This is one deterministic reference trajectory, not six independent Adam repetitions.",
 "4. In the frozen pool space, all six comparisons select the same shared corner. In the frozen Adam space, continuous refinement gives larger Phi throughout, with median indicator ratio 1.596 and CPU-matched ratio 1.368. Yet post-enrichment error is worse in 4/6 comparisons, with median paired error ratio 1.087.",
 "5. Using only the 14 common pilot points gives median final error 1.249e-6 and median training CPU time about 31.9 seconds, versus 87.7 seconds for the hybrid. This lower-budget control indicates the value of boundary information and fresh candidates in this example.",
 '',
 "Early coverage of difficult regions followed by later error reduction remains a possible description, but the experiment does not establish a mechanism unique to continuous updates or general superiority of this Adam scheme over sampling informed by the domain structure. Pointwise difficulty and marginal contribution to the test solution set require separate assessment. Some refined points also have larger full-order projection novelty but worse enrichment gains; a scalar novelty measure alone provides no guarantee.",
 '',
 "Further analysis should relate indicator values, error distributions, and enrichment gains, retaining fresh sampling with boundary coverage as a baseline. N>15, high-dimensional parameter spaces, and the full Boltzmann operator are not tested. Six seeds support a mechanism study, not universal or multiplicity-adjusted statistical conclusions.",
 '', "## Experimental settings and scope",'',
 "- Full grid Nx=100, Nv=32; parameter domain [-1,1]×[1,10]; initial snapshot (0,5.5).",
 "- The selection indicator Phi is the squared weighted residual after the mass-constrained ROM inner minimization. Reported indicator gains are ratios of Phi, not residual norms.",
 "- New snapshots and test references use FOM residual tolerance 1e-11. Inner solves use the full analytic Jacobian and backtracking Gauss–Newton, with scaled SciPy TRF as a fallback for ill-conditioning or stagnation.",
 "- The common independent test set contains 384 points: 256 scrambled Sobol interior points, 64 boundary points, and 64 near-corner points. Test values do not enter search, tuning, or timing allowances.",
 "- Evaluation occurs at N=5,10,15. The prespecified primary metric is the combined 384-point maximum relative state error at N=15. This is a finite-test maximum, not a certified supremum over the continuous domain.",
 "- Six paired seeds, 901…906, use a nominal budget of 50 evaluations per round. Fixed pools retain evaluations at previously selected points but forbid reselection. The 14-point pilot-only control uses only 14 evaluations and is explicitly an unequal-budget ablation.",
 "- Hybrid and fresh+corners share the four corners and first 10 random points. The remaining 36 evaluations fund either 9 updates on each of 4 Adam trajectories or 36 further independent points. Adam uses learning rate 0.08, without tuning to these test results.",
 "- Training process CPU time and wall time use one BLAS thread. Full trajectories are compared at matched evaluation budgets, not equal wall time. Frozen-space controls additionally truncate random sampling to a strict CPU budget.",
 "- Six random methods with six seeds yield 36 trajectories. One separate original corners-plus-center Adam trajectory is deterministic and is not treated as six independent repetitions.",
 '', "## Numerical accuracy checks",'',
 f"- Maximum relative residual difference between accelerated and original kernels: {summary['precision']['max_kernel_residual_relative']:.3e}; relative Jacobian difference: {summary['precision']['max_kernel_jacobian_relative']:.3e}.",
 f"- Maximum relative state difference between GN and independent strict TRF: {summary['precision']['max_solver_state_difference']:.3e}; maximum relative difference between the envelope gradient and reoptimized finite differences: {summary['precision']['max_gradient_relative']:.3e}.",
 "- Keeping the original N=15 spaces fixed and tightening only the reference solutions changes the typical maximum errors from about 1.805e-6/1.812e-6 to pool 9.081e-7 and Adam 1.205e-6. The earlier 0.4% difference is not robust. The boundary advantage remains about 47-fold.",
 '', "## Complete N=15 trajectories: medians over six seeds",'',
 "| Method | Combined maximum error | Typical maximum | Boundary maximum | Typical mean | Training CPU seconds |",
 '|---|---:|---:|---:|---:|---:|']
 for m in METHODS:
  ss=summary['methods'][m];v=ss['15'];lines.append(f"| {ZH[m]} | {v['all']['max']['median']:.4e} | {v['typical']['max']['median']:.4e} | {v['boundary']['max']['median']:.4e} | {v['typical']['mean']['median']:.4e} | {ss['training_cpu_median']:.1f} |")
 v=original['diagnostics'][-1]
 lines.append(f"| Original five-start Adam (one deterministic reference) | {v['all']['max']:.4e} | {v['typical']['max']:.4e} | {v['boundary']['max']:.4e} | {v['typical']['mean']:.4e} | {original['training_cpu_seconds']:.1f} |")
 lines+=['',"Training times exclude test diagnostics, generation of full-order test solutions, and checkpoint writes. All methods use the same inner solver; sampling methods also use analytic inner derivatives but need no outer parameter updates. Actual unique solve counts may differ. Raw records retain nominal evaluations, unique solves, and CPU time.",
 '', "## Paired mechanism comparisons",'', "| Comparison (numerator/denominator) | Median combined maximum-error ratio | Numerator wins | Exploratory paired-bootstrap 95% interval |",'|---|---:|---:|---:|']
 for key,v in summary['paired'].items():
  a,b=key.split('/');ci=v['bootstrap95'];lines.append(f"| {ZH[a]} / {ZH[b]} | {v['median']:.4f} | {v['wins']}/6 | [{ci[0]:.4f}, {ci[1]:.4f}] |")
 lines+=['',"Ratios below one favor the numerator. Six-seed bootstrap intervals describe stability in this sample; they do not establish universal or multiplicity-adjusted conclusions.",
 '', "## Frozen spaces: from search gains to enrichment gains",'',
 "High-accuracy full-order solutions at the first seven selected parameters of the old pool and Adam trajectories define two frozen spaces. Within each space, every method shares the pilot and ROM. The full-order solution selected by fresh+corners or hybrid is added separately, followed by reevaluation on the same 384-point test set.",
 '', "| Frozen space | Median indicator ratio, hybrid/fresh+C | Median post-enrichment error ratio | Median CPU-matched indicator ratio | CPU-matched random evaluation count |",'|---|---:|---:|---:|---:|']
 for path,f in summary['frozen'].items():
  counts=f['matched_random_counts'];lines.append(f"| {path}, historical N=7 | {np.median(f['indicator_ratio']):.4f} | {np.median(f['post_error_ratio']):.4f} | {np.median(f['cpu_matched_indicator_ratio']):.4f} | {min(counts)}–{max(counts)} |")
 lines+=['',"Per-run JSON records contain all seed results, selected points, inner stopping states, indicator gains, and snapshot projection novelty.",
 '', "## Snapshot-accuracy intervention with fixed selections",'',
 "| Original selection trajectory | 384-point maximum with original snapshots | Maximum with recomputed high-accuracy snapshots | Maximum reference change from 1e-11 to 1e-13 |",
 '|---|---:|---:|---:|']
 for method,r in summary['snapshot_precision_replay'].items():
  lines.append(f"| {method} | {r['original_snapshot_max']:.4e} | {r['tight_snapshot_max']:.4e} | {r['max_truth_change_1e11_to_1e13']:.3e} |")
 lines+=['',"This intervention fixes every original selection, unlike rerunning the greedy algorithm with tighter solves. Reference accuracy is checked at the five worst test points of each recomputed trajectory.",
 '', "## Limits of the interpretation",'',
 "- Adding corners intervenes on boundary coverage; refreshing the pool intervenes on fixed-set coverage; random sampling versus Adam refinement after a common pilot isolates continuous updates.",
 "- Finite-sample fractions of large residuals and q90/max ratios are auxiliary diagnostics. Changes in q90/max are not interpreted as changes in set volume.",
 "- Indicator gains require a separate check of post-enrichment error; the frozen-space experiments test this connection directly.",
 "- Active-boundary projection and inexact inner solves do not directly satisfy existing interior-region Adam theorems with exact gradients.",
 '', "## Reproduction",'',
 "Compile with `g++ -O3 -std=c++17 -shared -fPIC bgk_exact_kernel.cpp -o bgk_exact_kernel.so`.",
 "Run `python bgk_decisive.py init --workers 6`, `python bgk_decisive.py precision`, `python bgk_decisive.py run --workers 6`, `python bgk_frozen_controls.py`, and `python summarize_decisive.py` in order.",
 "Outer selections and test data are retained in the run outputs. Parameters, random seeds, and tolerances for rebuilding reference solutions are recorded in protocol.json. Completed runs are skipped; partial runs require explicit recovery rather than silently resetting their trajectories."]
 (OUT/'BGK_DECISIVE_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['methods','original_adam']},indent=2))

if __name__=='__main__':summarize()
