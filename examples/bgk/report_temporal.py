"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    import json
    from pathlib import Path
    from examples.bgk.bgk_temporal import OUT
    
    s=json.loads((OUT/'BGK_TEMPORAL_SUMMARY.json').read_text())
    q=json.loads((OUT/'BGK_TEMPORAL_QUALITY.json').read_text())
    b=json.loads((OUT/'BGK_TEMPORAL_BRANCH.json').read_text())
    assert all(s['complete'].values()) and len(s['complete'])==5
    lines=["# BGK continuous greedy: selection order and error cross-validation",
    '', "This experiment examines whether the order of Adam selections, from difficult regions early on to typical regions later, explains the transition from initially larger to eventually smaller errors than a fixed pool. No selection order is prescribed as optimal, and the fact that a larger indicator need not improve enrichment is treated as an existing distinction.",
    '', "## Experimental setup", '',
    "The nonlinear BGK screening model is retained; it is not the full Boltzmann model in the paper. Settings: Nx=100, Nv=32×32, (a,b)∈[-1,1]×[1,10], and the common initial parameter (0,5.5). The fixed pool uses the original 50 parameters. Adam uses the four corners and center as five starting points, 50 indicator evaluations per round, step size 0.08, and coordinatewise clipping.",
    '', "Two checks are performed: retain all original selections, recompute snapshots at tighter tolerances, and replay enrichment through N=15; then rerun the same algorithm with tighter full-order and inner solves through N=25, stopping if the ratio of smallest to largest snapshot singular value falls below 10⁻¹¹. The full-order tolerance is 10⁻¹¹, with an additional endpoint check at 10⁻¹³.",
    '', "The primary metric is the maximum relative state error on the same independent set of 256 uniform Sobol test parameters. Mean error and pointwise win rates assess whether gains extend across the distribution. The original 20 uniform test points and boundary/corner-neighborhood tests are retained separately. Crossovers on different test sets or metrics must not be combined into one conclusion.",
    '', "Typical refers to the uniform parameter distribution; interior location alone does not imply typicality. Two fixed difficulty measures are independent of the compared selection trajectories: error in the common initial space, and projection error onto the mean plus 3 POD modes constructed from another 64 independent uniform parameters. Percentiles use the fixed 256-point reference set. These are operational measures, not unique intrinsic difficulties. Regions are fixed by the top 10% or 20% of initial-space or POD error, and their errors and per-step improvements are recorded.",
    '', "The first 8 and last 6 selection windows were chosen after inspecting the original trajectory and are retrospective diagnostics. The full selection sequence is reported; window averages do not replace an assessment of changes across stages.",
    '', "## Error curves on the same test set", '']
    for label,prefix in [("Original selections, replayed with tighter solves",'old'),("Tighter solves with new selections",'new')]:
     a=s['runs'][prefix+'_adam'];p=s['runs'][prefix+'_pool'];c=s['comparisons'][prefix+'_adam_vs_'+prefix+'_pool']
     lines+=['### '+label,'',"| N | Fixed-pool maximum error | Adam maximum error | Adam/pool maximum | Adam/pool mean | Adam pointwise win rate |",'|---:|---:|---:|---:|---:|---:|']
     for n in [2,3,5,6,8,9,10,12,15,20,25]:
      if n>min(len(a['ranks']),len(p['ranks'])):continue
      aa=a['metrics']['typical'];pp=p['metrics']['typical'];i=n-1
      lines.append(f"| {n} | {pp['max'][i]:.6e} | {aa['max'][i]:.6e} | {aa['max'][i]/pp['max'][i]:.4f} | {aa['mean'][i]/pp['mean'][i]:.4f} | {100*c['pointwise_uniform']['fraction_wins'][i]:.1f}% |")
     lines+=['',f"Within the computed range, sustained advantage begins at N={c['typical_max']['sustained_win_to_endpoint']} for the uniform-test maximum, N={c['typical_mean']['sustained_win_to_endpoint']} for the mean, and N={c['old_typical_max']['sustained_win_to_endpoint']} for the original 20-point maximum. None means no such crossover occurs before the endpoint. Sustained refers only to the computed interval.",'']
    lines+=["## Selection difficulty and order",'',"A higher difficulty percentile indicates greater error under the fixed reference. The current-error percentile changes with the reduced space.",'',
    "| Trajectory | Selection window | Mean initial-difficulty percentile | Mean POD3-difficulty percentile | Mean current-error percentile | Boundary selections |",'|---|---|---:|---:|---:|---:|']
    for name in ['old_adam','new_adam']:
     for window,w in s['runs'][name]['windows'].items():
      lines.append(f"| {name} | {window} | {100*w['initial_percentile']:.1f}% | {100*w['pod3_percentile']:.1f}% | {100*w['current_error_percentile']:.1f}% | {w['boundary_count']}/{w['count']} |")
    lines+=['',"Complete original Adam selection record:",'',"| N after enrichment | Parameter | Initial-difficulty percentile | POD3-difficulty percentile |",'|---:|---|---:|---:|']
    for r in s['runs']['old_adam']['selections']:
     lines.append(f"| {r['rank']+1} | ({r['mu'][0]:.6f}, {r['mu'][1]:.6f}) | {100*r['initial_percentile']:.1f}% | {100*r['pod3_percentile']:.1f}% |")
    lines+=['',"## Changes in regional gains",'',
    "The table reports mean log10(e_N/e_{N+1}) per step; positive values indicate error reduction. Hard denotes the top 10% of uniform test parameters under the fixed difficulty measure; remaining denotes the other 90%. Early covers selections 1–8, late covers 9–14, and extended covers 15 onward. The table compares where gains occur, without equating geometric interior with typicality.",
    '',"| Trajectory | Difficulty reference | Region | Early per step | Late per step | Extended per step |",'|---|---|---|---:|---:|---:|']
    for name in ['old_adam','new_adam']:
     for proxy in ['initial','pod3']:
      for label in ['hard','remaining']:
       v=s['runs'][name]['strata'][f'{proxy}_q0.9_{label}']['step_mean_log10_gain']
       avg=lambda a: f'{sum(a)/len(a):.4f}' if a else '—'
       lines.append(f"| {name} | {proxy} | {label} | {avg(v[:8])} | {avg(v[8:14])} | {avg(v[14:])} |")
    lines+=['',"A worse mean error or a win rate below one half does not by itself refute the proposed explanation: later selections from ordinary regions may still reduce the maximum below the fixed-pool error tail. The analysis therefore combines selection difficulty, regional gains, and conditional interventions rather than replacing the maximum-error objective with the mean."]
    lines+=['',"## Order interventions with a common prefix",'',"Keep the first 8 original Adam snapshots and replace only the 9th. All three branches have the same dimension, full-order accuracy, and test set.",'',
    "| 9th snapshot | Parameter | Uniform-test maximum error | Uniform-test mean error |",'|---|---|---:|---:|']
    labels={'8':"Actual next selection",'9':"First later interior selection added early",'12':"Later selection with low fixed difficulty added early (exploratory)"}
    for k,r in b['branches'].items():
     lines.append(f"| {labels[k]} | ({r['added_mu'][0]:.6f}, {r['added_mu'][1]:.6f}) | {r['typical']['max']:.6e} | {r['typical']['mean']:.6e} |")
    lines+=['',"The first two branches test whether the first interior selection explains the crossover. The third was added after those results and uses two independent difficulty labels; it is exploratory. These branches measure one-step effects in a particular space, not a universal optimal selection rule.",
    '', "A separate intervention moves the last 6 original Adam snapshots before the first 8, keeping the initial snapshot and final snapshot set unchanged. This tests effects on intermediate spaces. Equality of the final spaces follows from their construction and is not evidence that one order is ultimately superior.",
    '',"| N | Later block first/original: maximum error | Later block first/original: mean error |",'|---:|---:|---:|']
    c=s['comparisons']['old_adam_swapped_vs_old_adam']
    for n in [2,3,5,7,9,12,15]:
     lines.append(f"| {n} | {c['typical_max']['ratios'][n-1]:.5f} | {c['typical_mean']['ratios'][n-1]:.5f} |")
    lines+=['',"## Interpretation",'', (OUT/'interpretation.md').read_text(),'', "## Numerical checks and limitations",'',
    f"Reported failures across all per-round test solves: {sum(r['solver_failures'] for r in q['runs'].values())}. The maximum-error ratio for the two orders of the same final snapshot set is {q['order_invariance']['maximum_error_ratio']:.10f}. The best mass-constrained projection error was checked against the deployed-model error. Tighter full-order and inner solves independently checked the worst endpoint parameters.",'',
    "Endpoint error checks:",'',"| Method | N | Original error | Error with tighter full-order and inner solves | Full-order solution change |",'|---|---:|---:|---:|---:|']
    for r in q['truth_floor']:
     lines.append(f"| {r['name']} | {r['rank']} | {r['old_error']:.5e} | {r['strict_solver_and_truth_error']:.5e} | {r['truth_difference']:.3e} |")
    endpoint=OUT/'BGK_TEMPORAL_ENDPOINT_PRECISION.json'
    if endpoint.exists():
     rows=json.loads(endpoint.read_text())
     lines+=['',"For N=25, both spaces were also rebuilt after tightening every snapshot solve from 10⁻¹¹ to 10⁻¹³, and all 256 uniform test parameters were reevaluated. The full-order and inner solves at the new worst parameter were tightened separately.",
     '',"| Method | Original uniform-test maximum | Maximum after refining all snapshots | Maximum-error ratio | Fully refined error at the worst parameter |",'|---|---:|---:|---:|---:|']
     for r in rows:
      lines.append(f"| {r['name']} | {r['original_max']:.6e} | {r['refined_snapshots_max']:.6e} | {r['max_error_ratio']:.8f} | {r['fully_refined_worst_point_error']:.6e} |")
    lines+=['',"Conclusions are restricted to this BGK example and the specified Adam settings. Difficulty measures, finite test sets, and retrospective windows limit causal generalization. Reordering existing snapshots is a conditional intervention, not a new self-consistent greedy trajectory obtained by reoptimization.",
    '',"## Files and reproduction",'',"- BGK_TEMPORAL_RESULTS.png: common-test errors, difficulty ranks, regional per-step gains, and best projection errors.",
    "- BGK_TEMPORAL_SUMMARY.json: aggregate results and per-round errors in each fixed region.",
    "- old_*.json / new_*.json: raw errors, projection errors, residuals, and selections for 420 test points per round.",
    "- *.npz: refined snapshots and independent difficulty references.",
    "- BGK_TEMPORAL_BRANCH.json / BGK_TEMPORAL_QUALITY.json: conditional interventions and numerical checks.",
    '',"Reproduction commands are listed in README_TEMPORAL.md. The large 384-point full-order solution cache is not bundled and can be regenerated deterministically from the original settings."]
    (OUT/'BGK_TEMPORAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report generated')


if __name__ == "__main__":
    main()
