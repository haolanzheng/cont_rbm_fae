"""Paired frozen-space checks and reports for the shared 80-step experiment."""
from .compare_equal_steps import OUT, PRIMARY, STEPS
from . import compare_optimizers as base
from . import optimizer_frozen_study as frozen
from . import report_optimizers as report
import csv,json
from functools import partial
import numpy as np


def main():
    frozen.OUTPUT=OUT
    frozen.optimize=partial(base.optimize,max_steps=STEPS)
    frozen.main()
    report.OUTPUT=OUT;report.METHODS=PRIMARY
    report.LABELS.update(gd='GD, 80 steps',sqp='SQP, at most 80 steps',reflective='Reflective TR, at most 80 steps')
    report.main()
    rows=json.loads((OUT/'results.json').read_text())
    assert len(rows)==288
    stats=[s for r in rows for s in r['rounds']]
    assert all(s['nit']<=STEPS for s in stats)
    extras=[]
    for d in report.DIMS:
        for init in base.INITS:
            for method in PRIMARY:
                rr=[r for r in rows if (r['d'],r['initialization'],r['method'])==(d,init,method)]
                ss=[s for r in rr for s in r['rounds'][:d//2]]
                exhausted=sum(s['status']=='fixed_budget' or 'Iteration limit reached' in s['status'] or 'MAXITER' in s['status'] for s in ss)
                early=sum(s['nit']<STEPS for s in ss)
                errors=sum(s['solver_success'] is False and 'Iteration limit reached' not in s['status'] and 'MAXITER' not in s['status'] for s in ss)
                extras.append(dict(d=d,initialization=init,method=method,cap=STEPS,rounds=len(ss),median_actual_iterations=float(np.median([s['nit'] for s in ss])),mean_actual_iterations=float(np.mean([s['nit'] for s in ss])),early_stop_rounds=early,budget_exhausted_rounds=exhausted,other_solver_failures=errors))
    with (OUT/'iteration_counts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(extras[0]));writer.writeheader();writer.writerows(extras)
    with (OUT/'summary.csv').open() as f:
        table=list(csv.DictReader(f))
    for row in table:
        row['nonconverged_at_stop_rounds']=row.pop('solver_failures')
        item=next(x for x in extras if (x['d'],x['method'],x['initialization'])==(int(row['d']),row['method'],row['initialization']))
        row.update({k:item[k] for k in ['cap','mean_actual_iterations','early_stop_rounds','budget_exhausted_rounds','other_solver_failures']})
    with (OUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text=(OUT/'summary.txt').read_text()
    text=text[:text.index('Termination diagnostics at N=d/2:')]
    text+='Actual iteration counts at N=d/2 (early stopping is retained):\n'
    for method in PRIMARY:
        ss=[s for r in rows if r['method']==method for s in r['rounds'][:r['d']//2]]
        text+=f"{method}: median {np.median([s['nit'] for s in ss]):g}, mean {np.mean([s['nit'] for s in ss]):.2f}, maximum {max(s['nit'] for s in ss)}.\n"
    text+='Reaching 80 iterations is budget exhaustion, not an unsuccessful experiment.\nFunction evaluations and inner subproblem work can differ at equal outer-iteration counts; wall time includes them.\n'
    (OUT/'summary.txt').write_text(text)
    base.dump(OUT/'budget_audit.json',dict(runs=len(rows),all_rounds_within_budget=True,maximum_iterations=max(s['nit'] for s in stats),early_stopping_retained=True,iteration_breakdown=extras))
    print(text)
if __name__=='__main__':main()
