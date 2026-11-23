"""Summarize every seed; never select a favorable seed for the error table."""
import json
from pathlib import Path
import numpy as np

METHODS=['pool2000','pool4000','resamp2000','local','lbfgs5','lbfgs10','lbfgs20','lbfgs100','adam100']

def main():
    runs=[];groups=[];audits=[]
    for d,n in [(11,50),(21,80),(33,100)]:
        for method in METHODS:
            group=[]
            for seed in [11,22,33]:
                data=json.loads(Path(f'fin_d{d}/{method}_{seed}.json').read_text());r=data['records']
                row={'d':d,'N':n,'method':method,'seed':seed,**data['metrics'][str(n)],
                     'search_seconds_concurrent':data['search_seconds'],'offline_seconds_concurrent':data['offline_seconds']}
                if 'iterations' in r[0]:
                    row.update({'median_iterations':float(np.median([a['iterations'] for a in r])),
                                'max_iterations':max(a['iterations'] for a in r),'median_fg_calls':float(np.median([a['fg_calls'] for a in r]))})
                runs.append(row);group.append(row)
            stats={'d':d,'N':n,'method':method,'seeds':[11,22,33]}
            for name in ['max','uniform_max','uniform_mean','uniform_q99','vertices_max','tilted_max']:
                values=[a[name] for a in group];stats[name+'_median']=float(np.median(values));stats[name+'_min']=min(values);stats[name+'_max']=max(values)
            groups.append(stats)
        local=[json.loads(Path(f'fin_d{d}/local_{s}.json').read_text()) for s in [11,22,33]]
        records=sum([x['records'] for x in local],[])
        witness=[json.loads(Path(f'fin_d{d}/local_{s}_witness.json').read_text()) for s in [11,22,33]]
        audits.append({'d':d,'N':n,'total_selections':len(records),
                       'first_order_stationary':sum(r['first_order_stationary'] for r in records),
                       'numerical_strict_local_maxima':sum(r['numerical_local_max'] for r in records),
                       'numerically_verified_nonglobal_local_maxima':sum(w['nonglobal_local_max_count'] for w in witness),
                       'median_iterations':float(np.median([r['iterations'] for r in records])),
                       'p90_iterations':float(np.quantile([r['iterations'] for r in records],.9)),
                       'max_iterations':max(r['iterations'] for r in records),
                       'median_fg_calls':float(np.median([r['fg_calls'] for r in records]))})
    timings=json.loads(Path('fin_isolated_timings.json').read_text())
    assert len(runs)==81 and len(timings)==12
    data={'test_points':4608,'uniform_points_for_mean':4096,'groups':groups,'runs':runs,'local_audits':audits,'isolated_timings_seed11':timings}
    Path('continuous_greedy_results.json').write_text(json.dumps(data,indent=2))
    for d in [11,21,33]:
        print('d =',d)
        for x in groups:
            if x['d']==d:print(x['method'],round(x['max_median'],6),round(x['uniform_mean_median'],6))

if __name__=='__main__':main()
