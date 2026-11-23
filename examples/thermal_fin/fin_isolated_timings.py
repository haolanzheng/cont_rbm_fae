"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Sequential timing repetitions after the experiment/audit queues finish."""
    import time,os,json
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from examples.thermal_fin.fin_local_study import Fin,run
    
    methods=['pool2000','pool4000','local','lbfgs5','lbfgs10','lbfgs20','lbfgs100','adam100','resamp2000']
    required=[Path(f'fin_d{d}/{m}_{s}.json') for d in [11,21,33] for m in methods for s in [11,22,33]]
    required += [Path(f'fin_d{d}/local_{s}_witness.json') for d in [11,21,33] for s in [11,22,33]]
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise FileNotFoundError('Missing timing prerequisites: '+', '.join(missing))
    result=[]
    with threadpool_limits(limits=1):
        for d,n in [(11,50),(21,80),(33,100)]:
            out=Path(f'fin_timing_d{d}');out.mkdir(exist_ok=True)
            for f in ['test_meta.npz','test_u.npy']:
                p=out/f
                if not p.exists():p.symlink_to((Path(f'fin_d{d}')/f).resolve())
            p=Fin(d)
            for method in ['lbfgs20','pool2000','lbfgs5','local']:
                run(p,method,11,n,out)
                x=json.loads((out/f'{method}_11.json').read_text());baseline=json.loads(Path(f'fin_d{d}/{method}_11.json').read_text())
                result.append({'d':d,'rank':n,'method':method,'search_seconds':x['search_seconds'],'enrichment_seconds':x['enrichment_seconds'],
                               'offline_seconds':x['offline_seconds'],'max_error':x['metrics'][str(n)]['max'],
                               'max_error_difference_from_main_run':abs(x['metrics'][str(n)]['max']-baseline['metrics'][str(n)]['max'])})
                Path('fin_isolated_timings.json').write_text(json.dumps(result,indent=2))
    print('TIMINGS COMPLETE',flush=True)


if __name__ == "__main__":
    main()
