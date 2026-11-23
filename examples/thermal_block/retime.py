"""Rerun training after removing unused caches; verify selected parameters."""
from .run import OUT,dump,train,validate,ThermalBlock,np,json
records=json.loads((OUT/'results.json').read_text());dump(OUT/'results_before_cache_cleanup.json',records)
max_difference=0.
for d in (4,14):
 p=ThermalBlock(2,d//2,28);dump(OUT/f'validation_d{d}.json',validate(p))
 for r in records:
  if r['d']!=d:continue
  rb,seconds,rounds=train(p,r['method'],r['seed'],24)
  assert len(rounds)==len(r['rounds'])
  diff=max(np.max(np.abs(np.array(a['z'])-b['z'])) for a,b in zip(rounds,r['rounds']));max_difference=max(max_difference,float(diff));assert diff<1e-12
  r['training_seconds_before_cache_cleanup']=r['training_seconds'];r['training_seconds']=seconds
  print(d,r['method'],r['seed'],seconds,flush=True)
dump(OUT/'results.json',records);dump(OUT/'retiming_validation.json',dict(max_selected_parameter_difference=max_difference,run_count=len(records)))
