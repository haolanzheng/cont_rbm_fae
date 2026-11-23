import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,json
p=Path(__file__).with_name('run.py')
ns={'__file__':str(p),'__name__':'audit_target'};exec(compile(p.read_text(),str(p),'exec'),ns)
folder=Path(__file__).resolve().parents[4]/'runs/piecewise_forcing_2d_20261112'
rows=[]
for d, ranks in [(64,[8,32]),(256,[8,32,64])]:
 data=json.loads((folder/f'd{d}.json').read_text());zdata=np.load(folder/f'd{d}_snapshots.npz');B,G,_=ns['model'](int(np.sqrt(d)),data['checks']['cutoff'])
 for seed in range(1000,1004):
  for N in ranks:
   Z=zdata[f'z_s{seed}_pool10000_n{N}'].T
   Q=np.linalg.qr(B@Z)[0];H=ns['residual_gram'](B,G,Q)
   # Fresh, disjoint seed ranges from the original training and its audits.
   pool_seed=910000+10000*d+100*seed+N;gd_seed=1910000+10000*d+100*seed+N
   pool=np.random.default_rng(pool_seed).uniform(-1,1,(10000,d))
   vals=np.einsum('ij,ij->i',pool@H,pool);zp=pool[np.argmax(vals)]
   zg=ns['ascend'](H,np.random.default_rng(gd_seed).uniform(-1,1,d),80)
   jp=float(zp@H@zp);jg=float(zg@H@zg)
   gp=float(zp@H@H@zp/(3*jp));gg=float(zg@H@H@zg/(3*jg))
   Qp,_=ns['append'](B,Q,zp);Qg,_=ns['append'](B,Q,zg)
   Hp=ns['residual_gram'](B,G,Qp);Hg=ns['residual_gram'](B,G,Qg)
   directp=float((np.trace(H)-np.trace(Hp))/3);directg=float((np.trace(H)-np.trace(Hg))/3)
   row=dict(d=d,N=N,basis_seed=seed,basis_method='pool10000',pool_seed=pool_seed,gd_seed=gd_seed,J_pool=jp,J_gd=jg,J_ratio=jg/jp,mean_gain_pool=gp,mean_gain_gd=gg,mean_gain_ratio=gg/gp,before_mean_error_squared=float(np.trace(H)/3),max_direct_gain_check=float(max(abs(directp-gp),abs(directg-gg))))
   rows.append(row);print(json.dumps(row),flush=True)
summary=[]
for d in [64,256]:
 rr=[r for r in rows if r['d']==d]
 summary.append(dict(d=d,cases=len(rr),J_gd_gt_pool=sum(r['J_ratio']>1 for r in rr),mean_gain_gd_gt_pool=sum(r['mean_gain_ratio']>1 for r in rr),J_ratio_range=[min(r['J_ratio'] for r in rr),max(r['J_ratio'] for r in rr)],mean_gain_ratio_range=[min(r['mean_gain_ratio'] for r in rr),max(r['mean_gain_ratio'] for r in rr)],mean_gain_ratio_median=float(np.median([r['mean_gain_ratio'] for r in rr]))))
out=dict(protocol='Frozen original pool10000 snapshot spaces; 1 fresh independent 10000-point pool vs 1 fresh independent GD80 start for each space. Exact mean-square reduction under uniform cube in truncated sine model. Finite experiment, not universal ordering.',results=rows,summary=summary)
(folder/'same_space_gain.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
