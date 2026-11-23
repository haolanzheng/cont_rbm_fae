"""Full greedy T-study using unchanged starts, tests and optimized NumPy GD.

Accuracy is recomputed through every saved plotting rank. Timings end at
floor(d/2) and compare all methods in randomized sequential order.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/piecewise_steps_mpl')
import argparse,json,csv,hashlib,platform
from pathlib import Path
import numpy as np
from run import model
from equal_time_pool import train
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'runs/piecewise_forcing_2d_20261112'
SUMMARY=BASE/'slides_summary'
SEEDS=[1000,1001,1002,1003]

def main(args):
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():raise FileExistsError('Use a new output directory.')
 old=json.loads((SUMMARY/'shared_test_errors.json').read_text());reports=[]
 for d in args.dims:
  B,G,checks=model(round(np.sqrt(d)));N=d//2
  max_n=len(next(r['energy_error'] for r in old[str(d)]['curves'] if r['method']=='gd80'))-1
  test_path=SUMMARY/f'd{d}_test.npz';test=np.load(test_path)['parameters'];U=test@B.T;norm2=np.sum(U**2,axis=1)
  if d==4:saved=json.loads((SUMMARY/'d4.json').read_text())
  else:saved=np.load(BASE/('dimension_study' if d==16 else 'square_dimension_study')/f'd{d}_snapshots.npz')
  curves=[];snapshots={};validation=[]
  for steps in args.steps:
   for seed in SEEDS:
    Q,Z,_=train(B,G,seed,0,steps=steps,max_n=max_n)
    snapshots[f'z_s{seed}_gd{steps}']=Z
    e2=np.maximum(0,norm2[:,None]-np.cumsum((U@Q)**2,axis=1))
    curve=np.r_[np.sqrt(norm2.max()),np.sqrt(e2.max(axis=0))]
    direct=float(np.linalg.norm(test@(B-Q[:,:N]@(Q[:,:N].T@B)).T,axis=1).max())
    np.testing.assert_allclose(curve[N],direct,atol=1e-13,rtol=1e-11)
    assert np.all(np.diff(curve)<=1e-12)
    orth=float(np.linalg.norm(Q.T@Q-np.eye(max_n)));assert orth<1e-11
    check=dict(steps=steps,seed=seed,orthogonality_defect=orth,direct_error_difference=float(abs(curve[N]-direct)))
    if steps==80:
     ref=np.array(next(r['parameters'] for r in saved['curves'] if r['method']=='gd80' and r['seed']==seed)) if d==4 else saved[f'z_s{seed}_gd80']
     np.testing.assert_array_equal(Z,ref[:max_n])
     old_curve=next(r['energy_error'] for r in old[str(d)]['curves'] if r['method']=='gd80' and r['seed']==seed)
     np.testing.assert_allclose(curve,old_curve,rtol=1e-10,atol=1e-12)
     check['original_parameters_identical']=True
    validation.append(check);curves.append(dict(method=f'gd{steps}',seed=seed,energy_error=curve.tolist()))
  # Existing pools use exactly the same test set and training settings.
  for r in old[str(d)]['curves']:
   if r['method'].startswith('pool'):curves.append(r)
  p40=SUMMARY/'pool40000_20261116'/f'd{d}.json'
  if p40.exists():
   for r in json.loads(p40.read_text())['results']:
    if r['M']==40000:curves.append(dict(method='pool40000',seed=r['seed'],energy_error=r['error_curve']))
  print(f'd={d}: full T=20 and T=80 greedy paths evaluated',flush=True)
  timings=[]
  if d in args.timing_dims:
   jobs=[(f'gd{t}',seed) for t in args.steps for seed in SEEDS]+[(f'pool{M}',seed) for M in [10000,40000] for seed in SEEDS]
   # Warm and validate all timing paths outside the timed repetitions.
   for method,seed in jobs:
    M=int(method[4:]) if method.startswith('pool') else 0
    steps=80 if M else int(method[2:]);Q,Z,_=train(B,G,seed,M,steps=steps)
    if not M:np.testing.assert_array_equal(Z,snapshots[f'z_s{seed}_{method}'][:N])
    else:
     ref=np.load(SUMMARY/'pool40000_20261116'/f'd{d}_snapshots.npz')[f'z_s{seed}_{method}']
     np.testing.assert_array_equal(Z,ref)
   order=np.random.default_rng(562000+d)
   for repeat in range(args.repeats):
    for idx in order.permutation(len(jobs)):
     method,seed=jobs[idx];M=int(method[4:]) if method.startswith('pool') else 0
     _,_,t=train(B,G,seed,M,steps=80 if M else int(method[2:]))
     timings.append(dict(method=method,seed=seed,repeat=repeat,**t))
  methods=list(dict.fromkeys(r['method'] for r in curves));summary=[]
  for method in methods:
   values=np.array([r['energy_error'][N] for r in curves if r['method']==method])
   times=[r['wall_seconds'] for r in timings if r['method']==method]
   summary.append(dict(method=method,error_median=float(np.median(values)),error_min=float(values.min()),error_max=float(values.max()),
    wall_ms_median=float(1000*np.median(times)) if times else None,
    wall_ms_q25=float(1000*np.quantile(times,.25)) if times else None,
    wall_ms_q75=float(1000*np.quantile(times,.75)) if times else None))
  paired=[]
  for seed in SEEDS:
   e20=next(r['energy_error'][N] for r in curves if r['method']=='gd20' and r['seed']==seed)
   e80=next(r['energy_error'][N] for r in curves if r['method']=='gd80' and r['seed']==seed)
   paired.append(dict(seed=seed,error20=e20,error80=e80,relative_change=e20/e80-1))
  report=dict(d=d,N=N,max_plot_N=max_n,checks=checks,curves=curves,timings=timings,summary=summary,paired=paired,validation=validation,
   test_file=str(test_path.relative_to(ROOT)),test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest())
  np.savez_compressed(out/f'd{d}_snapshots.npz',**snapshots)
  (out/f'd{d}.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
  print(json.dumps(dict(d=d,N=N,summary=summary)),flush=True)
 from threadpoolctl import threadpool_info
 env=threadpool_info();assert all(t['num_threads']==1 for t in env)
 settings=dict(steps=args.steps,alpha=.6,seeds=SEEDS,repeats=args.repeats,dimensions=args.dims,timing_dimensions=args.timing_dims,N='floor(d/2)',
  tests='Unchanged saved 4096 uniform interior points + 4096 uniform vertices',
  timing='Full greedy training, including sampling, search, matrix construction and enrichment; shared model, test errors, validation and I/O excluded.',
  initialization='Same uniform start at each greedy round for T20 and T80; independent training paths from V0.',
  platform=platform.platform(),threadpools=env,numpy=np.__version__)
 (out/'summary.json').write_text(json.dumps(dict(settings=settings,reports=[{k:r[k] for k in ['d','N','summary','paired']} for r in reports]),indent=2)+'\n')
 with (out/'summary.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['d','N','method','error_median','wall_ms_median'])
  for r in reports:
   for s in r['summary']:writer.writerow([r['d'],r['N'],s['method'],s['error_median'],s['wall_ms_median']])
 plot(reports,out)
 print('COMPLETE',flush=True)

def plot(reports,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 styles=[('gd20','GD, T=20','#0072B2','o','-'),('gd80','GD, T=80','#6B72A8','s','--'),('pool10000','Pool 10000','#C74432','D','-.'),('pool40000','Pool 40000','#009E73','^',':')]
 fig,ax=plt.subplots(figsize=(8.4,4.5),layout='constrained')
 for method,label,color,marker,ls in styles:
  pts=[(r['d'],next(s['error_median'] for s in r['summary'] if s['method']==method)) for r in reports if any(s['method']==method for s in r['summary'])]
  ax.plot(*np.array(pts).T,label=label,color=color,marker=marker,ls=ls,ms=4,lw=1.7)
 ax.set_xscale('log',base=2);ax.set_xticks([4,16,32,64,128,256],labels=['4','16','32','64','128','256'])
 ax.set_xlabel('$d$');ax.set_ylabel('$E_N$');ax.set_title(r'$N=\lfloor d/2\rfloor$');ax.grid(alpha=.18);ax.legend(frameon=False)
 fig.savefig(out/'error_vs_dimension.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for ax,d in zip(axes.flat,[16,64,144,256]):
  r=next(r for r in reports if r['d']==d)
  for method,label,color,marker,ls in styles:
   curves=[x['energy_error'][:r['N']+1] for x in r['curves'] if x['method']==method]
   y=np.median(curves,axis=0);ax.semilogy(range(len(y)),y,label=label,color=color,marker=marker,markevery=max(1,len(y)//7),ms=3.5,ls=ls)
  ax.set_title(f'$d={d}$');ax.set_xlabel('$N$');ax.set_ylabel('$E_N$');ax.grid(alpha=.18)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
 fig.savefig(out/'error_vs_N.pdf');plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--dims',nargs='+',type=int,default=[4,16]+[m*m for m in range(5,17)])
 p.add_argument('--timing-dims',nargs='+',type=int,default=[4,16,25,64,100,144,256])
 p.add_argument('--steps',nargs='+',type=int,default=[20,80]);p.add_argument('--repeats',type=int,default=5)
 main(p.parse_args())
