import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,json
p=Path(__file__).with_name('run.py')
ns={'__file__':str(p),'__name__':'audit_target'}
exec(compile(p.read_text(),str(p),'exec'),ns)
rows=[];rng=np.random.default_rng(93825)
for m,K in [(3,47),(4,64),(8,129)]:
 B,G,ch=ns['model'](m,K); k=np.arange(1,K+1,dtype=float);left=np.arange(m)/m
 I=np.sqrt(2)*(np.cos(np.pi*k[:,None]*left)-np.cos(np.pi*k[:,None]*(left+1/m)))/(np.pi*k[:,None])
 lamb=(np.pi**2*(k[:,None]**2+k[None,:]**2)).ravel()
 F=np.einsum('ki,lj->klij',I,I).reshape(K*K,m*m)
 S=F/lamb[:,None];W=F/np.sqrt(lamb[:,None]);Gdirect=W.T@W
 Z=rng.uniform(-1,1,(m*m,min(5,m*m)))
 snapshots=S@Z;Ar=snapshots.T@(lamb[:,None]*snapshots)
 Q=np.linalg.qr(B@Z)[0];H=ns['residual_gram'](B,G,Q)
 Uc=rng.uniform(-1,1,(m*m,23));truth=S@Uc
 ur=snapshots@np.linalg.solve(Ar,snapshots.T@(F@Uc))
 direct=np.sum(lamb[:,None]*(truth-ur)**2,axis=0)
 compressed=np.einsum('it,ij,jt->t',Uc,H,Uc)
 z=Uc[:,0];j=z@H@z;newQ,_=ns['append'](B,Q,z);newH=ns['residual_gram'](B,G,newQ)
 actual=(np.trace(H)-np.trace(newH))/3;formula=z@H@H@z/(3*j)
 rows.append({'m':m,'cutoff':K,'Gram_relative_Frob_error':float(np.linalg.norm(G-Gdirect)/np.linalg.norm(Gdirect)),'Gram_relative_max_error':float(np.max(abs(G-Gdirect))/np.max(abs(Gdirect))),'Galerkin_error_max_relative_discrepancy':float(np.max(abs(direct-compressed)/direct)),'mean_gain_actual':float(actual),'mean_gain_formula':float(formula),'mean_gain_relative_discrepancy':float(abs(formula-actual)/actual),'constant_source_energy':ch['constant_load_energy_squared']})
# Rebuild stored snapshot spaces and all their reported analytic mean errors.
folder=Path(__file__).resolve().parents[4]/'runs/piecewise_forcing_2d_20261112'
checks=[]
for d in [16,64,256]:
 data=json.loads((folder/f'd{d}.json').read_text());zdata=np.load(folder/f'd{d}_snapshots.npz');B,G,_=ns['model'](int(np.sqrt(d)),data['checks']['cutoff']);delta=[]
 for r in data['results']:
  key=f"z_s{r['seed']}_{r['method']}_n{r['N']}";Z=zdata[key].T
  Q=np.linalg.qr(B@Z)[0];R=B-Q@(Q.T@B)
  exact=np.sum(R*R)/3
  delta.append(abs(exact-r['uniform_mean_squared_error'])/exact)
 checks.append({'d':d,'reconstructed_snapshot_spaces':len(delta),'max_relative_MSE_difference':max(delta)})
report={'direct_sine_checks':rows,'stored_snapshot_reconstruction':checks}
print(json.dumps(report,indent=2))
(folder/'audit.json').write_text(json.dumps(report,indent=2))
