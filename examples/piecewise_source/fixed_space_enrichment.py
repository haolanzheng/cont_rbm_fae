"""Link the saved source trajectory to error after one independent enrichment.

Every candidate enriches the same GD-generated V8, not the space enriched by
preceding candidates. Tests and pool selections are reused without searching
for additional test points. Verify the rank-one error identity against direct
projection in the energy coordinates.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/piecewise_enrichment_mpl')
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import model,append,residual_gram
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/piecewise_forcing_2d_20261112'
SUMMARY=BASE/'slides_summary'

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    B,G,_=model(8);saved=np.load(BASE/'square_dimension_study/d64_snapshots.npz')['z_s1000_gd80']
    Q=np.zeros((64,0))
    for z in saved[:8]:Q,_=append(B,Q,z)
    H=residual_gram(B,G,Q)
    history_file=SUMMARY/'d64_n8_history.npz';history=np.load(history_file)
    np.testing.assert_allclose(H,history['H'],rtol=1e-13,atol=1e-15)
    test_file=SUMMARY/'d64_test.npz';test=np.load(test_file)['parameters']
    assert test.shape==(8192,64)
    R=B-Q@(Q.T@B);residual=test@R.T;before=np.sum(residual**2,axis=1)
    Hdirect=R.T@R;np.testing.assert_allclose(H,Hdirect,rtol=1e-10,atol=1e-15)
    def evaluate(z):
        denominator=float(z@H@z);assert denominator>1e-15
        squared=before-(test@H@z)**2/denominator
        assert squared.min()>-1e-12
        Qnew,_=append(B,Q,z)
        direct=np.sum((test@(B-Qnew@(Qnew.T@B)).T)**2,axis=1)
        np.testing.assert_allclose(squared,direct,atol=1e-14,rtol=1e-10)
        assert direct.max()<=before.max()+1e-14
        return dict(J=float(denominator/np.linalg.norm(H)),E9=float(np.sqrt(direct.max())),
                    identity_max_abs_error=float(np.max(abs(squared-direct))))
    gd=[dict(t=t,**evaluate(z)) for t,z in enumerate(history['parameters'])]
    selected=json.loads((SUMMARY/'pool_source_comparison.json').read_text())
    pools=[]
    for M in [256,4096,10000]:
        row=selected['selected'][str(M)];z=np.array(row['parameter']);v=evaluate(z)
        np.testing.assert_allclose(v['J'],row['J'],rtol=1e-12)
        pools.append(dict(M=M,**v))
    report=dict(d=64,n=8,seed=1000,T=80,alpha=.6,test_count=len(test),
        definition='Add each candidate snapshot independently to the same V8; evaluate maximum energy error on the unchanged common test set.',
        E8=float(np.sqrt(before.max())),gd=gd,pools=pools,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [history_file,test_file,SUMMARY/'pool_source_comparison.json']})
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.4,4.2),layout='constrained')
    ax.plot([r['t'] for r in gd],[r['E9'] for r in gd],color='#0072B2',lw=2,marker='o',markevery=20,ms=5,label='GD candidate')
    for row,color,ls in zip(pools,['#555555','#009E73','#C74432'],[':', '-.', '--']):
        ax.axhline(row['E9'],color=color,ls=ls,lw=1.8,label=f"Pool {row['M']}")
    ax.set_xlabel(r'GD iteration $t$');ax.set_ylabel(r'$E_9$');ax.set_xlim(0,80);ax.set_xticks([0,20,40,60,80])
    ax.grid(alpha=.18);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.legend(frameon=False,ncol=4,fontsize=10.5,loc='lower center',bbox_to_anchor=(.5,1.02))
    fig.savefig(out/'piecewise_source_enrichment_history.pdf');plt.close(fig)
    print(json.dumps(dict(E8=report['E8'],GD_initial=gd[0],GD_final=gd[-1],pools=pools,
        GD_max_increase=float(np.max(np.diff([r['E9'] for r in gd])))),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
