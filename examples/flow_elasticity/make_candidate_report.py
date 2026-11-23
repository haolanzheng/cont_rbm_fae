"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    #!/usr/bin/env python3
    """Create figures and a concise, self-contained report from saved JSON."""
    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    
    ROOT=Path('candidate_bench_results');ROOT.mkdir(exist_ok=True)
    elastic=json.load(open(ROOT/'elastic_full.json'))['grids']['48x12']['runs']
    valid=json.load(open(ROOT/'validation.json'))
    step=valid['step_time_refinement']['runs']
    audit=json.load(open(ROOT/'optimizer_audit.json'))
    
    def arr(runs,method,key):
        return np.array([[h[key] for h in r['history']] for r in runs if r['method']==method])
    
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.0),constrained_layout=True)
    cases=[(axs[0],elastic,'deployment_normalized_worst_energy_error','Perforated elastic beam'),
           (axs[1],step,'deployment_normalized_worst_velocity_error','Backward-facing-step flow')]
    sty={'adam':('C0','Adam, T=100'),'power':('C2','normalized gradient, T=100'),
         'pool':('C1','random pool, M=101'),'pool4096':('C3','random pool, M=4096'),
         'oracle':('k','indicator oracle')}
    for ax,runs,key,title in cases:
        for m in ['adam','power','pool','pool4096','oracle']:
            A=arr(runs,m,key);n=np.arange(1,A.shape[1]+1);c,label=sty[m]
            med=np.median(A,axis=0);ax.plot(n,med,color=c,label=label,lw=2,ls='--' if m=='power' else '-')
            if A.shape[0]>1:
                lo,hi=np.quantile(A,[.25,.75],axis=0);ax.fill_between(n,lo,hi,color=c,alpha=.13)
        ax.set(title=title,xlabel='RB dimension N',ylabel='normalized worst-case error')
        ax.set_yscale('log');ax.grid(True,alpha=.25)
    axs[0].legend(fontsize=8)
    fig.savefig(ROOT/'error_curves.png',dpi=210);plt.close(fig)
    
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.0),constrained_layout=True)
    for ax,cname,title in [(axs[0],'elastic_pq','Elasticity'),(axs[1],'step_pq','Step flow')]:
        aa=valid[cname]['round_audits'];r=[a['before_round'] for a in aa]
        p=np.array([a['p']['0.9'] for a in aa]);pplot=np.maximum(p,1.5e-5)
        q=np.array([a['q_adam_T100']['0.9'] for a in aa])
        # For a zero-hit Monte Carlo estimate, propagate its one-sided 95% upper
        # bound into the corresponding M=101 pool-success upper bound.
        ps=np.where(p>0,1-(1-p)**101,1-(1-1.5e-5)**101)
        ax.plot(r,pplot,'o-',label='p(0.9); zero hits shown at 95% upper bound')
        ax.plot(r,ps,'s-',label='M=101 pool success from p')
        ax.plot(r,q,'^-',label='q(0.9,100), single-start Adam')
        ax.set(title=title,xlabel='greedy round',ylabel='probability',yscale='log',ylim=(8e-6,1.4))
        ax.set_xticks(r)
        ax.grid(True,which='both',alpha=.25)
    axs[0].legend(fontsize=8)
    fig.savefig(ROOT/'pq_audit.png',dpi=210);plt.close(fig)
    
    def stat(runs,method,key,idx):
        a=arr(runs,method,key)[:,idx]
        return float(np.median(a)),[float(x) for x in np.quantile(a,[.25,.75])],[float(a.min()),float(a.max())]
    
    ekey='deployment_normalized_worst_energy_error';skey='deployment_normalized_worst_velocity_error'
    rows=[]
    for name,runs,key,idxs in [('Elastic',elastic,ekey,[3,7,11]),('Step flow',step,skey,[3,7,9])]:
        for idx in idxs:
            for method in ['adam','power','pool','pool4096','oracle']:
                med,iqr,rng=stat(runs,method,key,idx)
                rows.append((name,idx+1,method,med,iqr,rng))
    
    lines=[]
    lines += ['# Continuous-greedy candidate audit: backward-facing-step flow and perforated elasticity','',
    '## Protocol','',
    'Both tests use a single fresh continuous starting point per greedy round and exactly 100 optimizer updates.  The matched random pool contains 101 fresh directions, counting the continuous initial evaluation.  A 4096-point pool and the exact largest-eigenvalue direction are additional references.  Parameter selection uses only a computable residual indicator; full-order solutions enter only after selection and in the independent deployment audit.  Every method uses the same one-snapshot enrichment and the same physical parameter distribution.','',
    'The reported deployment quantity is','',
    r'\[\frac{\sup_{\|z\|_2=1}\|u(z)-P_{V_N}u(z)\|}{\sup_{\|z\|_2=1}\|u(z)\|}.\]','',
    r'Because both models are linear in the coefficient vector, this supremum is evaluated exactly from a small generalized eigenproblem rather than from a finite test set.  The pointwise relative-error supremum is not used: for a homogeneous linear input family it is identically one whenever \(N<d\).','',
    '## PDE models','',
    r'**Elastic structure.** A two-dimensional plane-stress cantilever on \([0,4]\times[0,1]\), with two elliptical holes, a two-layer Young modulus, and smooth spatial heterogeneity.  Twenty overlapping localized traction modes act on the top and right boundaries.  The final mesh has 1,075 triangles and 1,204 free displacement degrees of freedom.  Selection maximizes a diagonally preconditioned residual norm; deployment uses the energy norm.','',
    r'**Step flow.** Taylor--Hood \(P_2/P_1\) discretization of the steady incompressible Navier--Stokes equations on a backward-facing-step channel, followed by the linearized perturbation equations at \(\mathrm{Re}=80\).  The fine mesh has 1,036 triangles, 3,984 free velocity unknowns, and 573 pressure unknowns.  Sixteen overlapping divergence-free localized forces near the step are multiplied by a fixed smooth time pulse.  The final time is 2.8 and the audited time step is 0.04.  The residual dual norm projects out pressure gradients.  The base-flow Picard update reached \(1.99\times10^{-9}\); the initial forcing dictionary has relative discrete divergence \(1.9\times10^{-17}\).  A lower-channel probe finds reverse flow down to \(-0.069\) near \(x=0.78\).','',
    '## Main results','',
    '| Case | N | method | median error | IQR | full range |','|---|---:|---|---:|---:|---:|']
    for name,N,m,med,iqr,rng in rows:
        lines.append(f'| {name} | {N} | {m} | {med:.6f} | [{iqr[0]:.6f}, {iqr[1]:.6f}] | [{rng[0]:.6f}, {rng[1]:.6f}] |')
    lines += ['',
    r'On the fine elasticity mesh at \(N=12\), Adam gives 0.007407, the matched pool 0.009741 (+31.5%), and the 4096-point pool 0.008815 (+19.0%).  Adam, normalized gradient, and exact indicator maximization agree to plotting precision.  The same ordering and nearly the same relative gaps occur on the 564-unknown coarse mesh.','',
    r'For step flow at \(N=10\), Adam gives 0.21208, the matched pool 0.26125 (+23.2%), and the 4096-point pool 0.22780 (+7.4%).  At \(N=8\) the corresponding gaps are 19.7% and 7.0%.  Repeating on the coarser spatial mesh with the same 0.04 time step gives 0.20699, 0.25944, and 0.23834, respectively.  Thus the matched-budget advantage survives both spatial and temporal refinement.','',
    '## Search-geometry and optimizer audits','',
    r'At four representative rounds, 200 independent single-start Adam runs all exceeded \(\gamma=0.99\).  In the step-flow case, only \(1.5\times10^{-5}\) to \(6\times10^{-5}\) of 200,000 uniform sphere samples exceeded \(\gamma=0.9\).  In elasticity there were zero such hits; the one-sided 95% upper bound is about \(1.5\times10^{-5}\).  This directly realizes \(p_{N,\gamma}\ll q_{N,100}(\gamma)\).','',
    'The result is not tied to a narrowly selected learning rate.  At representative round 8, 100-start sweeps gave at least 99.97% of the indicator oracle for elasticity over learning rates 0.02--0.20 and for step flow over 0.02--0.12.  Normalized gradient reaches a median ratio around 99.9% in 10 steps, but its worst start is only about 88%; 50 steps are needed for uniformly near-oracle behavior in this audit.','',
    '## Interpretation and limits','',
    'These are strong positive tests of the **continuous-search mechanism**: the high-value set has tiny sampling mass while a single gradient trajectory has a large attraction region.  They are not yet a final nonlinear headline example.  For both current models the parameter-to-state map is linear, so the frozen-round indicator is quadratic.  Once its small matrix is explicitly assembled, a largest-eigenvalue solver or normalized gradient is preferable to generic Adam.  The experiments therefore validate the theory and identify a favorable PDE class, while a nonlinear operator-parameter example is still needed to show that the advantage survives when no eigenproblem reduction is available.','',
    r'The wall-clock data do not show a universal speed advantage.  On elasticity, vectorized pool evaluation is faster than sequential Python Adam; on step flow, residual-matrix assembly dominates and the methods have similar times.  The meaningful comparison here is error per 101 candidate evaluations and the measured \(p\)-versus-\(q\) gap.','',
    'Exact indicator maximization also does not solve the globally optimal basis-selection problem.  In step flow, a few 4096-point pool runs finish below the oracle-greedy error because a weaker early choice can lead to a better later subspace.  The continuous method wins in median and is far less variable at the matched budget; it does not dominate every possible greedy path.','',
    '![Worst-case error curves](error_curves.png)','',
    '![High-value mass and attraction probability](pq_audit.png)','']
    content='\n'.join(lines)
    # Python interprets a few TeX escapes (notably \f and \t) inside ordinary
    # string literals.  Restore those controls before writing Markdown/KaTeX.
    for control,escaped in [('\x0c',r'\f'),('\t',r'\t'),('\r',r'\r'),('\x08',r'\b')]:
        content=content.replace(control,escaped)
    (ROOT/'REPORT.md').write_text(content)
    print(ROOT/'REPORT.md')


if __name__ == "__main__":
    main()
