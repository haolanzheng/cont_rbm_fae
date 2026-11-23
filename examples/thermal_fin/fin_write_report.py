"""Explicit historical report/audit command; importing it does not run the experiment."""

def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    """Build the human-readable report from the complete, unfiltered result set."""
    import json,platform,sys
    from pathlib import Path
    import numpy as np,scipy,matplotlib,threadpoolctl
    
    data=json.loads(Path('continuous_greedy_results.json').read_text())
    g={(x['d'],x['method']):x for x in data['groups']}
    t={(x['d'],x['method']):x for x in data['isolated_timings_seed11']}
    names={'pool2000':"Fixed pool, 2000 points",'pool4000':"Fixed pool, 4000 points",'resamp2000':"Fresh pool, 2000 points per round",'local':"Fully converged local search",'lbfgs5':"L-BFGS-B, at most 5 steps",'lbfgs10':"L-BFGS-B, at most 10 steps",'lbfgs20':"L-BFGS-B, at most 20 steps",'lbfgs100':"L-BFGS-B, at most 100 steps",'adam100':"Original Adam, 100 steps"}
    
    def metric_table(metric):
        rows=["| Method | d=11, N=50 | d=21, N=80 | d=33, N=100 |",'|---|---:|---:|---:|']
        for m in names:rows.append('| '+names[m]+' | '+' | '.join(f'{g[d,m][metric+"_median"]:.6f}' for d in [11,21,33])+' |')
        return '\n'.join(rows)
    
    audit_rows=["| d, N | Selections | First-order checks passed | Strict local-maximum checks passed | Also has a nonglobality witness | Median / maximum iterations |",'|---|---:|---:|---:|---:|---:|']
    for a in data['local_audits']:
        audit_rows.append(f'| {a["d"]}, {a["N"]} | {a["total_selections"]} | {a["first_order_stationary"]} | {a["numerical_strict_local_maxima"]} | {a["numerically_verified_nonglobal_local_maxima"]} | {a["median_iterations"]:.0f} / {a["max_iterations"]} |')
    
    time_rows=["| d, N | Method | Selection time / s | Training time / s | Seed-11 maximum error |",'|---|---|---:|---:|---:|']
    for d,n in [(11,50),(21,80),(33,100)]:
        for m in ['pool2000','lbfgs5','lbfgs20','local']:
            a=t[d,m];time_rows.append(f'| {d}, {n} | {names[m]} | {a["search_seconds"]:.3f} | {a["offline_seconds"]:.3f} | {a["max_error"]:.6f} |')
    
    seed_rows=["| d, N | Method | seed 11 | seed 22 | seed 33 |",'|---|---|---:|---:|---:|']
    for d,n in [(11,50),(21,80),(33,100)]:
        for m in names:
            r=sorted([x for x in data['runs'] if x['d']==d and x['method']==m],key=lambda x:x['seed'])
            seed_rows.append(f'| {d}, {n} | {names[m]} | '+' | '.join(f'{a["max"]:.6f}' for a in r)+' |')
    
    report=r'''# Continuous greedy: local maxima, limited iterations, and theoretical separation

Date: 2026-09-16. The main experiments comprise three parameter dimensions, nine selection methods, and three training seeds: 81 runs in total. Nine additional nonglobality audits and 12 sequential timing reruns were completed.

## Main findings

1. **Local maximization has positive empirical support.** At the same reduced dimension, fully converged local search has a smaller median test maximum error than fixed pools of 2000 and 4000 points and a fresh pool of 2000 points in all three dimensions. Reductions against the fixed 2000-point pool are 20.2%, 21.4%, and 9.4%; against the best pool median in each dimension, they are 15.5%, 18.7%, and 9.4%. These compare medians, rather than taking the median of seedwise reductions.
2. **These results do not mean that a few iterations reach a local maximum.** The pooled median iteration counts for fully converged local search are 40, 106, and 215. Converging the local solve becomes substantially more expensive in higher dimensions.
3. **A small iteration budget gives a different tradeoff.** The five-step variant has median maximum errors close to or below the fixed 2000-point pool, but is not best against every pool in every dimension. In separate seed-11 timings, its training times are 23.5%, 26.9%, and 14.1% lower. These are observations on the local CPU implementation.
4. **Mean error does not improve at the same time.** Fully converged local search has larger mean error under the uniform distribution than the fixed pool, especially at d=33. These results support neither simultaneous improvement of maximum and mean errors nor an explanation based on implicit regularization.
5. **The theoretical analysis has two distinct levels.** One is a conditional weak-greedy convergence statement allowing unsuccessful rounds. The other separates finite-candidate search from single-start continuous search in an analytic Poisson model. The latter concerns excess error at fixed rank, not the general absolute-accuracy complexity of reduced basis methods.

## 1. Relation to the reference experiment

The original complete source was unavailable, so this is an independent implementation rather than a bitwise rerun. It retains the thermal-fin PDE, all 16 material/geometry regions, reference mesh h=1/32, 10017 P1 nodes, parameter maps, and fixed X norm from the supplied PDF. Triangles use SW–NE diagonals; the reference does not specify that convention. Test points follow the stated construction and seeds, but original coordinates were unavailable for direct comparison.

The experiment removes alpha and uses final dimensions (d,N)=(11,50),(21,80),(33,100). Its values therefore should not be used for direct method rankings against the reference tables with alpha and larger N.

### Equations and parameters

Each material region satisfies -div(kappa grad u)=0. At the root, kappa ∂νu=1; elsewhere on the outer boundary, kappa ∂νu+Bi u=0. Temperature and flux are continuous across internal interfaces. The reference geometry has 16 material regions and four pairs of fins.

For d=2m+1, the first m conductivities, Bi, and the first m geometric variables are active; the remaining quantities take their reference physical values. Conductivities lie in [0.1,10] and Bi in [0.01,5], with logarithmic parameter maps. Spacing lies in [0.1,5], fin length in [0.1,10], and thickness in [0.1,2.5], with linear maps. The parameter domain is the full cube [-1,1]^d, without restriction to a sphere, a face, or fixed active coordinates.

The basis is orthogonalized in the X norm defined by a fixed physical reference operator. The initial snapshot uses the normalized center z=0, which differs from the physical reference parameter. Identical monomials in the 49-term affine expansion are combined algebraically, yielding Q=20,35,49 without truncation.

### Indicator and parameter selection

\[
\eta_n(z)^2=\frac{r_n(z)^T X^{-1}r_n(z)}{\|u_n(z)\|_X^2},
\qquad J_n(z)=\log \eta_n(z)^2.
\]

Each round evaluates the indicator and gradient using only the current reduced Galerkin solution and offline residual Gram data. Full-order errors are not used for selection. One full-order snapshot is computed after selection. Every method constructs a space of exactly dimension N.

Continuous search uses one uniform start per round, shared across iteration budgets. As reduced spaces diverge, the methods face different objectives. There is no best-of-multiple-start selection. The fixed 4000-point pool contains the fixed 2000-point pool with the same seed.

L-BFGS-B handles the box constraints and can move between the interior and boundary without prescribing a face. Fully converged local search permits at most 1000 L-BFGS-B iterations. If the final projected gradient exceeds 1e-5, SLSQP continues from that endpoint for at most 300 iterations, without resampling. The point with the largest indicator along the trajectory is returned. Rounds with incomplete numerical convergence are retained and reported.

Adam 100 is a historical reference with learning rate 0.05, betas=(0.9,0.99), clipping, and best-iterate return. No learning-rate sweep was performed for this comparison, so this is not a tuned optimal Adam configuration.

## 2. Maximum error at equal reduced dimension

Each dimension uses the same independent 4608-point test set: 4096 scrambled Sobol interior points, 256 random vertices, and 256 points sampled preferentially near the boundary. Tables show the median, over training seeds 11,22,33, of each run's maximum relative X error. The reference is the full finite-element solution. The reported maximum is over a finite test set, **not a certified supremum over the parameter domain**.

{{MAX_TABLE}}

Fully converged local search gives lower errors in 9/9 paired comparisons against fixed 2000, and in 8/9 against both fixed 4000 and fresh 2000. Three seeds do not establish statistical significance or general dominance. At d=21, local-search errors are 0.58085, 0.53066, and 0.73806, with substantial variation. The 100-step variant also has a better median than fully converged local search at d=21: solving the selection objective more accurately need not improve the final space.

### Mean-error counterexample

The following mean uses only the 4096 uniform Sobol interior points; vertices and boundary-biased points are excluded from uniform-distribution statistics.

{{MEAN_TABLE}}

Gains are concentrated in the maximum error on the test set with additional boundary samples and do not extend to the uniform mean. The concentration of continuous selections near the boundary may explain part of this tradeoff, but the experiment does not provide sufficient controlled interventions to establish causality.

## 3. Are the selected points nonglobal local maxima?

This check does not require the unknown global maximum of the indicator. Each fully converged local-search result is assessed using:

- Box-projected gradient infinity norm <=1e-5.
- Largest finite-difference Hessian eigenvalue <-1e-6 on the span of critical directions outside strongly active constraints. If this space is empty, only the first-order condition is required.
- A separate search over 4000 uniform points and 1024 vertices, never used for selection, for a point with larger indicator. Any such witness is checked by a direct full-space residual and Riesz-norm calculation rather than Gram contraction alone.

The first two checks give conservative numerical evidence of a local maximum; the third disproves global optimality. None estimates the unknown global maximum.

{{AUDIT_TABLE}}

These checks are not interval-arithmetic certificates. Some points failing the strict curvature test are nearly flat and cannot automatically be classified as saddles. A few first-order residuals remain slightly above tolerance. In particular, 72/297 selections at d=33 fail the strict local-maximum check, so not every round is numerically verified as a local maximum.

The results support the claim that a single-start construction using many nonglobal local maxima can beat a pool in final test maximum error. They do not isolate the causal contribution of each local solution or show that greater locality is better.

## 4. Limited iterations and training time

The main runs execute concurrently and are not used for timing conclusions. After those runs and witness checks, seed 11 is rerun sequentially with one BLAS thread. Training time includes parameter selection, initial and added full-order snapshots, basis updates, and reduced-data updates. It excludes common mesh assembly, test generation/evaluation, post hoc curvature/witness checks, and file output. Fixed-pool initialization also excludes candidate-coefficient generation and allocation, slightly favoring the pool. Timing reruns reproduce the main-run final errors.

{{TIME_TABLE}}

The fixed pool uses cross-round caching and incremental contractions. The timing comparison does not substitute a more expensive fresh-pool implementation to enlarge the apparent advantage. Five-step continuous search accelerates selection by about 5.85,3.15,1.80 times, but shared full-order and basis-update costs reduce full-training speedups to about 1.31,1.37,1.16.

Fully converged local search costs about 2.06 and 6.52 times as much as fixed-pool training at d=21 and d=33. The 20-step variant is also slower at d=33. One sequential timing per dimension gives no confidence interval or fixed speedup for GPUs or other implementations.

Five steps means at most five L-BFGS-B iterations, not five indicator evaluations. Line searches require additional calls. Across the three dimensions, the median number of value-and-gradient calls per round is 8, with maxima 31,29,28. Raw data retain all call counts.

## 5. First theoretical direction: additional exponents in random-training-set bounds

Cohen–Dahmen–DeVore–Nichols (2020) establish polynomial complexity bounds for random training sets under sparse polynomial approximation assumptions. Additional exponents in those upper bounds do not establish an unavoidable random-pool lower bound or a general curse of dimensionality. [Paper and Theorem 4.1](https://www.numdam.org/item/10.1051/m2an/2020004.pdf)

A conditional connection is proposed: if each single-start search, conditional on the entire history, produces a gamma0-weak-greedy point with probability at least q0, and q0 and gamma0 do not deteriorate with accuracy or iteration, then unsuccessful rounds can be allowed while obtaining the Hilbert-space projection-error estimate

\[
\Pr\{\sigma_N\le C_s\gamma_0^{-2}(q_0N/2)^{-s}\}
\ge1-e^{-q_0N/8},\qquad d_n(\mathcal M)\lesssim n^{-s}.
\]

The argument combines determinant estimates on successful rounds with conditional Bernoulli counting. It requires neither success in every round nor multiple starts. Details are in the separate theory notes; the determinant argument follows [DeVore–Petrova–Wojtaszczyk](https://arxiv.org/pdf/1204.2290).

**The remaining gap is explicit:** uniform q0, gamma0, and optimization costs have not been derived from thermal-fin or general A^r solution-map assumptions. Uniform equivalence constants between the raw relative residual and true projection error are also unavailable. This is a conditional theorem framework, not a resolution of the original complexity problem. The initialization mass of useful attraction regions is central. A beta log pi term need not be introduced first, and no such density term is tested here.

## 6. Second theoretical direction: candidate-set lower bounds and continuous-search upper bounds

A separate analytic, strictly coercive 1D Poisson model has two parameters and solutions in the span of three energy-orthogonal modes. It illustrates a mechanism, rather than providing a new high-dimensional physical benchmark.

Let phi_k=sqrt(2)/(k*pi) sin(k*pi*x), u(t,z)=phi_1+t phi_2+t g(z) phi_3, with (t,z)∈[-1,1]^2. Set f=-u_xx and impose zero Dirichlet conditions. The function g has a broad peak of height a*=sqrt(3)-1 and a narrow peak of height 2. The complete C^3 definition is recorded in the theory notes.

Starting from the rank-1 space generated by the central snapshot, the squared raw indicator is t²(1+g²), exactly the current squared absolute energy error. After adding a snapshot with g=a, the rank-2 space has exact global maximum relative energy error

\[
E(a)=\frac{\max\{a/\sqrt2,(2-a)/\sqrt6\}}{\sqrt{1+a^2}}.
\]

The nonglobal local peak gives E(a*)=0.417681, whereas the global peak gives E(2)=0.632456. The local-peak error is about 34% lower. Thus even with an exact indicator, maximizing the current error and minimizing the final maximum error at fixed rank are different objectives.

Any selector restricted to adding one snapshot from M iid uniform candidates, including an oracle with access to the final error, must satisfy

\[
M=\Omega\bigl(\delta^{-1/2}\log(1/\rho)\bigr)
\]

to attain E<=E(a*)+delta with probability at least 1-rho.

The useful snapshot set has probability mass Theta(sqrt(delta)). In contrast, stable single-start gradient ascent from an attraction region of fixed positive mass reaches the same target in O(log(1/delta)) steps. The comparison can use the same fixed success probability without multiple starts.

Here delta is **excess error at fixed rank**, not the general absolute target epsilon→0 for reduced basis approximation. This solution family is represented exactly at rank 3, so the model does not resolve general reduced-dimension or offline-complexity questions.

### Numerical checks and search failures

The same model was tested with 2000 independent single-start L-BFGS-B runs. Fully converged local search has median final maximum error 0.417681, and 75.6% of runs reach E(a*)+1e-4. Of these runs, 1512 pass the nonglobal-local-peak check, 165 reach the global peak, and 323 fail the local-maximum check. Those 323 runs must remain visible; the 90% useful-basin probability for gradient flow is not the success probability of L-BFGS-B.

Actual pool greedy gives medians 0.630942 for M=2000 and 0.631536 for M=4000. As the pool grows, selection approaches the global peak and the final rank-2 error approaches 0.632456. In contrast, a best-snapshot oracle within the pool improves as M grows. The former reflects the difference between the greedy and final-space objectives; the latter measures candidate-coverage limitations. These effects are distinct.

## 7. Verification and reproducible files

Across the three dimensions, analytic gradients and centered differences agree to maximum relative error 3.1e-8. At low rank, Gram indicators and direct full-space residual calculations differ by less than 1.5e-13 relatively, and old/new pool evaluations agree. For all 81 final spaces, direct full-order error evaluation at each worst test parameter differs from the reduced contraction formula by at most 4.7e-11. Final X-orthogonality defects have infinity norm below 4e-13. These checks establish internal implementation consistency, not mesh convergence.

Main output files: `continuous_greedy_results.json` (aggregate and all seeds), `fin_d*/METHOD_SEED.json` (round histories), `*_basis.npz` (bases and selections), `local_*_witness.json` (nonglobality witnesses), `fin_isolated_timings.json` (sequential timings), and `fin_final_error_audit.json` (independent error checks). `CONTINUOUS_GREEDY_THEORY_PROGRESS.md` contains the derivations; `CONTINUOUS_GREEDY_REPRODUCTION.md` describes reproduction.

Large full-order test-solution caches are not bundled; saved test coordinates and the solver permit regeneration. The original experiment retained all 81 bases and selection sets. Shading in `continuous_greedy_summary.png/pdf` denotes the range of three seeds, not a confidence interval. Generated artifacts are excluded from the code repository.

## Appendix: final maximum errors for every seed

{{SEED_TABLE}}
'''
    for key,value in {'MAX_TABLE':metric_table('max'),'MEAN_TABLE':metric_table('uniform_mean'),'AUDIT_TABLE':'\n'.join(audit_rows),'TIME_TABLE':'\n'.join(time_rows),'SEED_TABLE':'\n'.join(seed_rows)}.items():
        report=report.replace('{{'+key+'}}',value)
    Path('CONTINUOUS_GREEDY_REPORT.md').write_text(report,encoding='utf-8')
    Path('environment_versions.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'threadpoolctl':threadpoolctl.__version__,'blas':threadpoolctl.threadpool_info()},indent=2))
    print('report written:',len(report),'characters')


if __name__ == "__main__":
    main()
