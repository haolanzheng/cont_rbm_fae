"""Finite-step probability bounds for the first normalized-Poisson search.

Only V_1=span{the center snapshot} is covered. Units are scaled by the
largest remaining response weight, so w_1=1, w_i=4/(i+1)^2. The numerical
update is exactly the normalized clipped update used by NormalizedLoad.
The bound uses coordinate growth followed by a scalar tail-energy recursion.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import dawsn, erf

OUT = Path(__file__).resolve().parents[3] / 'runs/cube_candidate_screen_20261104'
ALPHA = .6
GAMMA = .9


def log_mgf_uniform_square(c):
    """log integral_0^1 exp(c*x*x) dx, stable for either sign of c."""
    c = np.asarray(c, dtype=float)
    out = np.empty_like(c)
    small = np.abs(c) < 1e-5
    # log M(c)=c/3+2c^2/45+8c^3/2835+O(c^4).
    out[small] = c[small]/3+2*c[small]**2/45+8*c[small]**3/2835
    positive = (c > 0) & ~small
    root = np.sqrt(c[positive])
    out[positive] = c[positive]+np.log(dawsn(root))-np.log(root)
    negative = (c < 0) & ~small
    root = np.sqrt(-c[negative])
    out[negative] = .5*np.log(np.pi)-np.log(2)+np.log(erf(root))-np.log(root)
    return out


def chernoff_upper(weights, threshold):
    """An evaluated Chernoff upper bound, not an exact tail probability.

    The formula is valid at every t>=0; numerical minimization only chooses
    a useful t. Computations use float64, not interval arithmetic.
    """
    weights = np.asarray(weights)
    objective = lambda t: float(np.sum(log_mgf_uniform_square(t*weights))-t*threshold)
    result = minimize_scalar(objective, bounds=(0, 1000), method='bounded')
    value = min(0., objective(result.x), objective(1000.))
    return float(np.exp(value)), float(result.x), value


def constants(d):
    w = 4 / np.arange(2, d+2, dtype=float)**2
    ell = np.linalg.norm(w)
    eta_min = ALPHA/np.sqrt(ell*ell+1)
    eta_max = ALPHA/ell
    bmax = 2*eta_max
    radius_constant = (2+eta_max)*bmax/np.log1p(bmax)
    tail_constant = 4*eta_min*(1-eta_max/4)
    return w, eta_min, eta_max, bmax, radius_constant, tail_constant


def tail_envelope(r, w):
    """Maximum sum_{i>=2} w_i*x_i with sum x_i=r, 0<=x_i<=1."""
    r = np.clip(r, 0, len(w)-1)
    k = np.floor(r).astype(int)
    prefix = np.r_[0., np.cumsum(w[1:])]
    next_weight = np.r_[w[1:], 0.]
    return prefix[k]+(r-k)*next_weight[k]


def bound_grid(d, max_budget=5120):
    w, emin, emax, bmax, crad, ctail = constants(d)
    a = np.arange(1, 100)/100
    s = np.r_[np.linspace((d-1)/3+.001, d-1-.001, 160), d-1.]
    tails = []
    for value in s:
        tails.append(0. if value >= d-1 else chernoff_upper(np.ones(d-1), value)[0])
    aa, ss = np.meshgrid(a, s, indexing='ij')
    probability = (1-aa)*(1-np.asarray(tails)[None, :])
    radius = np.minimum(d, 1+ss+crad*(np.log(1/aa)+np.log1p(bmax)))
    first_steps = np.ceil(np.log(1/aa)/np.log1p(emin/(1+radius))).astype(int)
    r = radius-1
    tail_steps = np.zeros_like(first_steps)
    target = 2*(GAMMA**(-2)-1)
    for _ in range(max_budget):
        active = r > target
        if not np.any(active):
            break
        rr = r[active]
        r[active] = rr-ctail*(rr-2*tail_envelope(rr, w))/(2+rr)**2
        tail_steps[active] += 1
    if np.any(r > target):
        raise RuntimeError('Increase maximum recursion budget')
    required = first_steps+tail_steps
    worst_r = float(d-1)
    worst_tail_steps = 0
    while worst_r > target:
        worst_r -= ctail*(worst_r-2*float(tail_envelope(np.asarray(worst_r), w)))/(2+worst_r)**2
        worst_tail_steps += 1
    requested = [80, 160, 320, 640, 1000, 1280, 2000, 2560, 5120]
    rows = []
    for budget in requested:
        certified = np.where(required <= budget, probability, -1.)
        index = np.unravel_index(np.argmax(certified), certified.shape)
        value = max(0., float(certified[index]))
        basic = 0. if budget <= worst_tail_steps else float(-np.expm1(-(budget-worst_tail_steps)*np.log1p(emin/(d+1))))
        row = dict(d=d, T=budget, lower_bound=max(value, basic),
                   event_lower_bound=value, basic_lower_bound=basic,
                   worst_tail_steps=worst_tail_steps)
        if value > 0:
            row.update(a=float(aa[index]), s=float(ss[index]),
                       first_steps=int(first_steps[index]), tail_steps=int(tail_steps[index]),
                       radius_bound=float(radius[index]))
        rows.append(row)
    threshold = GAMMA**2/2
    q0_upper, tilt, log_q0_upper = chernoff_upper(w-threshold, threshold)
    for row in rows:
        m = row['T']+1
        row['pool_same_count_upper'] = float(-np.expm1(m*np.log1p(-min(q0_upper, 1.)))) if q0_upper < 1 else 1.
        row['q0_upper'] = q0_upper
    return rows, dict(d=d, q0_upper=q0_upper, log_q0_upper=log_q0_upper, tilt=tilt,
                     eta_min_times_lambda1=emin, eta_max_times_lambda1=emax,
                     radius_constant=crad, tail_constant=ctail)


def update(z, w):
    den = 1+np.sum(z*z, axis=1)
    value = np.sum(w*z*z, axis=1)/den
    g = 2*z*(w-value[:, None])/den[:, None]/np.linalg.norm(w)
    return np.clip(z+ALPHA*g/np.sqrt(1+np.sum(g*g, axis=1, keepdims=True)), -1, 1)


def validate(d):
    w, emin, emax, bmax, crad, ctail = constants(d)
    rng = np.random.default_rng(1551+d)
    z = rng.uniform(-1, 1, (10000, d))
    old_j = np.sum(w*z*z, axis=1)/(1+np.sum(z*z, axis=1))
    zp = update(z, w)
    new_j = np.sum(w*zp*zp, axis=1)/(1+np.sum(zp*zp, axis=1))
    assert np.all(new_j >= old_j-1e-13)
    assert np.all(np.abs(zp[:, 0]) >= np.abs(z[:, 0])-1e-13)
    face = z.copy()
    face[:, 0] = 1
    fp = update(face, w)
    r = np.sum(face[:, 1:]**2, axis=1)
    rp = np.sum(fp[:, 1:]**2, axis=1)
    envelope = r-ctail*(r-2*tail_envelope(r, w))/(2+r)**2
    assert np.all(fp[:, 0] == 1)
    assert np.all(rp <= envelope+1e-12)
    return dict(d=d, cases=10000, minimum_J_gain=float((new_j-old_j).min()),
                minimum_tail_bound_margin=float((envelope-rp).min()))


def empirical(d, count=4096, max_steps=1280):
    w = constants(d)[0]
    rng = np.random.default_rng(202611050+d)
    initial = rng.uniform(-1, 1, (count, d))
    z = initial.copy()
    hit = np.full(count, -1, dtype=int)
    threshold = GAMMA**2/2
    for t in range(max_steps+1):
        active = np.flatnonzero(hit < 0)
        if not len(active):
            break
        za = z[active]
        values = np.sum(w*za*za, axis=1)/(1+np.sum(za*za, axis=1))
        reached = values >= threshold
        hit[active[reached]] = t
        active = active[~reached]
        if t < max_steps and len(active):
            z[active] = update(z[active], w)
    np.savez_compressed(OUT/f'normalized_load_probability_d{d}.npz', initial_parameters=initial, first_hitting_step=hit)
    return [dict(d=d, T=t, starts=count, empirical=float(np.mean((hit >= 0) & (hit <= t))))
            for t in (0, 80, 160, 320, 640, 1000, 1280)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = dict(scope='first search after the center snapshot only',
                    alpha=ALPHA, gamma=GAMMA, target='J>=gamma^2 max J',
                    initialization='independent uniform coordinates in [-1,1]',
                    dimensions=[16, 32, 64], device='CPU float64',
                    bounds='analytic inequalities evaluated in float64, no interval arithmetic',
                    pool='T+1 independent parameters, comparison by evaluation count only')
    (OUT/'normalized_load_probability_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    bounds, constants_rows, validation, measured = [], [], [], []
    for d in settings['dimensions']:
        rows, numbers = bound_grid(d)
        bounds.extend(rows)
        constants_rows.append(numbers)
        validation.append(validate(d))
        measured.extend(empirical(d))
        print(json.dumps(dict(d=d, bounds=rows, empirical=measured[-7:]), indent=2), flush=True)
        result = dict(bounds=bounds, constants=constants_rows, validation=validation, empirical=measured)
        (OUT/'normalized_load_probability_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('First-search probability analysis completed.', flush=True)


if __name__ == '__main__':
    main()
