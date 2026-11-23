"""Reproducible numerical sanity checks for the accompanying proof note.

These checks can detect algebra/implementation mistakes. They are not
proofs, PDE benchmarks, or evidence that the conservative bounds predict
100-step practical performance. Requires only NumPy.
"""
from itertools import product
import json
import numpy as np


def value_grad(x, order):
    q = np.sum(x * x, axis=-1) - 1.0
    height = 1.0 / (2 * order)
    f = height - q ** (2 * order) / (2 * order)
    g = -2.0 * q[:, None] ** (2 * order - 1) * x
    return f, g


def regional_checks():
    rng = np.random.default_rng(192609)
    nseed, nstep = 32, 250
    worst_gain = 0.0
    worst_segment = 0.0
    cases = steps = rescued = 0
    total_initially_low = 0
    for order, dim, beta1, beta2, eps in product(
        [1, 2], [2, 7], [0.0, 0.5, 0.9], [0.0, 0.25, 0.999], [0.1, 1.0]
    ):
        height = 1.0 / (2 * order)
        ell, rho = 0.02 * height, 0.02
        qmax = (2 * order * (height - ell)) ** (1.0 / (2 * order))
        rmax = np.sqrt(1 + qmax) + rho
        critical_r = 1.0 / np.sqrt(4 * order - 1)
        G = max(
            2 * critical_r * (1 - critical_r ** 2) ** (2 * order - 1),
            2 * rmax * (rmax ** 2 - 1) ** (2 * order - 1),
        )
        qabs = max(1.0, rmax ** 2 - 1)
        L = 2 * qabs ** (2 * order - 2) * ((4 * order - 1) * rmax ** 2 + 1)
        dmin, dmax = 1.0 / (G + eps), 1.0 / eps
        w = 4 * dmax ** 2 / (dmin * (1 - beta1))
        bounds = [rho / (dmax * G), dmin / (4 * L * dmax ** 2)]
        if beta1:
            bounds.append(dmin * (1 - beta1) / (np.sqrt(32) * beta1 * L * dmax ** 2))
        h = 0.9 * min(bounds)
        # All seeds have at most half of the peak value. Both sides of the
        # ridge are included; the inside has positive tangential curvature.
        init_value = rng.uniform(1.05 * ell, 0.5 * height, nseed)
        q = (2 * order * (height - init_value)) ** (1.0 / (2 * order))
        q *= np.where(rng.random(nseed) < 0.5, -1, 1)
        radius = np.sqrt(1 + q)
        direction = rng.normal(size=(nseed, dim))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        x = radius[:, None] * direction
        f0, _ = value_grad(x, order)
        total_initially_low += int(np.sum(f0 < 0.75 * height))
        mprev = np.zeros_like(x)
        vprev = np.zeros_like(x)
        best = f0.copy()
        for t in range(nstep):
            f, g = value_grad(x, order)
            m = beta1 * mprev + (1 - beta1) * g
            v = beta2 * vprev + (1 - beta2) * g ** 2
            mhat = m / (1 - beta1 ** (t + 1))
            vhat = v / (1 - beta2 ** (t + 1))
            direction = mhat / (np.sqrt(vhat) + eps)
            err = mhat - g
            phi = f - w * h * np.sum(err ** 2, axis=1)
            xnew = x + h * direction
            fnew, gnew = value_grad(xnew, order)
            mnew = beta1 * m + (1 - beta1) * gnew
            enew = mnew / (1 - beta1 ** (t + 2)) - gnew
            phinew = fnew - w * h * np.sum(enew ** 2, axis=1)
            rhs = h * dmin / 4 * np.sum(g ** 2, axis=1)
            rhs += w * h * (1 - beta1) / 4 * np.sum(err ** 2, axis=1)
            worst_gain = min(worst_gain, float(np.min(phinew - phi - rhs)))
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                fs, _ = value_grad(x + frac * h * direction, order)
                worst_segment = min(worst_segment, float(np.min(fs - phi)))
                assert np.all(fs >= ell - 2e-11), "A trial segment left its component"
            assert np.all(phinew >= f0 - 2e-11)
            assert np.all(phinew - phi >= rhs - 2e-11)
            best = np.maximum(best, fnew)
            x, mprev, vprev = xnew, m, v
            steps += nseed
        rescued += int(np.sum(best >= 0.75 * height))
        cases += 1
    return dict(
        cases=cases, trajectory_steps=steps,
        all_seeds_initially_below_75_percent=total_initially_low,
        seeds_reaching_75_percent_within_check_window=rescued,
        worst_lyapunov_margin=worst_gain,
        worst_segment_margin=worst_segment,
        note="Rescue count is a diagnostic on analytic ridges, not a PDE performance comparison.",
    )


def sharp_ratio_checks():
    count = 0
    max_normalized_ratio = 0.0
    for b1, b2, n in product(
        [0.0, 0.1, 0.5, 0.9, 0.99], [0.01, 0.25, 0.5, 0.9, 0.99, 0.999],
        [1, 2, 5, 10, 100, 1000]
    ):
        if b1 ** 2 >= b2:
            continue
        r = b1 ** 2 / b2
        ratio = (1 - b2 ** n) * (1 - r ** n) / (1 - b1 ** n) ** 2
        assert ratio <= 1 + 2e-13
        max_normalized_ratio = max(max_normalized_ratio, ratio)
        count += 1
    return dict(cases=count, max_ratio_to_uniform_bound=max_normalized_ratio)


def cycle_check():
    lam, eps, h = 2.0, 0.01, 0.1
    a = h / 2 - eps / lam
    x = a
    error = 0.0
    for _ in range(100):
        x = x - h * lam * x / (lam * abs(x) + eps)
        error = max(error, abs(abs(x) - a))
    assert error < 1e-14
    return dict(amplitude=a, max_cycle_error=error,
                two_step_linear_contraction=(1 - 4 * eps / (h * lam)) ** 2)


if __name__ == "__main__":
    print(json.dumps({
        "regional_lyapunov": regional_checks(),
        "moment_ratio": sharp_ratio_checks(),
        "fixed_step_obstruction": cycle_check(),
    }, indent=2))
