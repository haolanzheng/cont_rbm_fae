"""Strong spectral selectors and the costed GD update."""

import numpy as np
import pytest

from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.sampling import normalized_gd_search
from examples.baseline.strong_baselines import adaptive_cov_pool, run, select


def test_gd100_matches_independent_numpy_recurrence():
    H = np.array([[2.0, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 0.1]])
    initial = np.array([0.2, 0.8, -0.3])
    z = initial / np.linalg.norm(initial)
    values = [float(z @ H @ z)]
    points = [z.copy()]
    scale = np.linalg.norm(H)
    for _ in range(100):
        z = z + (H @ z) / scale
        z /= np.linalg.norm(z)
        points.append(z.copy())
        values.append(float(z @ H @ z))
    actual, history = normalized_gd_search(H, initial, 100, 1.0)
    np.testing.assert_allclose(history, values, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual, points[int(np.argmax(values))], atol=1e-12)


@pytest.mark.parametrize(
    "method",
    ["dense_eigh", "lanczos", "gd100", "lbfgs100"],
)
def test_strong_selectors_find_quadratic_maximum(method):
    H = np.diag([1.0, 0.3, 0.1, 0.02])
    z0 = np.array([0.3, 0.5, -0.2, 0.6])
    point, data = select(H, method, z0, None)
    np.testing.assert_allclose(np.linalg.norm(point), 1.0, atol=1e-12)
    assert float(point @ H @ point) > 0.99
    if method != "dense_eigh":
        assert data["h_actions"] > 0


def test_lbfgs_is_invariant_to_small_indicator_scale():
    shape = np.diag([1.0, 0.3, 0.1, 0.02])
    start = np.array([0.3, 0.5, -0.2, 0.6])
    large, _ = select(shape, "lbfgs100", start, None)
    small, _ = select(1e-10 * shape, "lbfgs100", start, None)
    assert float(large @ shape @ large) > 0.999
    assert float(small @ shape @ small) > 0.999


def test_fixed_and_resampled_pools_share_first_round():
    problem = FEMProblem.build(3, 4, element_order=3)
    fixed = run(problem, "fixed_random4096", 1000)
    fresh = run(problem, "fresh_random4096", 1000)
    np.testing.assert_allclose(fixed["curve"][1], fresh["curve"][1], atol=1e-12)
    np.testing.assert_allclose(
        fixed["rounds"][0]["rho"], fresh["rounds"][0]["rho"], atol=1e-12
    )


def test_adaptive_pool_uses_only_value_evaluations_and_preserves_best():
    H = np.diag([1.0, 0.3, 0.1, 0.02])
    point, data = adaptive_cov_pool(H, np.random.default_rng(1000))
    np.testing.assert_allclose(np.linalg.norm(point), 1.0, atol=1e-12)
    assert float(point @ H @ point) > 0.9
    assert data["h_actions"] == 4096
    assert data["stages"] == 4
