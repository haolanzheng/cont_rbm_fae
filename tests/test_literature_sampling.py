"""Checks for the published RB training-set sampling implementations."""

import numpy as np

from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.all_methods_comparison import run_standard
from examples.baseline.literature_sampling import (
    aega_scan,
    run,
    score_pool,
    smm_subset,
)


def test_aega_saturation_scan_finds_active_set_maximum():
    rng = np.random.default_rng(7)
    points = rng.standard_normal((80, 4))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    previous = np.diag([1.0, 0.8, 0.4, 0.2])
    current = np.diag([0.7, 0.5, 0.2, 0.1])
    saved = np.sqrt(score_pool(previous, points))
    index, value, remove, evaluations = aega_scan(current, points, saved, 1.0, 0.3)
    direct = score_pool(current, points)
    np.testing.assert_allclose(value, direct[index], atol=1e-14)
    np.testing.assert_allclose(value, direct.max(), atol=1e-14)
    assert 1 <= evaluations <= len(points)
    assert np.all(saved[remove] < 0.3)


def test_smm_subset_uses_indicator_levels_from_full_set():
    points = np.eye(6)
    scores = np.array([0.04, 0.09, 0.16, 0.36, 0.64, 1.0])
    subset = smm_subset(points, scores, 0.2, 4)
    chosen = np.argmax(subset, axis=1)
    np.testing.assert_array_equal(chosen, [0, 2, 3, 4])


def test_all_methods_account_for_training_and_indicator_evaluations():
    problem = FEMProblem.build(3, 4, element_order=3)
    for method in ("fresh_random", "aega", "smm"):
        result = run(
            problem,
            method,
            1000,
            pool_size=32,
            full_size=64,
            safety_size=64,
            target_rank=2,
        )
        assert result["indicator_evaluations"] > 0
        assert result["timings"]["total_training_seconds"] > 0
        assert result["rank"] <= problem.d
        assert len(result["curve"]) == result["rank"] + 1
        assert len(result["cumulative_training_seconds"]) == result["rank"] + 1
    oracle = run_standard(problem, "dense_eigh", 1000, target_rank=2)
    fixed = run_standard(
        problem,
        "fixed_random10000",
        1000,
        pool_size=32,
        target_rank=2,
    )
    assert oracle["exact_worst_error"] <= oracle["target_tolerance"]
    assert fixed["indicator_evaluations"] % 32 == 0
    assert len(fixed["curve"]) == fixed["rank"] + 1
