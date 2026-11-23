"""Frozen regression for unchanged policies; best-iterate Adam has its own oracle.

The archived Adam fixture records the old final-iterate policy and is deliberately
not rewritten to pass the corrected algorithm.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from examples.baseline.hf import Problem
from examples.baseline.run import run_greedy
from examples.thermal_fin.hf import Fin
from examples.thermal_fin.rom import RB, ReferenceRB, PoolCache
from examples.thermal_fin.sampling import select

BASE = json.loads(
    (Path(__file__).parent / "fixtures/original_small_runs.json").read_text()
)


@pytest.mark.parametrize("method", ["power", "pool", "fresh_pool", "exact_indicator"])
def test_spectral_original_trajectory(method):
    result = run_greedy(
        Problem.build(6, 4, 0.25, 0.78), method, 2, 1000, steps=10, pool_size=16
    )
    expected = BASE["spectral"][method]
    assert len(result["rows"]) == len(expected["rows"])
    for actual, reference in zip(result["rows"], expected["rows"]):
        assert set(actual) == set(reference)
        for key in reference:
            np.testing.assert_allclose(
                actual[key], reference[key], rtol=2e-8, atol=2e-10, err_msg=key
            )


def test_fin_gradient_pool_and_original_selection():
    p = Fin(3, 4)
    rb = RB(p, 4)
    for z in [np.zeros(3), np.array([0.3, -0.5, 0.7]), np.array([-0.7, 0.4, -0.2])]:
        rb.add(p.fom(z))
    reference = BASE["thermal_fin"]
    assert (p.n, p.q) == (reference["dofs"], reference["q"])
    points = np.array(reference["z"])
    np.testing.assert_allclose(
        rb.values(points), reference["values"], rtol=1e-8, atol=1e-8
    )
    for z, g in zip(points, reference["gradients"]):
        value, gradient = rb.fg(z)
        np.testing.assert_allclose(gradient, g, rtol=2e-6, atol=2e-7)
        eps = 2e-5
        fd = [
            (rb.fg(z + eps * e, False) - rb.fg(z - eps * e, False)) / (2 * eps)
            for e in np.eye(3)
        ]
        np.testing.assert_allclose(gradient, fd, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(np.exp(value), rb.explicit(z), rtol=1e-7)
    np.testing.assert_allclose(
        PoolCache(rb, points).values(), rb.values(points), rtol=1e-8, atol=1e-8
    )
    point, info = select(rb, "adam100", points[0])
    np.testing.assert_allclose(point, reference["selection"], rtol=2e-5, atol=2e-6)
    assert info["fg_calls"] == 101
