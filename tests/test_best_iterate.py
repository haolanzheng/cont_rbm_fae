"""The paper return rule is a maximum over the frozen search history, not z_T."""

from types import SimpleNamespace

import numpy as np
import pytest

from contgreedy._jax import jax, jnp
from contgreedy.domain import Box
from contgreedy.sampling import AdamConfig, adam_ascent
from contgreedy.sampling.host import BestObserved
from examples.baseline import sampling, run


def numpy_spectral_adam(H, z0, steps, lr):
    """Independent NumPy oracle with the paper's best-iterate return rule."""
    z = np.array(z0, dtype=float, copy=True)
    z /= np.linalg.norm(z)
    m, v = np.zeros_like(z), np.zeros_like(z)
    points, values = [z.copy()], [float(z @ H @ z)]
    for t in range(1, steps + 1):
        hz = H @ z
        g = 2 * (hz - (z @ hz) * z)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        z = z + lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        z /= np.linalg.norm(z)
        points.append(z.copy())
        values.append(float(z @ H @ z))
    return points[int(np.argmax(values))], values


@pytest.mark.parametrize(
    "steps,lr,location",
    [
        (0, 0.1, "initial"),
        (1, 0.1, "final"),
        (1, 1.0, "initial"),
        (10, 0.2, "interior"),
    ],
)
def test_adam_best_covers_entire_history(steps, lr, location):
    def objective(z):
        return -jnp.sum((z - 0.3) ** 2)

    result = adam_ascent(
        jax.value_and_grad(objective),
        jnp.zeros(1),
        Box().project,
        AdamConfig(steps=steps, learning_rate=lr),
    )
    values = np.asarray(result.values)
    index = int(np.argmax(values))
    if location == "initial":
        assert index == 0
    elif location == "final":
        assert index == steps
    else:
        assert 0 < index < steps
        assert values[-1] < values[index]
    np.testing.assert_allclose(objective(result.point), values[index], atol=1e-14)
    assert float(result.value) == values[index]
    assert result.value_evaluations == result.gradient_evaluations == steps + 1
    assert len(values) == steps + 1


def test_adam_ties_retain_initial_point():
    def constant(z):
        return jnp.array(1.0), jnp.ones_like(z)

    result = adam_ascent(constant, jnp.zeros(1), Box().project, AdamConfig(steps=3))
    np.testing.assert_array_equal(result.point, [0.0])


@pytest.mark.parametrize("steps,lr", [(0, 0.05), (1, 0.05), (50, 0.5)])
def test_spectral_adam_matches_independent_best_iterate_oracle(steps, lr):
    H = np.diag([1.0, 0.2, 0.05])
    z0 = np.array([0.3, 0.8, 0.5])
    point, history = sampling.adam_search(H, z0, steps, lr)
    expected, values = numpy_spectral_adam(H, z0, steps, lr)
    np.testing.assert_allclose(history, values, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(point, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(point @ H @ point, max(history), atol=1e-13)
    if steps == 50:
        assert 0 < np.argmax(history) < steps
        assert history[-1] < max(history)


def test_spectral_greedy_best_iterate_numpy_reference(monkeypatch):
    problem = run.Problem.build(6, 4, 0.25, 0.78)
    actual = run.run_greedy(problem, "adam", 2, 1000, steps=10, pool_size=16)
    monkeypatch.setattr(run, "adam_search", numpy_spectral_adam)
    reference = run.run_greedy(problem, "adam", 2, 1000, steps=10, pool_size=16)
    assert actual["search_return_policy"] == "best_iterate_0_to_T"
    for row, expected in zip(actual["rows"], reference["rows"]):
        assert row.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(row[key], expected[key], rtol=2e-8, atol=2e-10)
        if "inner_history" in row:
            np.testing.assert_allclose(
                row["selected_indicator"] ** 2,
                max(row["inner_history"]),
                atol=1e-12,
            )


def test_power_retains_best_even_for_nonmonotone_objective():
    # PSD is the paper case; indefinite H makes a regression against z_T decisive.
    H = np.diag([-2.0, 1.0])
    point, values = sampling.power_search(H, np.array([1.0, 1.0]), 3)
    assert values[-1] < values[0]
    np.testing.assert_allclose(point @ H @ point, max(values))


def test_host_tracker_endpoint_accounting():
    observed = BestObserved(lambda x: (float(1 - x[0] ** 2), -2 * x))
    observed.evaluate(np.array([0.5]))
    observed.evaluate(np.array([0.0]))
    observed.evaluate(np.array([0.8]))
    observed.ensure(np.array([0.5]))
    observed.ensure(np.array([0.8]))
    assert observed.calls == 3
    observed.ensure(np.array([0.9]))
    assert observed.calls == 4
    np.testing.assert_array_equal(observed.result()[0], [0.0])
    assert observed.result()[1] == 1.0


@pytest.mark.parametrize("case", ["haldane", "flow_reversal", "microwave"])
def test_scipy_selectors_do_not_return_worse_endpoint(monkeypatch, case):
    from examples.screening import haldane_continuous_greedy as haldane
    from examples.screening import flow_reversal_continuous_greedy as flow
    from examples.flow import natural_continuous_greedy_bench as microwave

    def minimize(fg, start, **kwargs):
        for x in [start, np.array([0.5]), np.array([0.9])]:
            fg(x)
        return SimpleNamespace(x=np.array([0.9]), nit=2, success=True, message="test")

    module = {"haldane": haldane, "flow_reversal": flow, "microwave": microwave}[case]
    monkeypatch.setattr(module.optimize, "minimize", minimize)
    model = SimpleNamespace(
        domain=(0.0, 1.0),
        indicator_and_grad=lambda data, x: (1 - (x - 0.5) ** 2, -2 * (x - 0.5)),
        to_q=lambda p: np.asarray(p),
        from_q=lambda q: np.asarray(q),
        fg_q=lambda V, q: (1 - (q[0] - 0.5) ** 2, -2 * (q - 0.5)),
    )
    if case == "microwave":
        point, value, info = module.box_bfgs(model, None, np.array([0.1]), 2)
        assert value == 1.0
        assert info["calls"] == 3
    elif case == "haldane":
        point, counts, info = module.select_continuous(model, None, 0.1, 2)
        assert counts.objective_calls == 3
        assert info["optimizer_end"] == 0.9
    else:
        point, calls, _, _, info = module.select_continuous(model, None, 0.1, 2)
        assert calls == 3
        assert info["optimizer_end"] == 0.9
    np.testing.assert_allclose(point, 0.5)
