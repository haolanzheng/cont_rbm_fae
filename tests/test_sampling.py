import numpy as np
import pytest
from contgreedy._jax import jax, jnp
from contgreedy.domain import Box, Ball, Sphere
from contgreedy.sampling import AdamConfig, adam_ascent, pool_search
from contgreedy.sampling.laws import uniform_sphere, uniform_ball


def quadratic(z):
    return -jnp.sum((z - 0.3) ** 2)


def test_adam_zero_steps_and_budget():
    value_grad = jax.value_and_grad(quadratic)
    result = adam_ascent(
        value_grad, jnp.array([0.5, -0.2]), Box().project, AdamConfig(steps=0)
    )
    np.testing.assert_array_equal(result.point, [0.5, -0.2])
    assert result.gradient_evaluations == 1
    result = adam_ascent(
        value_grad, jnp.array([0.5, -0.2]), Box().project, AdamConfig(steps=15)
    )
    assert result.values.dtype == jnp.float64
    assert result.value == result.values.max()
    assert result.gradient_evaluations == 16
    assert result.value > quadratic(jnp.array([0.5, -0.2]))


def test_pool_and_domain_constraints():
    pts = jnp.array([[-1.0, -1.0], [0.3, 0.3], [1.0, 1.0]])
    z, value = pool_search(quadratic, pts)
    np.testing.assert_allclose(z, [0.3, 0.3])
    assert float(value) == 0
    np.testing.assert_allclose(Sphere(2).project(jnp.zeros(3)), [2, 0, 0])
    np.testing.assert_allclose(Ball(2).project(jnp.array([3.0, 4.0])), [1.2, 1.6])
    rng = np.random.default_rng(3)
    assert np.max(np.linalg.norm(uniform_ball(rng, 100, 4), axis=1)) <= 1
    np.testing.assert_allclose(np.linalg.norm(uniform_sphere(rng, 100, 4), axis=1), 1)


def test_invalid_adam_configuration():
    with pytest.raises(ValueError):
        AdamConfig(beta2=1)
