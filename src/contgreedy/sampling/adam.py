"""JAX ascent with explicit historical algorithm semantics.

The objective and gradient are supplied together; gradients may be Euclidean,
tangent, or pulled back to another coordinate system. No silent conversion is
performed. Initialization RNGs stay outside this deterministic kernel.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple
from contgreedy._jax import jax, jnp


@dataclass(frozen=True)
class AdamConfig:
    steps: int = 100
    learning_rate: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    return_best: bool = True  # False is only for explicit legacy comparisons.

    def __post_init__(self):
        if self.steps < 0 or not isinstance(self.steps, int):
            raise ValueError("steps must be a nonnegative integer")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("Adam moments must lie in [0, 1)")


class SearchResult(NamedTuple):
    point: object
    value: object
    values: object
    value_evaluations: int
    gradient_evaluations: int


@partial(jax.jit, static_argnames=("value_grad", "project", "config"))
def adam_ascent(value_grad, z0, project, config=AdamConfig(), args=()):
    """Evaluate steps 0..T, reset moments, and return best or final iterate.

    Dynamic objective arrays belong in args to avoid recompiling closures.
    Evaluation counts count candidate points, not batched kernel launches.
    The paper policy is return_best=True: retain the earliest maximizer over
    all T+1 evaluated iterates. The caller supplies a feasible initial point.
    """
    z0 = jnp.asarray(z0, dtype=jnp.float64)
    q0, g0 = value_grad(z0, *args)
    initial = (z0, jnp.zeros_like(z0), jnp.zeros_like(z0), g0, z0, q0)

    def update(state, t):
        z, m, v, g, best_z, best_q = state
        m = config.beta1 * m + (1 - config.beta1) * g
        v = config.beta2 * v + (1 - config.beta2) * jnp.square(g)
        mh = m / (1 - config.beta1**t)
        vh = v / (1 - config.beta2**t)
        z = project(z + config.learning_rate * mh / (jnp.sqrt(vh) + config.epsilon))
        q, g = value_grad(z, *args)
        better = q > best_q
        best_z = jnp.where(better, z, best_z)
        best_q = jnp.where(better, q, best_q)
        return (z, m, v, g, best_z, best_q), q

    final, tail = jax.lax.scan(update, initial, jnp.arange(1, config.steps + 1))
    values = jnp.concatenate((jnp.atleast_1d(q0), tail))
    point = final[4] if config.return_best else final[0]
    value = final[5] if config.return_best else values[-1]
    return SearchResult(point, value, values, config.steps + 1, config.steps + 1)
