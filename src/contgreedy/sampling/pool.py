"""Fixed and refreshed pools share selection; their lifetime belongs to the driver."""

from functools import partial
from contgreedy._jax import jax, jnp


@partial(jax.jit, static_argnames=("value",))
def pool_search(value, points, args=()):
    values = jax.vmap(value, in_axes=(0, *([None] * len(args))))(points, *args)
    index = jnp.argmax(values)
    return points[index], values[index]
