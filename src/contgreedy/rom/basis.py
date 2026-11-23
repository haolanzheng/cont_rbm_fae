"""Metric-aware orthogonalization with an explicit dependent-snapshot outcome."""

from contgreedy._jax import jnp


def orthonormal_append(basis, snapshot, metric=None, tolerance=1e-12):
    """Two-pass modified Gram-Schmidt. metric maps a vector to X times it.

    Variable-rank orchestration stays on the host; dense contractions use JAX.
    SciPy sparse metric operators remain valid callbacks.
    """
    basis = jnp.asarray(basis)
    q = jnp.asarray(snapshot)
    apply = (lambda x: x) if metric is None else metric
    for _ in range(2):
        q = q - basis @ (basis.conj().T @ jnp.asarray(apply(q)))
    norm = jnp.sqrt(jnp.maximum(jnp.vdot(q, jnp.asarray(apply(q))).real, 0))
    if float(norm) < tolerance:
        return basis, float(norm)
    return jnp.column_stack((basis, q / norm)), float(norm)
