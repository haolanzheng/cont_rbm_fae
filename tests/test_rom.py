import numpy as np
from contgreedy.rom import orthonormal_append, greedy


def test_weighted_basis_and_dependent_round():
    X = np.diag([2.0, 3.0, 4.0])
    metric = lambda x: X @ np.asarray(x)
    basis = np.empty((3, 0))
    for u in [np.array([1.0, 2.0, 0.0]), np.array([1.0, 0.0, 1.0])]:
        basis, _ = orthonormal_append(basis, u, metric)
    np.testing.assert_allclose(basis.T @ X @ basis, np.eye(2), atol=1e-14)
    final, history = greedy(
        basis,
        rounds=2,
        select=lambda state, n: (np.array([1.0, 2.0, 0.0]), {}),
        truth=lambda z: z,
        enrich=lambda b, u: orthonormal_append(b, u, metric)[0],
        rank=lambda b: b.shape[1],
    )
    assert final.shape[1] == 2
    assert len(history) == 2
    assert sum(r["truth_calls"] for r in history) == 2
