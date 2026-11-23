"""P1 mass/stiffness/Robin assembly for the unit-square GRF sampler."""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def matrices(n=64):
    x = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    xy = np.c_[xx.ravel(), yy.ravel()]
    tri = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + n + 1
            d = c + 1
            tri.extend([[a, b, d], [a, d, c]])
    tri = np.asarray(tri)
    ne = len(tri)
    area = 1 / (2 * n * n)
    T = np.concatenate([np.ones((ne, 3, 1)), xy[tri]], axis=2)
    grad = np.linalg.inv(T)[:, 1:, :].transpose(0, 2, 1)
    Me = area * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12
    Ke = area * np.einsum("eai,ebi->eab", grad, grad)
    ii = np.repeat(tri, 3, axis=1).ravel()
    jj = np.tile(tri, (1, 3)).ravel()
    M = sp.coo_matrix(
        (np.tile(Me, (ne, 1, 1)).ravel(), (ii, jj)), shape=(len(xy), len(xy))
    ).tocsc()
    K = sp.coo_matrix((Ke.ravel(), (ii, jj)), shape=M.shape).tocsc()
    R = sp.lil_matrix(M.shape)
    for ax in (0, 1):
        for val in (0, 1):
            ids = np.flatnonzero(abs(xy[:, ax] - val) < 1e-12)
            ids = ids[np.argsort(xy[ids, 1 - ax])]
            for a, b in zip(ids[:-1], ids[1:]):
                R[np.ix_([a, b], [a, b])] += np.array([[2, 1], [1, 2]]) / (6 * n)
    return xy, tri, M, K, R.tocsc(), la.cholesky(Me, lower=True)
