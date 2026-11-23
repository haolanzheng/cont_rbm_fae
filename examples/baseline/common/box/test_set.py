"""Shared cube vertex tests for maximum absolute energy error."""
import numpy as np


def make_vertex_test_set(d, vertex_count=65536, seed=202610270):
    if d<2 or vertex_count<1:raise ValueError('Require d >= 2 and a positive count.')
    # Squared errors are even: fix z_1=+1 to avoid testing both z and -z.
    if d-1<=int(np.floor(np.log2(vertex_count))):
        count=2**(d-1);vertices=np.ones((count,d))
        vertices[:,1:]=2*((np.arange(count)[:,None]>>np.arange(d-1))&1)-1
        exhaustive=True
    else:
        rng=np.random.default_rng(seed+10000000+d);tails=np.empty((0,d-1))
        while len(tails)<vertex_count:
            tails=np.unique(np.vstack([tails,rng.choice([-1.,1.],(vertex_count,d-1))]),axis=0)
        rng.shuffle(tails);vertices=np.column_stack([np.ones(vertex_count),tails[:vertex_count]])
        exhaustive=False
    return vertices,exhaustive
