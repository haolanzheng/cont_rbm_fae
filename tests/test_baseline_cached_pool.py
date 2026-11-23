"""Check rank-one scores against independent residual evaluations."""
import numpy as np
from examples.baseline.common.cached_pool import CachedEnergyPool


def test_cached_pool_matches_energy_projection():
    rng=np.random.default_rng(721)
    d=18
    C=rng.normal(size=(d,d))
    Z=rng.uniform(-1,1,(257,d))
    cache=CachedEnergyPool(Z,C)
    Q=np.linalg.qr(rng.normal(size=(d,d)))[0]
    for n in range(d):
        E=C-Q[:,:n]@(Q[:,:n].T@C)
        direct=np.sum((Z@E.T)**2,axis=1)
        np.testing.assert_allclose(cache.scores,direct,atol=2e-11,rtol=2e-12)
        np.testing.assert_array_equal(cache.select(),Z[np.argmax(direct)])
        cache.update(Q[:,n])


def test_refresh_preserves_small_errors():
    rng=np.random.default_rng(51)
    d=128
    C=np.diag(.78**np.arange(d))
    Z=rng.uniform(-1,1,(300,d))
    cache=CachedEnergyPool(Z,C)
    for n in range(100):
        exact=np.sum((Z[:,n:]*np.diag(C)[n:])**2,axis=1)
        np.testing.assert_allclose(cache.scores,exact,rtol=2e-9,atol=1e-29)
        np.testing.assert_array_equal(cache.select(),Z[np.argmax(exact)])
        cache.update(np.eye(d)[:,n])
    assert cache.refresh_count>=3


def test_training_matches_direct_scores_and_skips_pool_matrix():
    from unittest.mock import patch
    from examples.baseline.common.run_consistent_fem import train
    from examples.baseline.consistent_fem import EnergyROM
    rng=np.random.default_rng(85)
    C=rng.normal(size=(8,8))
    for domain in ('sphere','ball','cube'):
        direct,ref,_,_=train(C,np.eye(8),domain,'pool256',1000,4,cache_pool=False,diagnostics=False)
        with patch.object(EnergyROM,'matrix',side_effect=AssertionError('Unused pool matrix assembled')):
            cached,actual,_,_=train(C,np.eye(8),domain,'pool256',1000,4,diagnostics=False)
        np.testing.assert_array_equal(actual['parameters'],ref['parameters'])
        np.testing.assert_allclose(cached.Q,direct.Q,atol=1e-14)
    K=np.diag(np.arange(1,9))
    direct,ref,_,_=train(C,K,'cube','pool256',1000,4,cache_pool=False,diagnostics=False)
    fallback,actual,_,_=train(C,K,'cube','pool256',1000,4,diagnostics=False)
    np.testing.assert_array_equal(actual['parameters'],ref['parameters'])
