"""CPU-only small regressions; do not allocate all login-node cores."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import pytest


@pytest.fixture(autouse=True)
def single_blas_thread():
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError:
        yield
        return

    with threadpool_limits(limits=1):
        yield
