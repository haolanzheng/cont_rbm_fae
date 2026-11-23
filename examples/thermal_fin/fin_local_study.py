"""Compatibility facade for historical thermal-fin commands and imports."""

from .hf import Fin
from .rom import RB, ReferenceRB, PoolCache
from .sampling import kkt, select
from .data import make_test
from .evaluate import uncached_values, local_audit, evaluate, validate
from .run import run, main

if __name__ == "__main__":
    main()
