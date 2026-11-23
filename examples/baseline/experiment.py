"""Compatibility facade for the original spectral experiment command."""

from .hf import Problem, elliptic_matrix, sine_loads, inv_sqrt_spd
from .rom import a_orthonormal_append, matrices, true_worst, equivalence_constants
from .sampling import normalize_rows, power_search, adam_search, pool_search
from .evaluate import estimate_high_set_probability, landscape_audit, aggregate
from .plot import make_figures
from .run import run_greedy, main

if __name__ == "__main__":
    main()
