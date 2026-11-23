"""Parameter-space candidate generation and search, independent of truth solves."""

from .adam import AdamConfig, SearchResult, adam_ascent
from .pool import pool_search
from .riemannian_lbfgs import (
    rlbfgs_ascent,
    sphere_exp,
    sphere_transport,
    tangent,
)

__all__ = [
    "AdamConfig",
    "SearchResult",
    "adam_ascent",
    "pool_search",
    "rlbfgs_ascent",
    "sphere_exp",
    "sphere_transport",
    "tangent",
]
