"""Cartesian mesh candidates for the sphere and ball."""
import numpy as np


def tensor_candidates(d, h, domain):
    """Actual Cartesian tensor grid; sphere directions are its radial image."""
    intervals = round(2 / h)
    axis = np.linspace(-1.0, 1.0, intervals + 1)
    raw = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    radius = np.linalg.norm(raw, axis=1)
    if domain == "ball":
        candidates = raw[radius <= 1.0 + 1e-14]
    else:
        keep = radius > 0
        normalized = raw[keep] / radius[keep, None]
        # Preserve original float64 coordinates; rounding only identifies
        # rays that differ by a positive scalar within this exact rational grid.
        _, indices = np.unique(np.round(normalized, 13), axis=0, return_index=True)
        candidates = normalized[np.sort(indices)]
    return candidates, {"h": h, "tensor_count_before_filtering": len(raw),
                         "actual_candidate_count": len(candidates),
                         "axis_count": len(axis), "dimension": d,
                         "domain": domain,
                         "construction": ("Cartesian tensor grid restricted to closed unit ball"
                                          if domain == "ball" else
                                          "Radial image of nonzero Cartesian tensor grid; duplicate rays removed; not surface-area-uniform")}
