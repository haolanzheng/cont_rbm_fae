"""Physical conductivity/Biot log chart and linear geometry chart."""

import numpy as np


def physical(z, lo, hi, log):
    z = np.asarray(z)
    a = (z + 1) * 0.5
    p = lo + (hi - lo) * a
    p[..., log] = np.exp(np.log(lo[log]) + np.log(hi[log] / lo[log]) * a[..., log])
    dp = np.broadcast_to((hi - lo) * 0.5, p.shape).copy()
    dp[..., log] = p[..., log] * np.log(hi[log] / lo[log]) * 0.5
    return p, dp
