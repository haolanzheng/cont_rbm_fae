import numpy as np

from examples.hyperelasticity.fae_decoder_comparison import effective_rank


def test_effective_rank_reports_known_spectrum():
    codes = np.zeros((10, 4))
    codes[:, 0] = np.linspace(-1, 1, 10)
    result = effective_rank(codes)
    assert result["samples"] == 10
    assert result["participation_rank"] == 1
    assert result["entropy_rank"] == 1
    assert result["components_for_variance"] == {"0.9": 1, "0.95": 1, "0.99": 1}
