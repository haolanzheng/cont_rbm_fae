import pytest

from examples.hyperelasticity.fae_scaling import needs_capacity


def test_capacity_decision_uses_validation_reference():
    assert needs_capacity(0.43)
    assert not needs_capacity(0.08)
    assert not needs_capacity(0.1, reference=0.1)
    with pytest.raises(ValueError):
        needs_capacity(float("nan"))
    with pytest.raises(ValueError):
        needs_capacity(float("inf"))
