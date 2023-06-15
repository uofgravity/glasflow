from glasflow.transforms.utils import SCALE_ACTIVATIONS, get_scale_activation
import torch
import pytest


@pytest.mark.parametrize("name", list(SCALE_ACTIVATIONS.keys()))
def test_get_scale_activation_str(name):
    assert get_scale_activation(name) is SCALE_ACTIVATIONS[name]


def test_get_scale_activation_fn():
    def fn(x):
        return torch.sigmoid(x)

    assert get_scale_activation(fn) is fn


def test_get_scale_activation_invalid():
    with pytest.raises(ValueError, match=r"Unknown activation: .*"):
        get_scale_activation("invalid")
