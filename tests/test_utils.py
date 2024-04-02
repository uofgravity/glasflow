"""Test for the glasflow utilities"""

from glasflow.utils import get_torch_size
import pytest
import torch


def test_get_torch_size_int():
    """Assert an int is converted to a tuple"""
    assert get_torch_size(2) == torch.Size((2,))


@pytest.mark.parametrize("shape", [(2, 2), [2, 2], torch.tensor([2, 2])])
def test_get_torch_size_iterables(shape):
    """Assert iterables still work"""
    assert get_torch_size(shape) == torch.Size((2, 2))
