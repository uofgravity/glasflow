# -*- coding: utf-8 -*-
from glasflow.transforms.coupling import AffineCouplingTransform
import pytest
import torch
from unittest.mock import MagicMock, create_autospec, patch


@pytest.fixture
def affine_transform():
    transform = create_autospec(AffineCouplingTransform)
    transform.num_transform_features = 2
    transform._allowed_scaling_methods = ["nflows", "wide"]
    return transform


@pytest.mark.parametrize("scaling_method", ["nflows", "wide"])
@patch("glasflow.nflows.transforms.coupling.AffineCouplingTransform.__init__")
def test_affine_init(mock, affine_transform, scaling_method):
    """Assert the init method pass inputs to the parent class"""
    mask = [1, -1]
    fn = "function"
    ut = False
    AffineCouplingTransform.__init__(
        affine_transform,
        mask,
        fn,
        unconditional_transform=ut,
        scaling_method=scaling_method,
    )
    mock.assert_called_once_with(mask, fn, unconditional_transform=ut)
    assert affine_transform.scaling_method == scaling_method


@patch("glasflow.nflows.transforms.coupling.AffineCouplingTransform.__init__")
def test_affine_init_invalid_scaling(mock, affine_transform):
    """Assert an error is raised if an invalid scaling method is passed"""
    mask = [1, -1]
    fn = "function"
    ut = False
    with pytest.raises(ValueError) as excinfo:
        AffineCouplingTransform.__init__(
            affine_transform,
            mask,
            fn,
            unconditional_transform=ut,
            scaling_method="test",
        )
    mock.assert_called_once_with(mask, fn, unconditional_transform=ut)
    assert "Invalid scaling method" in str(excinfo.value)


def test_scale_and_shift_wide_transform(affine_transform):
    """Assert the range of returned values is [0, 3]."""
    params = torch.Tensor([[1, 2, 2, 6], [3, 4, -3, -100]])
    scale, shift = AffineCouplingTransform._scale_and_shift_wide(
        affine_transform, params
    )
    print(scale, shift)
    assert torch.equal(shift, torch.Tensor([[1, 2], [3, 4]]))
    assert torch.all(scale >= 0)
    assert scale[0, 1] == 3.0


def test_scale_and_shift_nflows(affine_transform):
    """Assert the correct scale and shift is callled for nflows"""
    params = "params"
    affine_transform.scaling_method = "nflows"
    with patch(
        "glasflow.nflows.transforms.coupling.AffineCouplingTransform._scale_and_shift"
    ) as mock:
        AffineCouplingTransform._scale_and_shift(affine_transform, params)
    mock.assert_called_once_with(params)


def test_scale_and_shift_wide(affine_transform):
    """Assert the correct scale and shift is callled for wide"""
    params = "params"
    affine_transform.scaling_method = "wide"
    affine_transform._scale_and_shift_wide = MagicMock()
    AffineCouplingTransform._scale_and_shift(affine_transform, params)
    affine_transform._scale_and_shift_wide.assert_called_once_with(params)


def test_scale_and_shift_invalid_method(affine_transform):
    """Assert an error is raised if an invalid scaling method is used."""
    params = "params"
    affine_transform.scaling_method = "test"
    with pytest.raises(RuntimeError) as excinfo:
        AffineCouplingTransform._scale_and_shift(affine_transform, params)
    assert "Unknown scaling method" in str(excinfo.value)
