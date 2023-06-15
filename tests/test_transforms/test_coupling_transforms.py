# -*- coding: utf-8 -*-
from glasflow.transforms.coupling import AffineCouplingTransform
import pytest
from unittest.mock import create_autospec, patch


@pytest.fixture
def affine_transform():
    return create_autospec(AffineCouplingTransform)


@patch(
    "glasflow.transforms.coupling.get_scale_activation",
    return_value="act_fn",
)
@patch("glasflow.nflows.transforms.coupling.AffineCouplingTransform.__init__")
def test_affine_coupling_init(mock_init, mock_get, affine_transform):
    mask = [1, -1]
    create_fn = object()

    AffineCouplingTransform.__init__(
        affine_transform,
        mask,
        create_fn,
        unconditional_transform=None,
        scale_activation="test",
    )

    mock_get.assert_called_once_with("test")
    mock_init.assert_called_once_with(
        mask,
        create_fn,
        unconditional_transform=None,
        scale_activation="act_fn",
    )
