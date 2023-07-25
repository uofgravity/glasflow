# -*- coding: utf-8 -*-
import torch
from glasflow import USE_NFLOWS
from glasflow.flows import RealNVP
from glasflow.transforms.utils import SCALE_ACTIVATIONS

import pytest


@pytest.mark.parametrize("volume_preserving", [False, True])
def test_coupling_flow_init(volume_preserving):
    """Test the initialise method"""
    RealNVP(2, 2, volume_preserving=volume_preserving)


def test_real_nvp_defaults():
    flow = RealNVP(2, 2)
    x = torch.randn(10, 2)
    flow.forward(x)


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
def test_realnvp_wide_scale():
    """Test RealNVP with the 'wide' scaling method"""
    flow = RealNVP(2, 2, scaling_method="wide")
    x = torch.randn(10, 2)
    z, log_j = flow.forward(x)
    x_out, log_j_out = flow.inverse(z)

    assert torch.allclose(x, x_out)
    assert torch.allclose(log_j, -log_j_out)


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("scale_activation", list(SCALE_ACTIVATIONS.keys()))
def test_realnvp_scale_activation(scale_activation):
    flow = RealNVP(2, 2, scale_activation=scale_activation)
    x = torch.randn(10, 2)
    z, log_j = flow.forward(x)
    x_out, log_j_out = flow.inverse(z)
    assert torch.allclose(x, x_out)
    assert torch.allclose(log_j, -log_j_out)


@pytest.mark.skipif(
    USE_NFLOWS is False, reason="Test only applies when using nflows"
)
def test_affine_coupling_warning_nflows(caplog):
    """Assert a warning is printed if using `scale_activation` with nflows"""
    RealNVP(2, 2, scale_activation="log1")
    assert "Trying without `scale_activation`" in caplog.text
    assert (
        "Using affine coupling transform without `scale_activation`"
        in caplog.text
    )
