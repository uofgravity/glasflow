# -*- coding: utf-8 -*-
import torch
from glasflow.flows import RealNVP

import pytest


@pytest.mark.parametrize("volume_preserving", [False, True])
def test_coupling_flow_init(volume_preserving):
    """Test the initialise method"""
    RealNVP(2, 2, volume_preserving=volume_preserving)


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
