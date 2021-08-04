# -*- coding: utf-8 -*-
"""
Tests for neural spline flows.
"""
import numpy as np
import pytest
import torch

from glasflow.flows import CouplingNSF


@pytest.mark.parametrize("num_bins", [4, 10])
def test_coupling_nsf_init(num_bins):
    """Test the initialise method"""
    CouplingNSF(2, 2, num_bins=num_bins)


@pytest.mark.integration_test
def test_coupling_nsf_forward_inverse():
    """Make sure the flow is invertible"""
    x = torch.randn(10, 2)
    flow = CouplingNSF(2, 2)

    with torch.no_grad():
        x_prime, log_prob = flow.forward(x)
        x_out, log_prob_inv = flow.inverse(x_prime)

    np.testing.assert_array_almost_equal(x, x_out)
    np.testing.assert_array_almost_equal(log_prob, -log_prob_inv)
