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
@pytest.mark.flaky(reruns=5)
def test_coupling_nsf_forward_inverse():
    """Make sure the flow is invertible"""
    x = torch.randn(10, 2)
    flow = CouplingNSF(2, 2)

    with torch.no_grad():
        x_prime, log_prob = flow.forward(x)
        x_out, log_prob_inv = flow.inverse(x_prime)

    np.testing.assert_array_almost_equal(x, x_out)
    np.testing.assert_array_almost_equal(log_prob, -log_prob_inv)


@pytest.mark.integration_test
def test_coupling_nsf_uniform():
    """Integration test to mke sure core functions work as intended with the \
        unifrom latent space.
    """
    flow = CouplingNSF(2, 2, distribution="uniform")

    x = torch.rand(100, 2)

    with torch.no_grad():
        z, log_j = flow.forward(x)
        x_inv, log_j_inv = flow.inverse(z)
        log_prob = flow.log_prob(x)
        x_out = flow.sample(10)

    # n_dims * log(1 - 0) = 0
    # So just Jacobian
    expceted_log_prob = log_j.numpy()

    np.testing.assert_array_almost_equal(x, x_inv)
    np.testing.assert_array_almost_equal(log_j, -log_j_inv)
    np.testing.assert_array_equal(log_prob, expceted_log_prob)
    assert x_out.shape == (10, 2)
