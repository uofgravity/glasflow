# -*- coding: utf-8 -*-
"""
Tests for neural spline flows.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from glasflow.flows import CouplingNSF
from glasflow.distributions import MultivariateUniform


@pytest.mark.parametrize("num_bins", [4, 10])
def test_coupling_nsf_init(num_bins):
    """Test the initialise method"""
    CouplingNSF(2, 2, num_bins=num_bins)


def test_init_uniform_distribution():
    """Assert a uniform distribution is created and used"""
    expected_low = torch.zeros(2)
    expected_high = torch.ones(2)
    dist = MagicMock(spec=MultivariateUniform)

    with patch(
        "glasflow.distributions.MultivariateUniform", return_value=dist
    ) as mock_dist, patch(
        "glasflow.flows.nsf.CouplingFlow.__init__"
    ) as mock_init:
        CouplingNSF(
            n_inputs=2,
            n_transforms=2,
            distribution="uniform",
            tail_bound=10.0,
            tail_type="linear",
        )

    dist_kwargs = mock_dist.call_args[1]
    assert torch.equal(dist_kwargs["low"], expected_low)
    assert torch.equal(dist_kwargs["high"], expected_high)

    kwargs = mock_init.call_args[1]
    assert kwargs["tail_bound"] == 1.0
    assert kwargs["tails"] is None
    assert kwargs["distribution"] is dist
    assert kwargs["batch_norm_between_transforms"] is False


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
@pytest.mark.flaky(reruns=5)
def test_coupling_nsf_uniform():
    """Integration test to make sure core functions work as intended with the
    uniform latent space.
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
    expected_log_prob = log_j.numpy()

    np.testing.assert_array_almost_equal(x, x_inv)
    np.testing.assert_array_almost_equal(log_j, -log_j_inv)
    np.testing.assert_array_equal(log_prob, expected_log_prob)
    assert x_out.shape == (10, 2)
