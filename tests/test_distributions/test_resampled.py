"""Tests for the resampled distributions"""
from glasflow.distributions.resampled import ResampledGaussian
from glasflow.nets.mlp import MLP
import pytest
import torch


@pytest.mark.integration_test
@pytest.mark.parametrize("trainable", [False, True])
def test_resampled_gaussian_update_weights(trainable):
    """Assert the weights can be updated.

    Test with both fixed and trainable mean and variance.
    """
    dims = 2
    acc_fn = MLP(
        dims,
        1,
        [
            10,
        ],
        activate_output=torch.sigmoid,
    )
    dist = ResampledGaussian(dims, acc_fn, trainable=trainable)

    x = torch.randn(10, 2)

    dist.zero_grad()
    # Loss is normal flow loss
    loss = -torch.mean(dist.log_prob(x))
    loss.backward()
    dist.estimate_normalisation_constant()
