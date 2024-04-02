"""Tests for the resampled distributions"""

from glasflow.distributions.resampled import ResampledGaussian
from glasflow.nflows.distributions import StandardNormal
from glasflow.nets.mlp import MLP
import pytest
import torch


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


def test_log_prob_gaussian():
    """Assert the gaussian log-probability is correct"""
    shape = (2,)
    dist = ResampledGaussian(shape, lambda x: 1)
    ref_dist = StandardNormal(shape)
    x = ref_dist.sample(10)
    out = dist._log_prob_gaussian(x)
    expected = ref_dist.log_prob(x)
    torch.equal(out, expected)


def test_sample():
    """Assert samples are drawn with the correct shape"""
    dims = 2
    n = 10
    dist = ResampledGaussian(
        dims, lambda x: torch.ones(len(x), 1), trainable=False
    )
    out = dist._sample(n)
    assert out.shape == (n, dims)


def test_log_prob():
    """Test the log-prob method"""
    dims = 2
    n = 10
    x = torch.randn(n, 2)
    dist = ResampledGaussian(
        dims, lambda x: torch.ones(len(x), 1), trainable=False
    )
    out = dist.log_prob(x)
    assert len(out) == n
