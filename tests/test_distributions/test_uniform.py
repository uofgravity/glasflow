"Tests for the Multivariate uniform distribution"
from unittest.mock import create_autospec

import pytest
import torch

from glasflow.distributions.uniform import MultivariateUniform


@pytest.fixture()
def dist():
    return create_autospec(MultivariateUniform)


@pytest.mark.parametrize(
    "low, high", [(0.0, 1.0), (torch.zeros(2), torch.ones(2))]
)
def test_init(low, high):
    """Assert different input types are valid"""
    dist = MultivariateUniform(low, high)
    assert dist._log_prob_value is not None


def test_init_invalid_shapes():
    """Assert an error is raised if the shapes are different"""
    with pytest.raises(
        ValueError, match=r"low and high are not the same shape"
    ):
        MultivariateUniform(torch.tensor(0), torch.tensor([1, 1]))


def test_init_invalid_bounds():
    """Assert an error is raised if low !< high"""
    with pytest.raises(
        ValueError, match=r"low has elements that are larger than high"
    ):
        MultivariateUniform(torch.tensor([1, 0]), torch.tensor([1, 1]))


@pytest.mark.parametrize(
    "inputs, target",
    [
        (torch.tensor([0.5, 0.5]), -torch.log(torch.tensor(4.0))),
        (torch.tensor([-1.0, -1.0]), -torch.tensor(torch.inf)),
        (torch.tensor([-1.0, 0.5]), -torch.tensor(torch.inf)),
        (
            torch.tensor([[-1.0, 0.5], [1.0, 1.0]]),
            torch.log(torch.tensor([0.0, 1.0 / 4.0])),
        ),
    ],
)
def test_log_prob(dist, inputs, target):
    """Assert log prob returns the correct value"""
    dist.low = torch.zeros(2)
    dist.high = 2 * torch.ones(2)
    dist._log_prob_value = -torch.log(torch.tensor(4.0))

    out = MultivariateUniform._log_prob(dist, inputs, None)
    assert torch.equal(out, target)


@pytest.mark.parametrize("num_samples", [1, 10])
def test_sample(dist, num_samples):
    """Assert the correct number of samples are returned and they're in the
    correct range.
    """
    n_dims = 2
    dist.low = torch.zeros(n_dims)
    dist.high = 2 * torch.ones(n_dims)
    dist._shape = dist.low.shape

    samples = MultivariateUniform._sample(dist, num_samples, None)

    assert samples.shape == (num_samples, n_dims)
    assert (samples < dist.high).all()
    assert (samples >= dist.low).all()


def test_context_error_log_prob(dist):
    """Assert an error is raised if log_prob is called and is not None."""
    with pytest.raises(
        NotImplementedError,
        match="Context is not supported by MultidimensionalUniform!",
    ):
        MultivariateUniform._log_prob(dist, torch.ones(1), torch.ones(1))


def test_context_error_sample(dist):
    """Assert an error is raised if sample is called and context is not None."""
    with pytest.raises(
        NotImplementedError,
        match="Context is not supported by MultidimensionalUniform!",
    ):
        MultivariateUniform._sample(dist, 10, torch.ones(1))
