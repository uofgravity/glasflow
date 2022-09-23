# -*- coding: utf-8 -*-

from glasflow.flows.coupling import CouplingFlow
from glasflow.nflows.transforms.coupling import AffineCouplingTransform
import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(linear_transform="svd"),
        dict(n_conditional_inputs=1),
        dict(mask=[[-1, 1], [1, -1]]),
    ],
)
def test_coupling_flow_init(kwargs):
    """Test the init method"""
    CouplingFlow(AffineCouplingTransform, 2, 2, **kwargs)


@pytest.mark.integration_test
def test_coupling_flow_w_mask():
    """Test the forward pass with a custom mask"""
    n_inputs = 2
    flow = CouplingFlow(
        AffineCouplingTransform, n_inputs, 2, mask=[[-1, 1], [1, -1]]
    )

    x = torch.randn(10, 2)

    z, log_jac = flow.forward(x)
    z = z.detach().numpy()
    log_jac = log_jac.detach().numpy()
    assert z.shape == (10, 2)
    assert log_jac.shape == (10,)

    log_prob = flow.log_prob(x)
    log_prob = log_prob.detach().numpy()
    assert log_prob.shape == (10,)

    x = flow.sample(10)
    x = x.detach().numpy()
    assert x.shape == (10, 2)


@pytest.mark.integration_test
def test_coupling_flow_forward_w_conditional():
    """Test the forward pass with a conditional input"""
    n_inputs = 2
    n_conditionals = 2
    flow = CouplingFlow(
        AffineCouplingTransform,
        n_inputs,
        2,
        n_conditional_inputs=n_conditionals,
    )

    x = torch.randn(10, 2)
    conditional = torch.randn(10, 2)

    z, log_prob = flow.forward(x, conditional=conditional)

    z = z.detach().numpy()
    log_prob = log_prob.detach().numpy()
    assert z.shape == (10, 2)
    assert log_prob.shape == (10,)


@pytest.mark.integration_test
def test_coupling_flow_sample_w_conditional():
    """Test sampling with a conditional input"""
    n = 10
    n_inputs = 2
    n_conditionals = 2
    flow = CouplingFlow(
        AffineCouplingTransform,
        n_inputs,
        2,
        n_conditional_inputs=n_conditionals,
    )
    conditional = torch.randn(n, 2)
    x = flow.sample(n, conditional=conditional)
    x = x.detach().numpy()
    assert x.shape == (n, 2)


@pytest.mark.integration_test
def test_coupling_flow_sample_and_log_prob_w_conditional():
    """Test sampling with a conditional input"""
    n = 10
    n_inputs = 2
    n_conditionals = 2
    flow = CouplingFlow(
        AffineCouplingTransform,
        n_inputs,
        2,
        n_conditional_inputs=n_conditionals,
    )

    conditional = torch.randn(n, 2)
    x, log_prob = flow.sample_and_log_prob(n, conditional=conditional)
    x = x.detach().numpy()
    log_prob = log_prob.detach().numpy()
    assert x.shape == (n, 2)
    assert log_prob.shape == (n,)


@pytest.mark.parametrize(
    "mask, expected",
    [
        (None, torch.tensor([[-1, 1], [1, -1], [-1, 1]]).int()),
        (
            torch.tensor([-1, 1]),
            torch.tensor([[-1, 1], [1, -1], [-1, 1]]).int(),
        ),
        ([-1, 1], torch.tensor([[-1, 1], [1, -1], [-1, 1]]).int()),
        (np.array([-1, 1]), torch.tensor([[-1, 1], [1, -1], [-1, 1]]).int()),
        (
            torch.tensor([[1, -1], [1, -1], [-1, 1]]),
            torch.tensor([[1, -1], [1, -1], [-1, 1]]).int(),
        ),
    ],
)
def test_validate_mask(mask, expected):
    """Assert the correct mask is returned"""
    out = CouplingFlow.validate_mask(mask, 2, 3)
    assert torch.equal(out, expected)


def test_validate_mask_invalid_length():
    """Assert a mask that is an invalid length raises an error"""
    with pytest.raises(ValueError) as excinfo:
        CouplingFlow.validate_mask([1, -1, 1], 2, 3)
    assert "does not match number of inputs" in str(excinfo.value)


def test_validate_mask_invalid_depth():
    """Assert a mask that is an invalid depth raises an error"""
    with pytest.raises(ValueError) as excinfo:
        CouplingFlow.validate_mask([[1, -1], [-1, 1]], 2, 3)
    assert "does not match number of transforms" in str(excinfo.value)
