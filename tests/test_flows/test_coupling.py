# -*- coding: utf-8 -*-

from glasflow.flows.coupling import CouplingFlow
from glasflow.nflows.transforms.coupling import AffineCouplingTransform
import torch

import pytest


@pytest.mark.parametrize(
    "kwargs", [{}, dict(linear_transform="svd"), dict(n_conditional_inputs=1)]
)
def test_coupling_flow_init(kwargs):
    """Test the init method"""
    CouplingFlow(AffineCouplingTransform, 2, 2, **kwargs)


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
