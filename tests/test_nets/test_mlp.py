"""Tests for the MLP submodule"""
from glasflow.nets.mlp import MLP
import pytest
import torch


@pytest.mark.integration_test
def test_update_weights():
    """Check that the weights can be updated."""
    net = MLP(2, 1, [4])

    x = torch.randn(10, 2)
    y = torch.randn(10, 1)

    net.zero_grad()
    y_pred = net(x)
    loss = torch.mean((y - y_pred) ** 2.0)
    loss.backward()
