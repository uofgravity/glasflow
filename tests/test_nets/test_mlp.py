"""Tests for the MLP submodule"""

from unittest.mock import create_autospec
from glasflow.nets.mlp import MLP
import pytest
import torch


@pytest.fixture
def mlp():
    return create_autospec(MLP)


def test_mlp_init_hidden_layers(mlp):
    """Make sure the correct hidden layers are added"""
    MLP.__init__(mlp, 2, 1, [20, 10])

    assert mlp._input_layer.in_features == 2
    assert mlp._input_layer.out_features == 20
    # One layer will be the input layer
    assert len(mlp._hidden_layers) == 1
    assert mlp._output_layer.in_features == 10
    assert mlp._output_layer.out_features == 1


@pytest.mark.parametrize("activate_output", [False, True, torch.sigmoid])
def test_mlp_init_activate_output(mlp, activate_output):
    """Assert the different possible inputs are valid"""
    MLP.__init__(
        mlp,
        1,
        1,
        [
            1,
        ],
        activate_output=activate_output,
    )
    assert mlp._activate_output is bool(activate_output)


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
