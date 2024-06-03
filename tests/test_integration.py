"""
Overall integration tests.
"""

import pytest
import torch


@pytest.mark.slow_integration_test
def test_flow_training(FlowClass):
    """General integration test for all flows"""
    n_inputs = 2
    flow = FlowClass(n_inputs=2, n_transforms=2)

    # Draw [0, 1) since some flows only support the unit interval
    x = torch.rand(10, n_inputs)

    opt = torch.optim.Adam(flow.parameters())

    for param in flow.parameters():
        param.grad = None
    loss = -flow.log_prob(x).mean()
    loss.backward()
    opt.step()
