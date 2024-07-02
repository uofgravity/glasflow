"""
General configuration for tests.
"""

import glasflow
import glasflow.flows
import pytest


@pytest.fixture(
    params=[
        glasflow.flows.RealNVP,
        glasflow.flows.CouplingNSF,
        glasflow.flows.MaskedAffineAutoregressiveFlow,
        glasflow.flows.MaskedPiecewiseCubicAutoregressiveAutoregressiveFlow,
        glasflow.flows.MaskedPiecewiseLinearAutoregressiveFlow,
        glasflow.flows.MaskedPiecewiseQuadraticAutoregressiveFlow,
        glasflow.flows.MaskedPiecewiseRationalQuadraticAutoregressiveFlow,
    ]
)
def FlowClass(request):
    return request.param


def pytest_sessionstart():
    """Log which nflows backend is being using"""
    print(f"glasflow config: USE_NFLOWS={glasflow.USE_NFLOWS}")
