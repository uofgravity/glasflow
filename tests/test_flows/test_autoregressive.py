from glasflow.flows.autoregressive import (
    MaskedAffineAutoregressiveFlow,
    MaskedPiecewiseCubicAutoregressiveAutoregressiveFlow,
    MaskedPiecewiseLinearAutoregressiveFlow,
    MaskedPiecewiseQuadraticAutoregressiveFlow,
    MaskedPiecewiseRationalQuadraticAutoregressiveFlow,
)
import pytest
import torch


@pytest.fixture(
    params=[
        MaskedAffineAutoregressiveFlow,
        MaskedPiecewiseCubicAutoregressiveAutoregressiveFlow,
        MaskedPiecewiseLinearAutoregressiveFlow,
        MaskedPiecewiseQuadraticAutoregressiveFlow,
        MaskedPiecewiseRationalQuadraticAutoregressiveFlow,
    ]
)
def FlowClass(request):
    return request.param


@pytest.mark.parametrize("use_random_permutations", [False, True])
@pytest.mark.parametrize("use_random_masks", [False, True])
def test_random_mask_and_perms(
    FlowClass, use_random_permutations, use_random_masks
):
    n = 10
    dims = 2
    flow = FlowClass(
        n_inputs=dims,
        n_transforms=4,
        use_random_masks=use_random_masks,
        use_random_permutations=use_random_permutations,
        use_residual_blocks=False if use_random_masks else True,
    )
    x = torch.rand(n, dims)
    z, logj = flow.forward(x)
    assert z.shape == (n, dims)
    assert len(logj) == n
