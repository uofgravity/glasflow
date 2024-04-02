"""
Multidimensional uniform distribution.
"""

from typing import Union

from glasflow.nflows.distributions import Distribution
import torch


class MultivariateUniform(Distribution):
    def __init__(
        self, low: Union[torch.Tensor, float], high: Union[torch.Tensor, float]
    ):
        """Multivariate uniform distribution defined on a box.

        Based on this implementation: \
            https://github.com/bayesiains/nflows/pull/17 but with fixes for
            CUDA support.

        Does not support conditional inputs.

        Parameters
        -----------
        low : Union[torch.Tensor, float]
            Lower range (inclusive).
        high : Union[torch.Tensor, float]
            Upper range (exclusive).
        """
        super().__init__()

        low, high = map(torch.as_tensor, [low, high])

        if low.shape != high.shape:
            raise ValueError("low and high are not the same shape")

        if not (low < high).byte().all():
            raise ValueError("low has elements that are larger than high")

        self._shape = low.shape
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.register_buffer(
            "_log_prob_value", -torch.sum(torch.log(high - low))
        )

    def _log_prob(self, inputs, context):
        if context is not None:
            raise NotImplementedError(
                "Context is not supported by MultidimensionalUniform!"
            )
        lb = self.low.le(inputs).type_as(self.low).prod(-1)
        ub = self.high.gt(inputs).type_as(self.low).prod(-1)
        return torch.log(lb.mul(ub)) + self._log_prob_value

    def _sample(self, num_samples, context):
        if context is not None:
            raise NotImplementedError(
                "Context is not supported by MultidimensionalUniform!"
            )
        low_expanded = self.low.expand(num_samples, *self._shape)
        high_expanded = self.high.expand(num_samples, *self._shape)
        samples = low_expanded + torch.rand(
            num_samples, *self._shape, device=self.low.device
        ) * (high_expanded - low_expanded)
        return samples
