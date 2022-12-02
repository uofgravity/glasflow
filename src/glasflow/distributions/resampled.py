"""Distributions that include Learned Accept/Reject Sampling (LARS)."""
from typing import Callable, Union

from glasflow.nflows.distributions import Distribution
from glasflow.nflows.utils import torchutils
from glasflow.utils import get_torch_size
import numpy as np
import torch
from torch import nn


class ResampledGaussian(Distribution):
    """Gaussian distribution that includes LARS.

    For details see: https://arxiv.org/abs/2110.15828

    Based on the implementation here: \
        https://github.com/VincentStimper/resampled-base-flows

    Does not support conditional inputs.

    Parameters
    ----------
    shape
        Shape of the distribution
    acceptance_fn
        Function that computes the acceptance. Typically a neural network.
    eps
        Decay parameter for the exponential moving average used to update
        the estimate of Z.
    truncation
        Maximum number of rejection steps. Called T in the original paper.
    trainable
        Boolean to indicate if the mean and standard deviation of the
        distribution are learnable parameters.
    """

    def __init__(
        self,
        shape: Union[int, tuple],
        acceptance_fn: Callable,
        eps: float = 0.05,
        truncation: int = 100,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self._shape = get_torch_size(shape)
        self.truncation = truncation
        self.acceptance_fn = acceptance_fn
        self.eps = eps

        self.register_buffer("norm", torch.tensor(-1.0))
        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5 * np.prod(self._shape) * np.log(2 * np.pi),
                dtype=torch.float64,
            ),
        )
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self._shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self._shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self._shape))
            self.register_buffer("log_scale", torch.zeros(1, *self._shape))

    def _log_prob_gaussian(self, norm_inputs: torch.tensor) -> torch.tensor:
        """Base Gaussian log probability"""
        log_prob = (
            -0.5
            * torchutils.sum_except_batch(norm_inputs**2, num_batch_dims=1)
            - torchutils.sum_except_batch(self.log_scale, num_batch_dims=1)
            - self._log_z
        )
        return log_prob

    def _log_prob(
        self, inputs: torch.tensor, context: torch.tensor = None
    ) -> torch.tensor:
        """Log probability"""
        if context is not None:
            raise ValueError("Conditional inputs not supported")

        norm_inputs = (inputs - self.loc) / self.log_scale.exp()
        log_p_gaussian = self._log_prob_gaussian(norm_inputs)
        acc = self.acceptance_fn(norm_inputs)

        # Compute the normalisation
        if self.training or self.norm < 0.0:
            eps_ = torch.randn_like(inputs)
            norm_batch = torch.mean(self.acceptance_fn(eps_))
            if self.norm < 0.0:
                self.norm = norm_batch.detach()
            else:
                # Update the normalisation estimate
                # eps defines the weight between the current estimate
                # and the new estimated value
                self.norm = (
                    1 - self.eps
                ) * self.norm + self.eps * norm_batch.detach()
            # Why this?
            norm = norm_batch - norm_batch.detach() + self.norm
        else:
            norm = self.norm

        alpha = (1 - norm) ** (self.truncation - 1)
        return (
            torch.log((1 - alpha) * acc[:, 0] / norm + alpha) + log_p_gaussian
        )

    def _sample(
        self, num_samples: int, context: torch.tensor = None
    ) -> torch.tensor:
        if context is not None:
            raise ValueError("Conditional inputs not supported")

        device = self._log_z.device
        samples = torch.zeros(num_samples, *self._shape, device=device)

        t = 0
        s = 0
        n = 0
        norm_sum = 0

        for _ in range(self.truncation):
            samples_ = torch.randn(num_samples, *self._shape, device=device)
            acc = self.acceptance_fn(samples_)
            if self.training or self.norm < 0:
                norm_sum = norm_sum + acc.sum().detach()
                n += num_samples

            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or (t == (self.truncation - 1)):
                    samples[s, :] = samples_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break

        samples = self.loc + self.log_scale.exp() * samples
        return samples

    def estimate_normalisation_constant(
        self, n_samples: int = 1000, n_batches: int = 1
    ) -> None:
        """Estimate the normalisation constant via Monte Carlo sampling.

        Should be called once the training is complete.

        Parameters
        ----------
        n_samples
            Number of samples to draw in each batch.
        n_batches
            Number of batches to use.
        """
        with torch.no_grad():
            self.norm = self.norm * 0.0
            # Get dtype and device
            dtype = self.norm.dtype
            device = self.norm.device
            for _ in range(n_batches):
                eps = torch.randn(
                    n_samples, *self._shape, dtype=dtype, device=device
                )
                acc_ = self.acceptance_fn(eps)
                norm_batch = torch.mean(acc_)
                self.norm = self.norm + norm_batch / n_batches
