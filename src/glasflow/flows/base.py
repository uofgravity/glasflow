# -*- coding: utf-8 -*-
"""Base class for all normalising flows."""
from torch.nn import Module


class Flow(Module):
    """
    Base class for flow objects implemented according to the outline in nflows.

    Supports conditonal transforms but not conditional latent distributions.

    Parameters
    ----------
    transform : :obj: `nflows.transforms.Transform`
        Object that applys the transformation, must have`forward` and
        `inverse` methods. See nflows for more details.
    distribution : :obj: `nflows.distributions.Distribution`
        Object the serves as the base distribution used when sampling
        and computing the log probrability. Must have `log_prob` and
        `sample` methods. See nflows for details
    """

    def __init__(self, transform, distribution):
        super().__init__()

        for method in ["forward", "inverse"]:
            if not hasattr(transform, method):
                raise RuntimeError(
                    f"Transform does not have `{method}` method"
                )

        for method in ["log_prob", "sample"]:
            if not hasattr(distribution, method):
                raise RuntimeError(
                    f"Distribution does not have `{method}` method"
                )

        self._transform = transform
        self._distribution = distribution

    def forward(self, x, conditional=None):
        """
        Apply the forward transformation and return samples in the latent
        space and log |J|
        """
        return self._transform.forward(x, context=conditional)

    def inverse(self, z, conditional=None):
        """
        Apply the inverse transformation and return samples in the
        data space and log |J| (not log probability)
        """
        return self._transform.inverse(z, context=conditional)

    def sample(self, num_samples, conditional=None):
        """
        Produces N samples in the data space by drawing from the base
        distribution and the applying the inverse transform.
        Does NOT need to be specified by the user
        """
        noise = self._distribution.sample(num_samples)
        samples, _ = self._transform.inverse(noise, context=conditional)
        return samples

    def log_prob(self, inputs, conditional=None):
        """
        Computes the log probability of the inputs samples by apply the
        transform.
        Does NOT need to specified by the user
        """
        noise, logabsdet = self._transform(inputs, context=conditional)
        log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def base_distribution_log_prob(self, z):
        """
        Computes the log probability of samples in the latent for
        the base distribution in the flow.

        Does not accept condtional inputs

        Parameters
        ----------
        z : :obj:`torch.Tensor`
            Tensor of latent samples

        Returns
        -------
        :obj: `torch.Tensor`
            Tensor of log-probabilities
        """
        return self._distribution.log_prob(z)

    def forward_and_log_prob(self, x, conditional=None):
        """
        Apply the forward transformation and compute the log probability
        of each sample

        Conditional inputs are only used for the forward transform.

        Returns
        -------
        :obj:`torch.Tensor`
            Tensor of samples in the latent space
        :obj:`torch.Tensor`
            Tensor of log probabilities of the samples
        """
        z, log_J = self.forward(x, conditional=conditional)
        log_prob = self.base_distribution_log_prob(z)
        return z, log_prob + log_J

    def sample_and_log_prob(self, N, conditional=None):
        """
        Generates samples from the flow, together with their log probabilities
        in the data space log p(x) = log p(z) + log|J|.
        For flows, this is more efficient that calling `sample` and `log_prob`
        separately.

        Conditional inputs are only used for the inverse transform.
        """
        z, log_prob = self._distribution.sample_and_log_prob(N)
        samples, logabsdet = self._transform.inverse(z, context=conditional)
        return samples, log_prob - logabsdet
