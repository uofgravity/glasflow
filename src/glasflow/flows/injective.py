"""Basic definitions for the flows module."""
from .base import Flow


class ConformalFlow(Flow):
    """Base class for all injective flow objects."""

    def reconstruct(self, inputs):
        mid_latent, _ = self._transform.forward(inputs)
        reconstruction, _ = self._transform.inverse(mid_latent)
        return reconstruction

    def reconstruct_and_log_prob(self, inputs, context=None):
        mid_latent, _ = self._transform.forward(inputs)
        reconstruction, log_conf_det = self._transform.inverse(mid_latent)

        log_pu = self._distribution.log_prob(mid_latent, context)
        log_prob = log_pu - log_conf_det
        return reconstruction, log_prob

    def log_prob(self, inputs, conditional=None):
        """
        Computes the log probability of the inputs samples by apply the
        transform.
        Does NOT need to specified by the user
        """
        return self._log_prob(inputs, conditional)

    def _log_prob(self, inputs, context=None):
        _, log_prob = self.reconstruct_and_log_prob(inputs, context)
        return log_prob
