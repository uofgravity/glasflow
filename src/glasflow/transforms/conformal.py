"""Components used in conformal embeddings."""

import geotorch
import torch
import torch.nn.functional as f
from torch import nn

from glasflow.nflows.transforms.base import Transform
from glasflow.nflows.transforms.normalization import ActNorm
import glasflow.nflows.utils.typechecks as check


class ConformalScaleShift(ActNorm):
    """
    Flow component of the type x |-> a*x + b, where a is a scalar.

    This is a conformal version of ActNorm, wherein a can be a vector.
    """

    def __init__(self, features, latent_dim):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__(features)

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.tensor(0.))
        self.shift = nn.Parameter(torch.zeros(features))
        self.latent_dim = latent_dim

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize(inputs)

        scale, shift = self._broadcastable_scale_shift(inputs)
        
        outputs = scale * inputs + shift

        batch_size = inputs.shape[0]
        logabsdet = self.latent_dim * torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / scale

        batch_size = inputs.shape[0]
        logabsdet = -self.latent_dim * torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0).mean()
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class SpecialConformal(Transform):
    """
    Flow component of the type x = (z- b ||z||^2)/(1 - 2bz + ||b||^2||z||^2).
    """

    def __init__(self, features, latent_dim):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.b = nn.Parameter(torch.zeros(features))
        self.latent_dim = latent_dim

    def forward(self, inputs, context=None):
        in_shape = inputs.shape
        x = inputs.view(in_shape[0], -1)
        x_norm_squared = torch.sum(x**2, dim=1, keepdim=True)
        outputs = ((inputs - self.b * x_norm_squared) /
             (1 - 2 * torch.sum(self.b * x, dim=1, keepdim=True) + torch.sum(self.b**2) * x_norm_squared))

        logabsdet = self.latent_dim * torch.log(
            torch.abs(1 - 2 * torch.sum(self.b * x, dim=1) + torch.sum(self.b ** 2) * x_norm_squared.flatten())
            )
        outputs = outputs.view(in_shape)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        in_shape = inputs.shape
        z = inputs.view(in_shape[0], -1)
        z_norm_squared = torch.sum(z**2, dim=1, keepdim=True)
        outputs = ((inputs + self.b * z_norm_squared) /
             (1 + 2 * torch.sum(self.b * z, dim=1, keepdim=True) + torch.sum(self.b**2) * z_norm_squared))
        logabsdet = - self.latent_dim * torch.log(
            torch.abs(1 + 2 * torch.sum(self.b * z, dim=1) + torch.sum(self.b ** 2) * z_norm_squared.flatten())
            )
        outputs = outputs.view(in_shape)
        return outputs, logabsdet


class Orthogonal(Transform):
    """
    Flow component of the type x = Rz.
    """
    def __init__(self, features, context=None):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")

        super().__init__()
        self.features = features

        self.matrix = nn.Parameter(torch.Tensor(features, features))
        geotorch.orthogonal(self, 'matrix')

    def forward(self, inputs, context=None):
        in_shape = inputs.shape
        x = inputs.view(in_shape[0], -1)
        outputs = x @ self.matrix
        logabsdet = inputs.new_zeros(in_shape[0])
        outputs = outputs.view(in_shape)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        in_shape = inputs.shape
        z = inputs.view(in_shape[0], -1)
        outputs = z @ self.matrix.t()
        logabsdet = inputs.new_zeros(in_shape[0])
        outputs = outputs.view(in_shape)
        return outputs, logabsdet


class Flatten(Transform):
    """Reshape the data into 1 dimension (not including the batch dimension)."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return inputs.reshape(batch_size, -1), logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return inputs.reshape(batch_size, *self.shape), logabsdet


class Pad(Transform):
    def __init__(self, x_channels, z_channels):
        super().__init__()
        self.x_channels = x_channels
        self.z_channels = z_channels

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return inputs[:,:self.z_channels,...], logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)

        if inputs.dim() == 4:
            outputs = f.pad(inputs, pad=(0, 0, 0, 0, 0, self.x_channels - self.z_channels))
        else:
            outputs = f.pad(inputs, pad=(0, self.x_channels - self.z_channels))
        return outputs, logabsdet
