"""Multi-layer perceptrons"""

from typing import Callable, List, Optional, Union
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import get_torch_size


class MLP(nn.Module):
    """A standard multi-layer perceptron.

    Based on the implementation in nflows and nessai

    Parameters
    ----------
    input_shape
        Input shape.
    output_shape
        Output shape.
    n_neurons_per_layer
        Number of neurons in the hidden layers.
    activation_fn
        Activation function
    activate_output
        Whether to activate the output layer. If a bool is specified the same
        activation function is used. If a callable inputs is specified, it
        will be used for the activation.
    dropout_probability : float
        Amount of dropout to apply after the hidden layers.

    Raises
    ------
    ValueError
        If the number of neurons per layers is empty.
    TypeError
        If :code:`activate_ouput` is an invalid type.
    """

    def __init__(
        self,
        input_shape: Union[tuple, int],
        output_shape: Union[tuple, int],
        n_neurons_per_layer: List[int],
        activation_fn: Union[bool, Callable] = F.relu,
        activate_output: bool = False,
        dropout_probability: float = 0.0,
    ):
        super().__init__()

        self._input_shape = get_torch_size(input_shape)
        self._output_shape = get_torch_size(output_shape)
        self._n_neurons_per_layer = n_neurons_per_layer
        self._activation_fn = activation_fn
        self._activate_output = activate_output

        if len(n_neurons_per_layer) == 0:
            raise ValueError("List of n neurons per layer cannot be empty")

        self._input_layer = nn.Linear(
            np.prod(input_shape), n_neurons_per_layer[0]
        )
        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(
                    n_neurons_per_layer[:-1], n_neurons_per_layer[1:]
                )
            ]
        )
        self._dropout_layers = nn.ModuleList(
            nn.Dropout(dropout_probability)
            for _ in range(len(self._hidden_layers))
        )
        self._output_layer = nn.Linear(
            n_neurons_per_layer[-1], np.prod(output_shape)
        )

        if activate_output:
            self._activate_output = True
            if activate_output is True:
                self._output_activation = self._activation_fn
            elif callable(activate_output):
                self._output_activation = activate_output
            else:
                raise TypeError(
                    "activate_output must be a boolean or a callable."
                    f"Got input of type {type(activate_output)}."
                )
        else:
            self._activate_output = False

    def forward(
        self, inputs: torch.tensor, context: Optional[torch.tensor] = None
    ) -> torch.tensor:
        """Forward method that allows for kwargs such as context.

        Parameters
        ----------
        inputs
            Inputs to the MLP
        context
            Conditional inputs, must be None. Only implemented for
            compatibility.

        Raises
        ------
        ValueError
            If the context is not None.
        ValueError
            If the input shape is incorrect.
        """
        if context is not None:
            raise ValueError("MLP with conditional inputs is not implemented.")
        if inputs.shape[1:] != self._input_shape:
            raise ValueError(
                "Expected inputs of shape {}, got {}.".format(
                    self._input_shape, inputs.shape[1:]
                )
            )

        inputs = inputs.reshape(-1, np.prod(self._input_shape))
        outputs = self._input_layer(inputs)
        outputs = self._activation_fn(outputs)

        for hidden_layer, dropout in zip(
            self._hidden_layers, self._dropout_layers
        ):
            outputs = hidden_layer(outputs)
            outputs = self._activation_fn(outputs)
            outputs = dropout(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._output_activation(outputs)
        outputs = outputs.reshape(-1, *self._output_shape)

        return outputs
