# -*- coding: utf-8 -*-
"""
Implementation of RealNVP.
"""
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
)
from .coupling import CouplingFlow
from ..transforms import AffineCouplingTransform


class RealNVP(CouplingFlow):
    """Implementation of Real Non-Volume Preserving Flows.

    See: https://arxiv.org/abs/1605.08803

    See :obj:`glasflow.flow.coupling.CouplingFlow` for the complete list of
    parameters and methods.

    Parameters
    ----------
    args :
        Positional arguments passed to the parent class.
    volume_preserving : bool, optional
        If True use additive transforms that preserve volume.
    kwargs :
        Keyword arguments passed to the parent class which in turn passes any
        kwargs to the transform class which will be either
        :py:obj:`nflows.transforms.coupling.AdditiveCouplingTransform` or
        :py:obj:`glasflow.transforms.coupling.AffineCouplingTransform`.
    """

    def __init__(self, *args, volume_preserving=False, **kwargs):
        if volume_preserving:
            transform_class = AdditiveCouplingTransform
        else:
            transform_class = AffineCouplingTransform
        super().__init__(transform_class, *args, **kwargs)
