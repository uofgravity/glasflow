# -*- coding: utf-8 -*-
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform
)
from .coupling import CouplingFlow


class RealNVP(CouplingFlow):
    """

    """
    def __init__(self, *args, volume_preserving=False, **kwargs):
        if volume_preserving:
            transform_class = AdditiveCouplingTransform
        else:
            transform_class = AffineCouplingTransform
        super().__init__(transform_class, *args, **kwargs)
