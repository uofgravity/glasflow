from .injective import ConformalFlow


from ..transforms import conformal
from .realnvp import RealNVP
from glasflow.nflows import transforms


class SphereCEFlow(ConformalFlow):
    def __init__(self, n_transforms: int, n_neurons: int = 32):
        n = 3
        m = 2

        conf_transform = transforms.CompositeTransform([
            conformal.ConformalScaleShift(n, m),
            conformal.Orthogonal(n),
            conformal.SpecialConformal(n, m),
            conformal.Pad(n, m),
        ])
        base_flow = RealNVP(
            n_inputs=m,
            n_transforms=n_transforms,
            n_neurons=n_neurons,
        )

        super().__init__(conf_transform, distribution=base_flow)