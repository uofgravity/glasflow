# -*- coding: utf-8 -*-
"""
glasflow
--------

Implementations of normalising flows in PyTorch based on nflows.

Code is hosted at: https://github.com/igr-ml/glasflow

nflows: https://github.com/bayesiains/nflows
"""
from .flows import (
    CouplingNSF,
    RealNVP,
)

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # for Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = [
    "CouplingNSF",
    "RealNVP",
]
