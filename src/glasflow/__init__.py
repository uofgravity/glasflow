# -*- coding: utf-8 -*-
"""
glasflow
--------

Implementations of normalising flows in PyTorch based on nflows.

Code is hosted at: https://github.com/igr-ml/glasflow

nflows: https://github.com/bayesiains/nflows
"""
import os

USE_NFLOWS = os.environ.get("GLASFLOW_USE_NFLOWS", "False").lower() in [
    "true",
    "1",
]
if USE_NFLOWS:
    print("glasflow is using `nflows` instead of the included submodule")
    import sys

    try:
        import nflows
    except ModuleNotFoundError:
        raise RuntimeError(
            "nflows is not installed. Set the environment variable "
            "`GLASFLOW_USE_NFLOWS=False` to use the included fork of nflows."
        )
    sys.modules["glasflow.nflows"] = nflows
else:
    print("glasflow is using the included fork of `nflows`")

from .flows import (  # noqa
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
