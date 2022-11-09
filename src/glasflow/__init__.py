# -*- coding: utf-8 -*-
"""
glasflow
--------

Implementations of normalising flows in PyTorch based on nflows.

Code is hosted at: https://github.com/uofgravity/glasflow

nflows: https://github.com/bayesiains/nflows
"""
import importlib.util
import logging
import os
import pkgutil
import sys

logger = logging.getLogger(__name__)


def _import_submodules(module):
    """Recursively import all submodules from a module.

    Imports all of the submodules and registers them as
    glasflow.<module>.<submodule>

    Based on https://stackoverflow.com/a/25562415
    """
    for _, name, is_pkg in pkgutil.walk_packages(module.__path__):
        full_name = module.__name__ + "." + name
        submodule = importlib.import_module(full_name)
        sys.modules["glasflow." + full_name] = submodule
        if is_pkg:
            _import_submodules(submodule)


if "nflows" in sys.modules or importlib.util.find_spec("nflows"):
    NFLOWS_INSTALLED = True
else:
    NFLOWS_INSTALLED = False

USE_NFLOWS = os.environ.get("GLASFLOW_USE_NFLOWS", "False").lower() in [
    "true",
    "1",
]
if USE_NFLOWS:
    logger.warning(
        "glasflow is using an externally installed version of nflows"
    )
    if not NFLOWS_INSTALLED:
        raise RuntimeError(
            "nflows is not installed. Set the environment variable "
            "`GLASFLOW_USE_NFLOWS=False` to use the included fork of nflows."
        )
    # Register glasflow.nflows so it points to nflows
    import nflows

    sys.modules["glasflow.nflows"] = nflows
    # Register all submodules in nflows so glaflow.nflows.<submodule> points to
    # the nflows installation
    _import_submodules(nflows)
else:
    logger.info("glasflow is using its own internal version of nflows")

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
    if USE_NFLOWS:
        __version__ += "+nflows-ext"
    else:
        __version__ += "+nflows-int"
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = [
    "CouplingNSF",
    "RealNVP",
]
