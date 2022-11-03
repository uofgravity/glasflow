# -*- coding: utf-8 -*-
"""Test importing glasflow"""
from importlib import reload
import os

import glasflow
import pytest


@pytest.fixture(autouse=True)
def reload_glasflow():
    """Make sure nessai is reloaded after these tests"""
    original_version = glasflow.__version__
    use_nflows_default = os.environ.get("GLASFLOW_USE_NFLOWS", None)
    yield
    if use_nflows_default is None:
        os.environ.pop("GLASFLOW_USE_NFLOWS")
    else:
        os.environ["GLASFLOW_USE_NFLOWS"] = use_nflows_default
    assert os.environ.get("GLASFLOW_USE_NFLOWS") == use_nflows_default
    reload(glasflow)
    assert glasflow.__version__ == original_version


@pytest.mark.requires("nflows")
@pytest.mark.integration_test
def test_glasflow_use_external_nflows(caplog):
    """Assert in the import works with the external version of nflows"""
    os.environ["GLASFLOW_USE_NFLOWS"] = "True"
    reload(glasflow)
    assert "using an externally installed version" in str(caplog.text)


@pytest.mark.integration_test
def test_glasflow_use_internal_nflows(caplog):
    """Assert the import works with the internal version of nflows"""
    caplog.set_level("INFO")
    os.environ["GLASFLOW_USE_NFLOWS"] = "False"
    reload(glasflow)
    assert "using its own internal version" in str(caplog.text)
