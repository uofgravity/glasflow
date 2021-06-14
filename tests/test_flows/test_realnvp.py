# -*- coding: utf-8 -*-
from glasflow.flows import RealNVP

import pytest

@pytest.mark.parametrize('volume_preserving', [False, True])
@pytest.mark.integration_test
def test_coupling_flow_init(volume_preserving):
    """Test the initialise method"""
    flow = RealNVP(2, 2, volume_preserving=volume_preserving)