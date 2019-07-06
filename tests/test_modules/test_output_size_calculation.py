import pytest
from bonsai import Bonsai


@pytest.fixture
def unet_fixed_size():
    cfg_path = "model_cfgs_for_tests/U-NET_fixed_size.cfg"
    bonsai = Bonsai(cfg_path)
    yield bonsai


@pytest.fixture
def unet_arbitrary_size():
    cfg_path = "model_cfgs_for_tests/U-NET.cfg"
    bonsai = Bonsai(cfg_path)
    yield bonsai


def test_size_calculation_fully_conv(unet_arbitrary_size):
    model_output = unet_arbitrary_size.model.output_sizes[-1]
    assert model_output == (3, None, None)


def test_size_calculation_fixed_size(unet_fixed_size):
    model_output = unet_fixed_size.model.output_sizes[-1]
    assert model_output == (3, 512, 512)
