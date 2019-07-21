import pytest
from modules.model_cfg_parser import basic_model_cfg_parsing
from modules.receptive_field_calculation import calc_receptive_field


@pytest.fixture()
def unet_cfg():
    unet_cfg = basic_model_cfg_parsing("example_models_for tests/configs/U-NET.cfg")
    yield unet_cfg


def test_unet_receptive_field(unet_cfg):
    calc_receptive_field(unet_cfg[1:])


@pytest.fixture()
def fcn_vgg16_cfg():
    fcn_vgg16_cfg = basic_model_cfg_parsing("example_models_for tests/configs/FCN-VGG16.cfg")
    yield fcn_vgg16_cfg


def test_fcn_vgg16_receptive_field(fcn_vgg16_cfg):
    calc_receptive_field(fcn_vgg16_cfg[1:])
