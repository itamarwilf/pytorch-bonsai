import pytest
from modules.model_cfg_parser import basic_model_cfg_parsing, validate_model_cfg
from modules.receptive_field_calculation import calc_receptive_field
from modules.errors import ModuleConfigError


@pytest.fixture()
def unet_cfg():
    unet_cfg = basic_model_cfg_parsing("model_cfgs_for_tests/U-NET.cfg")
    yield unet_cfg


def test_unet_receptive_field(unet_cfg):
    calc_receptive_field(unet_cfg[1:])


@pytest.fixture()
def fcn_vgg16_cfg():
    fcn_vgg16_cfg = basic_model_cfg_parsing("model_cfgs_for_tests/FCN-VGG16.cfg")
    yield fcn_vgg16_cfg


def test_fcn_vgg16_receptive_field(fcn_vgg16_cfg):
    calc_receptive_field(fcn_vgg16_cfg[1:])


@pytest.fixture()
def bad_config_1():
    bad_config1 = basic_model_cfg_parsing("model_cfgs_for_tests/bad_config1.cfg")
    yield bad_config1


def test_linear_without_input_size(bad_config_1):
    with pytest.raises(ModuleConfigError):
        validate_model_cfg(bad_config_1)


@pytest.fixture()
def bad_config_2():
    bad_config2 = basic_model_cfg_parsing("model_cfgs_for_tests/bad_config2.cfg")
    yield bad_config2


def test_linear_before_conv(bad_config_2):
    with pytest.raises(ModuleConfigError):
        validate_model_cfg(bad_config_2)
