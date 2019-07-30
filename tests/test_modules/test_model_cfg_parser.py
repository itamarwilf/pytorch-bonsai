import pytest
from bonsai.modules.model_cfg_parser import basic_model_cfg_parsing, validate_model_cfg
from bonsai.modules.errors import ModuleConfigError


@pytest.fixture()
def bad_config_1():
    bad_config1 = basic_model_cfg_parsing("tests/example_models_for_tests/configs/bad_config1.cfg")
    yield bad_config1
    print('something')


def test_linear_without_input_size(bad_config_1):
    with pytest.raises(ModuleConfigError):
        validate_model_cfg(bad_config_1)


@pytest.fixture()
def bad_config_2():
    bad_config2 = basic_model_cfg_parsing("tests/example_models_for_tests/configs/bad_config2.cfg")
    yield bad_config2


def test_linear_before_conv(bad_config_2):
    with pytest.raises(ModuleConfigError):
        validate_model_cfg(bad_config_2)
