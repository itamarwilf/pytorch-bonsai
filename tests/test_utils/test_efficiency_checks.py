import pytest
from utils.efficiency_checks import speed_testing
from bonsai import Bonsai


@pytest.fixture()
def vgg16():
    cfg_path = "model_cfgs_for_tests/FCN-VGG16.cfg"
    bonsai = Bonsai(cfg_path)
    yield bonsai


@pytest.fixture()
def unet():
    cfg_path = "model_cfgs_for_tests/U-NET.cfg"
    bonsai = Bonsai(cfg_path)
    yield bonsai


class TestSpeedTesting:

    def test_speed_testing_vgg16(self, vgg16):
        speed_testing(vgg16.model, (1, 3, 32, 32))

    def test_speed_testing_unet(self, unet):
        speed_testing(unet.model, (1, 4, 256, 256))
