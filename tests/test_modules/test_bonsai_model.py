import pytest
from modules.bonsai_model import BonsaiModel
import torch


class TestPConv2d:

    @pytest.fixture
    def pconv2d(self):
        cfg_path = "example_models_for tests/configs/pconv2d.cfg"
        bonsai = BonsaiModel(cfg_path, None)
        yield bonsai

    def test_print_pconv2d(self, pconv2d):
        print(pconv2d)

    def test_total_prunable_filters_single_conv_layer(self, pconv2d):
        assert pconv2d.total_prunable_filters() == 32

    def test_run_pconv2d(self, pconv2d):
        model_input = torch.rand(1, 4, 256, 256)
        model_output = pconv2d(model_input)
        assert model_output[0].size() == (1, 32, 256, 256)


class TestFCNVGG16:

    @pytest.fixture()
    def fcn_vgg16(self):
        cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
        model = BonsaiModel(cfg_path, None)
        yield model

    def test_module_list(self, fcn_vgg16):
        print(fcn_vgg16)

    def test_total_prunable_filters(self, fcn_vgg16):
        assert fcn_vgg16.total_prunable_filters() == 4224

    def test_run_fcn_vgg16(self, fcn_vgg16):
        model_input = torch.rand(1, 3, 32, 32)
        model_output = fcn_vgg16(model_input)
        assert model_output[0].size() == (1, 10)


class TestUNET:

    @pytest.fixture(scope="class")
    def unet(self):
        cfg_path = "example_models_for tests/configs/U-NET.cfg"
        model = BonsaiModel(cfg_path, None)
        yield model

    def test_print_module_list(self, unet):
        print(unet)

    def test_total_prunable_filters_unet(self, unet):
        assert unet.total_prunable_filters() == 3424

    def test_run_unet(self, unet):
        model_input = torch.rand(1, 4, 128, 128)
        model_output = unet(model_input)
        assert model_output[0].size() == (1, 3, 256, 256)


class TestVGG19:

    @pytest.fixture(scope="class")
    def vgg19(self):
        cfg_path = "example_models_for tests/configs/VGG19.cfg"
        model = BonsaiModel(cfg_path, None)
        yield model

    def test_print_module_list(self, vgg19):
        print(vgg19)

    # - TODO - calc actual number of prunable filters
    # def test_total_prunable_filters_unet(self, vgg19):
    #     assert vgg19.total_prunable_filters() == 3424

    def test_run_vgg19(self, vgg19):
        model_input = torch.rand(1, 3, 32, 32)
        model_output = vgg19(model_input)
        assert model_output[0].size() == (1, 10)
