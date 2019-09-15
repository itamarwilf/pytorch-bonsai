import pytest
from bonsai.modules.bonsai_parser import bonsai_parser
from bonsai.modules.bonsai_model import BonsaiModel

import torch
from tests.architectures.resnet import ResNet18
from tests.architectures.vgg import VGG
from tests.architectures.u_net import UNet


class TestResnet18:
    @pytest.fixture()
    def test_parse_resnet18(self):
        model_input = torch.rand(1, 3, 32, 32)
        model = ResNet18()
        bonsai_parsed_model = bonsai_parser(model, model_input)

    def test_saving_and_reading(self):
        model_input = torch.rand(1, 3, 32, 32)
        model = ResNet18()
        bonsai_parsed_model = bonsai_parser(model, model_input)
        cfg_dir = './../cfg_reservoir'
        cfg_path = cfg_dir + '/parsed_resnet18.cfg'
        bonsai_parsed_model.save_cfg(cfg_path)
        restored_model = BonsaiModel(cfg_path, None)

    def test_restoration_reliabilty(self):
        model_input = torch.rand(1, 3, 32, 32)
        model = ResNet18()
        bonsai_parsed_model = bonsai_parser(model, model_input)
        cfg_dir = './../cfg_reservoir'
        cfg_path = cfg_dir + '/parsed_resnet18.cfg'
        bonsai_parsed_model.save_cfg(cfg_path)
        restored_model = BonsaiModel(cfg_path, None)

        restored_model_dict = restored_model.state_dict()
        original_state_dict = model.state_dict()

        # print('original model', len(model.state_dict()))
        # print('restored model', len(restored_model.state_dict()))
        # [print('future',x) for x in  model_dict.keys()]
        # [print('past',x) for x in  loaded_state_dict.keys()]

        assert len(original_state_dict) == len(restored_model_dict)
        for model_key, loaded_value in zip(restored_model_dict.keys(), original_state_dict.values()):
            restored_model_dict[model_key] = loaded_value
        restored_model.load_state_dict(restored_model_dict)

        assert torch.all(torch.eq(model(model_input), restored_model(model_input)))


class TestUNet:
    @pytest.fixture()
    def test_parse_resnet18(self):
        model_input = torch.rand(1, 4, 128, 128)
        model = UNet(4,4)
        bonsai_parsed_model = bonsai_parser(model, model_input)

    def test_saving_and_reading(self):
        model_input = torch.rand(1, 4, 128, 128)
        model = UNet(4,4)
        bonsai_parsed_model = bonsai_parser(model, model_input)
        cfg_dir = './../cfg_reservoir'
        cfg_path = cfg_dir + '/parsed_unet.cfg'
        bonsai_parsed_model.save_cfg(cfg_path)
        restored_model = BonsaiModel(cfg_path, None)

    def test_restoration_reliabilty(self):
        model_input = torch.rand(1, 4, 128, 128)
        model = UNet(4,4)
        weights = model.state_dict()
        bonsai_parsed_model = bonsai_parser(model, model_input)
        cfg_dir = './../cfg_reservoir'
        cfg_path = cfg_dir + '/parsed_unet.cfg'
        bonsai_parsed_model.save_cfg(cfg_path)
        restored_model = BonsaiModel(cfg_path, None)
        restored_model.load_state_dict(weights)

        restored_model_dict = restored_model.state_dict()
        original_state_dict = model.state_dict()

        # print('original model', len(model.state_dict()))
        # print('restored model', len(restored_model.state_dict()))
        # [print('future',x) for x in  model_dict.keys()]
        # [print('past',x) for x in  loaded_state_dict.keys()]

        assert len(original_state_dict) == len(restored_model_dict)
        for model_key, loaded_value in zip(restored_model_dict.keys(), original_state_dict.values()):
            restored_model_dict[model_key] = loaded_value
        restored_model.load_state_dict(restored_model_dict)

        assert torch.all(torch.eq(model(model_input), restored_model(model_input)))
