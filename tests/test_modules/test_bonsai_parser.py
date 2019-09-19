import pytest
from bonsai.modules.bonsai_parser import bonsai_parser
from bonsai.modules.bonsai_model import BonsaiModel

import torch
from tests.resnet import ResNet18
from tests.vgg import VGG
from tests.u_net import UNet


class TestResNet18:
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

        assert len(original_state_dict) == len(restored_model_dict)

        for (rest_key, rest_value), (og_key, og_value) in zip(restored_model_dict.items(), original_state_dict.items()):
            if rest_value.size() == og_value.size():
                restored_model_dict[rest_key] = og_value
        restored_model.load_state_dict(restored_model_dict)

        # assert weight values are identical
        for (rest_key, rest_value), (og_key, og_value) in zip(restored_model.state_dict().items(), model.state_dict().items()):
            if rest_value.size() == og_value.size():
                assert torch.all(torch.eq(rest_value, og_value))

        res = restored_model(model_input)[0]
        og = model(model_input)

        print('bonsai restored model', restored_model)

        assert torch.all(torch.eq(og, res))

#
# class TestUNet:
#     def test_parse_unet(self):
#         model_input = torch.rand(1, 4, 128, 128)
#         model = UNet(4,4)
#         bonsai_parsed_model = bonsai_parser(model, model_input)
#
#     def test_saving_and_reading(self):
#         model_input = torch.rand(1, 4, 128, 128)
#         model = UNet(4,4)
#         bonsai_parsed_model = bonsai_parser(model, model_input)
#         cfg_dir = './../cfg_reservoir'
#         cfg_path = cfg_dir + '/parsed_unet.cfg'
#         bonsai_parsed_model.save_cfg(cfg_path)
#         restored_model = BonsaiModel(cfg_path, None)
#
#     def test_restoration_reliabilty(self):
#         model_input = torch.ones(1, 4, 128, 128)
#         model = UNet(4,4)
#
#         bonsai_parsed_model = bonsai_parser(model, model_input)
#         cfg_dir = './../cfg_reservoir'
#         cfg_path = cfg_dir + '/parsed_unet.cfg'
#         bonsai_parsed_model.save_cfg(cfg_path)
#         restored_model = BonsaiModel(cfg_path, None)
#         restored_model_dict = restored_model.state_dict()
#         original_state_dict = model.state_dict()
#
#         assert len(original_state_dict) == len(restored_model_dict)
#
#         for (rest_key, rest_value), (og_key, og_value) in zip(restored_model_dict.items(), original_state_dict.items()):
#             if rest_value.size() == og_value.size():
#                 restored_model_dict[rest_key] = og_value
#         restored_model.load_state_dict(restored_model_dict)
#
#         # assert weight values are identical
#         for (rest_key, rest_value), (og_key, og_value) in zip(restored_model.state_dict().items(), model.state_dict().items()):
#             if rest_value.size() == og_value.size():
#                 assert torch.all(torch.eq(rest_value, og_value))
#
#         res = restored_model(model_input)[0]
#         og = model(model_input)
#
#         print('bonsai restored model', restored_model)
#
#         assert torch.all(torch.eq(og, res))
