from bonsai import Bonsai
from modules.bonsai_model import BonsaiModel
import torch
import pytest


def test_build_unet():
    cfg_path = "model_cfgs_for_tests/FCN-VGG16.cfg"
    _ = Bonsai(cfg_path)
    return


def test_run_unet():
    cfg_path = "model_cfgs_for_tests/FCN-VGG16.cfg"
    model = Bonsai(cfg_path)
    model_input = torch.rand(1, 3, 32, 32)
    model_output = model(model_input)
    assert model_output[0].size() == (1, 10)




def test_module_list():
    cfg_path = "model_cfgs_for_tests/FCN-VGG16.cfg"
    model = Bonsai(cfg_path)
    print(model)


def test_total_prunable_filters():
    cfg_path = "model_cfgs_for_tests/FCN-VGG16.cfg"
    bonsai = Bonsai(cfg_path)
    assert bonsai.model.total_prunable_filters() == 4224

