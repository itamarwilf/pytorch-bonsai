from bonsai import Bonsai
import torch


def test_build_unet():
    cfg_path = "/home/itamar/Documents/pytorch-bonsai/example_configs/U-NET.cfg"
    _ = Bonsai(cfg_path)
    return


def test_run_unet():
    cfg_path = "/home/itamar/Documents/pytorch-bonsai/example_configs/U-NET.cfg"
    model = Bonsai(cfg_path)
    model_input = torch.rand(1, 4, 256, 256)
    model_output = model(model_input)
    assert model_output[0].size() == (1, 3, 512, 512)


def test_pconv2d():
    cfg_path = "model_cfgs_for_tests/pconv2d.cfg"
    model  = Bonsai(cfg_path)
    model_input = torch.rand(1, 4, 256, 256)
    model_output = model(model_input)
    assert model_output[0].size() == (1, 32, 256, 256)


def test_module_list():
    cfg_path = "model_cfgs_for_tests/pconv2d.cfg"
    model = Bonsai(cfg_path)
    print(model.module_list)
