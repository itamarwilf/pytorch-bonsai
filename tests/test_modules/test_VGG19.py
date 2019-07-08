from bonsai import Bonsai
import torch


def test_build_unet():
    cfg_path = "example_models_for tests/configs/VGG19.cfg"
    _ = Bonsai(cfg_path)
    return


def test_run_unet():
    cfg_path = "example_models_for tests/configs/VGG19.cfg"
    model = Bonsai(cfg_path)
    model_input = torch.rand(1, 3, 32, 32)
    model_output = model(model_input)
    assert model_output[0].size() == (1, 10)
