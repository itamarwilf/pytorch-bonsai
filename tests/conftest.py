import pytest
import torch
import os
from bonsai import Bonsai
from pruning.bonsai_prunners import WeightL2Prunner, ActivationL2Prunner, TaylorExpansionPrunner


@pytest.fixture()
def vgg19_with_weights_prunner():
    cfg_path = "example_models_for tests/configs/VGG19.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner)
    weight_path = "example_models_for tests/weights/vgg19_weights.pth"
    if os.path.exists(weight_path):
        bonsai.model.load_state_dict(torch.load(weight_path))
    yield bonsai


@pytest.fixture()
def vgg19_with_activation_prunner():
    cfg_path = "example_models_for tests/configs/VGG19.cfg"
    bonsai = Bonsai(cfg_path, ActivationL2Prunner, normalize=True)
    yield bonsai


@pytest.fixture()
def vgg19_with_grad_prunner():
    cfg_path = "example_models_for tests/configs/VGG19.cfg"
    bonsai = Bonsai(cfg_path, TaylorExpansionPrunner, normalize=True)
    weight_path = "example_models_for tests/weights/vgg19_weights.pth"
    if os.path.exists(weight_path):
        bonsai.model.load_state_dict(torch.load(weight_path))
    yield bonsai


@pytest.fixture()
def fcn_vgg16_with_activation_prunner():
    cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
    bonsai = Bonsai(cfg_path, ActivationL2Prunner, normalize=True)
    yield bonsai


@pytest.fixture()
def unet_with_weight_prunner():
    cfg_path = "example_models_for tests/configs/U-NET.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner, normalize=True)
    yield bonsai
