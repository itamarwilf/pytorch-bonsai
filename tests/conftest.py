import pytest
import torch
import os
from bonsai import Bonsai
from bonsai.pruning.bonsai_prunners import WeightL2Prunner, ActivationL2Prunner, TaylorExpansionPrunner
import requests

CHUNK_SIZE = 32768


def download_file_from_google_drive(_id, destination):
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': _id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': _id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


@pytest.fixture
def vgg19_weights_path():
    weight_path = "tests/example_models_for_tests/weights/vgg19_weights.pth"
    if not os.path.exists(weight_path):
        download_file_from_google_drive('1oC8R2AKx9Grl6QDAswZnrPn6LV-ixpgk', weight_path)
    yield weight_path


@pytest.fixture()
def vgg19_with_weights_prunner(vgg19_weights_path):
    cfg_path = "tests/example_models_for_tests/configs/VGG19.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner)
    bonsai.model.load_state_dict(torch.load(vgg19_weights_path))
    yield bonsai


@pytest.fixture()
def vgg19_with_activation_prunner(vgg19_weights_path):
    cfg_path = "tests/example_models_for_tests/configs/VGG19.cfg"
    bonsai = Bonsai(cfg_path, ActivationL2Prunner, normalize=True)
    bonsai.model.load_state_dict(torch.load(vgg19_weights_path))
    yield bonsai


@pytest.fixture()
def vgg19_with_grad_prunner(vgg19_weights_path):
    cfg_path = "tests/example_models_for_tests/configs/VGG19.cfg"
    bonsai = Bonsai(cfg_path, TaylorExpansionPrunner, normalize=True)
    bonsai.model.load_state_dict(torch.load(vgg19_weights_path))

    yield bonsai


@pytest.fixture()
def resnet18_with_weight_l2_prunner():
    cfg_path = "tests/example_models_for_tests/configs/resnet18.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner, normalize=True)
    # weight_path = "example_models_for_tests/weights/resnet18.pkl"
    # if os.path.exists(weight_path):
    #     bonsai.model.load_state_dict(torch.load(weight_path))
    yield bonsai


@pytest.fixture()
def fcn_vgg16_with_activation_prunner():
    cfg_path = "tests/example_models_for_tests/configs/FCN-VGG16.cfg"
    bonsai = Bonsai(cfg_path, ActivationL2Prunner, normalize=True)
    yield bonsai


@pytest.fixture()
def unet_with_weight_prunner():
    cfg_path = "tests/example_models_for_tests/configs/U-NET.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner, normalize=True)
    yield bonsai
