from torch import nn, optim
from bonsai import Bonsai
from pruning.bonsai_prunners import WeightL2Prunner
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, sampler
import pytest

NUM_TRAIN = 49000
NUM_VAL = 1000


@pytest.fixture()
def train_dl():
    cifar10_train = CIFAR10('.datasets/CIfAR10', train=True, download=True, transform=ToTensor())
    yield DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))


@pytest.fixture()
def val_dl():
    cifar10_val = CIFAR10('.datasets/CIFAR10', train=True, download=True, transform=ToTensor())
    yield DataLoader(cifar10_val, batch_size=64,
                     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))


@pytest.fixture()
def test_dl():
    cifar10_test = CIFAR10('.datasets/CIFAR10', train=False, download=True, transform=ToTensor())
    yield DataLoader(cifar10_test, batch_size=64)


@pytest.fixture()
def bonsai_blank():
    cfg_path = "model_cfgs_for_tests/FCN-VGG16.cfg"
    bonsai = Bonsai(cfg_path)
    yield bonsai


@pytest.fixture()
def optimizer(bonsai_blank):
    adam = optim.Adam(bonsai_blank.model.parameters(), lr=1e-3)
    yield adam


@pytest.fixture()
def criterion():
    yield nn.CrossEntropyLoss()


def test_build_bonsai_with_no_prunner():
    cfg_path = "model_cfgs_for_tests/U-NET.cfg"
    _ = Bonsai(cfg_path)


def test_build_bonsai_with_weight_prunner():
    cfg_path = "model_cfgs_for_tests/U-NET.cfg"
    _ = Bonsai(cfg_path, WeightL2Prunner)


def test_bonsai_rank_method_with_weight_prunner():
    cfg_path = "model_cfgs_for_tests/U-NET.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner)
    bonsai.rank(None, None)


class TestBonsaiFinetune:

    def test_bonsai_finetune(self, bonsai_blank, train_dl, optimizer, criterion):
        bonsai_blank.finetune(train_dl, optimizer, criterion)
