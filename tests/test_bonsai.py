import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from bonsai import Bonsai
from bonsai.config import config
from bonsai.modules.bonsai_parser import model_to_cfg_w_routing
from bonsai.modules.model_cfg_parser import write_pruned_config
from bonsai.pruning.bonsai_pruners import WeightL2Prunner
from u_net import UNet

NUM_TRAIN = 32
NUM_VAL = 16


@pytest.fixture
def logdir(tmpdir):
    config["logging"]["logdir"] = tmpdir
    yield


@pytest.fixture
def out_path(tmpdir):
    config["pruning"]["out_path"] = tmpdir
    yield


@pytest.fixture()
def train_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    yield transform_train


@pytest.fixture()
def test_transform():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    yield transform_test


@pytest.fixture()
def train_dl(train_transform):
    cifar10_train = CIFAR10('tests/.datasets/CIfAR10', train=True, download=True, transform=train_transform)
    yield DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))


@pytest.fixture()
def val_dl(test_transform):
    cifar10_val = CIFAR10('tests/.datasets/CIfAR10', train=True, download=True, transform=test_transform)
    yield DataLoader(cifar10_val, batch_size=64,
                     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))


@pytest.fixture()
def test_dl(test_transform):
    cifar10_test = CIFAR10('tests/.datasets/CIfAR10', train=False, download=True, transform=test_transform)
    yield DataLoader(cifar10_test, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))


@pytest.fixture()
def criterion():
    yield nn.CrossEntropyLoss()


@pytest.fixture()
def writer(tmpdir):
    yield SummaryWriter(log_dir=tmpdir)


def test_build_bonsai_with_no_prunner():
    cfg_path = "tests/example_models_for_tests/configs/U-NET.cfg"
    _ = Bonsai(cfg_path)


def test_build_bonsai_with_weight_prunner():
    cfg_path = "tests/example_models_for_tests/configs/U-NET.cfg"
    _ = Bonsai(cfg_path, WeightL2Prunner)


class TestEval:

    def test_eval_with_vgg19_weights(self, vgg19_with_weights_prunner, test_dl, criterion):
        vgg19_with_weights_prunner._eval(test_dl)


class TestBonsaiFinetune:

    def test_bonsai_finetune(self, vgg19_with_weights_prunner, train_dl, val_dl, criterion, out_path):
        vgg19_with_weights_prunner._finetune(train_dl, val_dl, criterion, 0)


class TestBonsaiRank:

    def test_bonsai_rank_method_with_weight_prunner(self, unet_with_weight_prunner):
        unet_with_weight_prunner._rank(None, None, 0)

    def test_bonsai_rank_method_with_activation_prunner(self, vgg19_with_activation_prunner, val_dl, criterion):
        vgg19_with_activation_prunner._rank(val_dl, criterion, 0)

    def test_bonsai_rank_method_with_gradient_prunner(self, vgg19_with_grad_prunner, val_dl, criterion):
        vgg19_with_grad_prunner._rank(val_dl, criterion, 0)


class TestWriteRecipe:

    def test_write_recipe(self, vgg19_with_weights_prunner, val_dl, tmpdir):
        vgg19_with_weights_prunner._rank(val_dl, None, 0)
        init_pruning_targets = vgg19_with_weights_prunner.prunner.get_prunning_plan(99)
        write_pruned_config(vgg19_with_weights_prunner.model.full_cfg, os.path.join(tmpdir, "testing.cfg"),
                            init_pruning_targets)


class TestFullPrune:

    def test_run_pruning_fcn_vgg16(self, fcn_vgg16_with_activation_prunner, train_dl, val_dl, test_dl, criterion,
                                   logdir, out_path):
        fcn_vgg16_with_activation_prunner.run_pruning(train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
                                                      criterion=criterion, iterations=3)

    def test_run_pruning_vgg19(self, vgg19_with_grad_prunner, train_dl, val_dl, test_dl, criterion, logdir, out_path):
        vgg19_with_grad_prunner.run_pruning(train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, criterion=criterion,
                                            iterations=9)

    def test_run_pruning_resnet18(self, resnet18_with_weight_l2_prunner, train_dl, val_dl, test_dl, criterion, logdir,
                                  out_path):
        resnet18_with_weight_l2_prunner.run_pruning(train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
                                                    criterion=criterion, prune_percent=0.05, iterations=5)


class TestConfigurationFileParser:

    def test_file_parsing(self, train_dl, val_dl, test_dl, criterion):

        if __name__ == '__main__':
            u_in = torch.rand(1, 4, 128, 128)
            u_net = UNet(4, 4)
            u_out = u_net(u_in)

            cfg_mem = model_to_cfg_w_routing(u_net, u_in.size(), u_out)
            cfg_mem.summary()  # prints model cfg summary
            cfg_mem.save_cfg('example_models/configs/unet_from_pytorch.cfg')
