from config import config
from torch import nn
from bonsai import Bonsai
from pruning.bonsai_prunners import WeightL2Prunner, ActivationL2Prunner, TaylorExpansionPrunner
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter
import torch
from modules.bonsai_parser import model_to_cfg_w_routing
from modules.model_cfg_parser import write_pruned_config
from u_net import UNet
import pytest
import os

NUM_TRAIN = 256
NUM_VAL = 128


@pytest.fixture
def logdir(tmpdir):
    print(tmpdir)
    logging_dict = config["logging"].get()
    logging_dict["logdir"] = tmpdir
    yield


@pytest.fixture
def out_path(tmpdir):
    pruning_dict = config["pruning"].get()
    pruning_dict["out_path"] = tmpdir
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
    cifar10_train = CIFAR10('.datasets/CIfAR10', train=True, download=True, transform=train_transform)
    yield DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))


@pytest.fixture()
def val_dl(test_transform):
    cifar10_val = CIFAR10('.datasets/CIfAR10', train=True, download=True, transform=test_transform)
    yield DataLoader(cifar10_val, batch_size=64,
                     sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))


@pytest.fixture()
def test_dl(test_transform):
    cifar10_test = CIFAR10('.datasets/CIfAR10', train=False, download=True, transform=test_transform)
    yield DataLoader(cifar10_test, batch_size=64)


@pytest.fixture()
def bonsai_blank():
    cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
    bonsai = Bonsai(cfg_path)
    yield bonsai


@pytest.fixture()
def criterion():
    yield nn.CrossEntropyLoss()


@pytest.fixture()
def writer(tmpdir):
    yield SummaryWriter(log_dir=tmpdir)


def test_build_bonsai_with_no_prunner():
    cfg_path = "example_models_for tests/configs/U-NET.cfg"
    _ = Bonsai(cfg_path)


def test_build_bonsai_with_weight_prunner():
    cfg_path = "example_models_for tests/configs/U-NET.cfg"
    _ = Bonsai(cfg_path, WeightL2Prunner)


def test_bonsai_rank_method_with_weight_prunner():
    cfg_path = "example_models_for tests/configs/U-NET.cfg"
    bonsai = Bonsai(cfg_path, WeightL2Prunner)
    bonsai.rank(None, None)


class TestEval:

    def test_eval_with_vgg19_weights(self, test_dl, criterion, writer):
        cfg_path = "example_models_for tests/configs/VGG19.cfg"
        bonsai = Bonsai(cfg_path)
        bonsai.model.load_state_dict(torch.load("example_models_for tests/weights/vgg19_weights.pth"))

        bonsai.eval(test_dl, criterion, None)


class TestBonsaiFinetune:

    def test_bonsai_finetune(self, bonsai_blank, train_dl, criterion, writer):
        bonsai_blank.finetune(train_dl, criterion, writer)

# download cifar10 val and test...


class TestBonsaiRank:

    def test_bonsai_rank_method_with_activation_prunner(self, val_dl, criterion):
        cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
        bonsai = Bonsai(cfg_path, ActivationL2Prunner)
        bonsai.rank(val_dl, criterion)

    def test_bonsai_rank_method_with_gradient_prunner(self, val_dl, criterion):
        cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
        bonsai = Bonsai(cfg_path, TaylorExpansionPrunner, normalize=True)
        bonsai.rank(val_dl, criterion)
        print("well")


class TestWriteRecipe:

    def test_write_recipe(self, val_dl, tmpdir):
        cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
        bonsai = Bonsai(cfg_path, WeightL2Prunner, normalize=True)
        bonsai.rank(val_dl, None)
        init_pruning_targets = bonsai.prunner.get_prunning_plan(99)
        write_pruned_config(bonsai.model.full_cfg, os.path.join(tmpdir, "testing.cfg"), init_pruning_targets)
        print("well")


class TestFullPrune:

    def test_run_pruning_fcn_vgg16(self, train_dl, val_dl, test_dl, criterion, logdir, out_path):
        cfg_path = "example_models_for tests/configs/FCN-VGG16.cfg"
        bonsai = Bonsai(cfg_path, TaylorExpansionPrunner, normalize=True)

        bonsai.run_pruning_loop(train_dl=train_dl, eval_dl=val_dl, criterion=criterion,
                                iterations=9)

    def test_run_pruning_vgg19(self, train_dl, val_dl, test_dl, criterion, logdir, out_path):
        cfg_path = "example_models_for tests/configs/VGG19.cfg"
        bonsai = Bonsai(cfg_path, ActivationL2Prunner, normalize=True)
        bonsai.model.load_state_dict(torch.load("example_models_for tests/weights/vgg19_weights.pth"))
        bonsai.run_pruning_loop(train_dl=train_dl, eval_dl=val_dl, criterion=criterion,
                                iterations=9)


class TestConfigurationFileParser:

    def test_file_parsing(self, train_dl, val_dl, test_dl, criterion):

        if __name__ == '__main__':
            u_in = torch.randn(1, 4, 128, 128)
            u_net = UNet(4, 4)
            u_out = u_net(u_in)

            cfg_mem = model_to_cfg_w_routing(u_net, u_in.size(), u_out)
            cfg_mem.summary()  # prints model cfg summary
            cfg_mem.save_cfg('../example_models/configs/unet_from_pytorch.cfg')
