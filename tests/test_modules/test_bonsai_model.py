import pytest
from modules.bonsai_model import BonsaiModel


@pytest.fixture
def pconv2d():
    cfg_path = "example_models_for tests/configs/pconv2d.cfg"
    bonsai = BonsaiModel(cfg_path, None)
    yield bonsai


def test_total_prunable_filters_single_conv_layer(pconv2d):
    assert pconv2d.total_prunable_filters() == 32


def test_total_prunable_filters_unet(unet_with_weight_prunner):
    assert unet_with_weight_prunner.model.total_prunable_filters() == 3424
