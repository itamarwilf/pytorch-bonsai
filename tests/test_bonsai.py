from bonsai import Bonsai
from pruning.bonsai_prunners import WeightL2Prunner
import torch
import pytest


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
