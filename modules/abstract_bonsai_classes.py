from typing import Dict, Any
import torch
from torch import nn


class BonsaiModule(nn.Module):

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super(BonsaiModule, self).__init__()
        self.bonsai_model = bonsai_model
        self.module_cfg = self._parse_module_cfg(module_cfg)

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        raise NotImplementedError

    def forward(self, layer_input):
        raise NotImplementedError


class Prunable(BonsaiModule):
    """
    interface of prunable Bonsai modules
    """

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)
        self.weights = None
        self.activation = None
        self.grad = None
        self.ranking = torch.zeros(self.module_cfg["out_channels"])

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        raise NotImplementedError

    def forward(self, layer_input):
        raise NotImplementedError

    def get_weights(self) -> torch.Tensor:
        """
        used to return weights with with output channels in the first dim. this is used for general implementation of
        ranking, and later each prunable module will handle the pruning based on the ranks regardless of his dim order
        :return: weights of prunabe module
        """
        raise NotImplementedError

    def prune_output(self):
        raise NotImplementedError

    def prune_input(self):
        raise NotImplementedError

    def reset(self):
        self.ranking = torch.zeros(self.module_cfg["out_channels"])

