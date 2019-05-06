from typing import Dict, Any
import numpy as np
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
        self.ranking = np.zeros(self.module_cfg["out_channels"])

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        raise NotImplementedError

    def forward(self, layer_input):
        raise NotImplementedError

    # def __init__(self):
    #     self.prune = False
    #     self.activation = None

    def prune_output(self):
        raise NotImplementedError

    def prune_input(self):
        raise NotImplementedError

    def reset(self):
        self.ranking = np.zeros(self.module_cfg["out_channels"])

