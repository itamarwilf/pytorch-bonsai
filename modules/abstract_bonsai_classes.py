from typing import Dict, Any

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


class Prunable:
    """
    interface of prunable Bonsai modules
    """
    def __init__(self):
        self.activation = None

    def prune_output(self):
        raise NotImplementedError

    def prune_input(self):
        raise NotImplementedError

    def _prune_params(self, grad):
        # TODO call bonsai model channel ranking func
        print(grad.size())
        # activation_index = len(self.activations) - self.grad_index - 1
        # activation = self.activations[activation_index]
        # values = \
        #     torch.sum((activation * grad), dim=0). \
        #         sum(dim=2).sum(dim=3)[0, :, 0, 0].data
        #
        # # Normalize the rank by the filter dimensions
        # values = \
        #     values / (activation.size(0) * activation.size(2) * activation.size(3))
        #
        # if activation_index not in self.filter_ranks:
        #     self.filter_ranks[activation_index] = \
        #         torch.FloatTensor(activation.size(1)).zero_().cuda()
        #
        # self.filter_ranks[activation_index] += values
        # self.grad_index += 1