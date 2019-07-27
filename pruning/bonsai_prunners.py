import torch
import numpy as np
from pruning.abstract_prunners import WeightBasedPrunner, ActivationBasedPrunner, GradBasedPrunner


class WeightL2Prunner(WeightBasedPrunner):

    @staticmethod
    def _compute_single_layer_ranks(module, *args, **kwargs):
        size = module.weights.size()
        weights = module.weights.contiguous().view(size[0], np.prod(size[1:]))
        return torch.sqrt(torch.sum(weights ** 2, dim=1))


class ActivationL2Prunner(ActivationBasedPrunner):

    @staticmethod
    def _compute_single_layer_ranks(module, *args, **kwargs):
        # activation map size is (batch_size x out_channels x width x height)
        activation = module.activation.detach().transpose(0, 1)
        size = activation.size()
        activation = activation.contiguous().view(size[0], np.prod(size[1:]))
        return torch.sqrt(torch.sum(activation ** 2, dim=1))


class TaylorExpansionPrunner(GradBasedPrunner):

    @staticmethod
    def _compute_single_layer_ranks(module, *args, **kwargs):
        # activation map and grad sizes are (batch_size X out_channels X width X height)
        activation = module.activation.detach().transpose(0, 1)
        grad = module.grad.detach().transpose(0, 1)
        ranks = activation * grad
        size = ranks.size()
        ranks = ranks.contiguous().view(size[0], np.prod(size[1:]))
        ranks = torch.mean(ranks, dim=1)

        return ranks
