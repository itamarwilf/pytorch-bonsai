import torch
from pruning.abstract_prunners import WeightBasedPrunner, ActivationBasedPrunner, GradBasedPrunner


class WeightL2Prunner(WeightBasedPrunner):

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        return torch.mean(torch.sqrt(module.weights ** 2), dim=(1, 2, 3))


class ActivationL2Prunner(ActivationBasedPrunner):

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        # activation map size is in_(channels X out_channels X width X height)
        return torch.mean(torch.sqrt(module.activation.detach() ** 2), dim=(0, 2, 3))


class TaylorExpansionPrunner(GradBasedPrunner):

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        activation = module.activation.detach()
        grad = module.grad.detach()
        # activation map and grad sizes are (in_channels X out_channels X width X height)
        ranks = torch.mean(activation * grad, dim=(0, 2, 3))
        # Normalize the rank by the filter dimensions
        # ranks /= activation.shape[1] * activation.shape[2] * activation.shape[3]

        return ranks
