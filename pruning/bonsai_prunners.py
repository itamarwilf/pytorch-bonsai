import torch
from pruning.abstract_prunners import WeightBasedPrunner, ActivationBasedPrunner, GradBasedPrunner


class WeightL2Prunner(WeightBasedPrunner):

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        return torch.mean(torch.sqrt(module.weights ** 2), dim=(1, 2, 3))


class ActivationL2Prunner(ActivationBasedPrunner):

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        return torch.mean(torch.sqrt(module.activation ** 2), dim=(1, 2, 3))


class TaylorExpansionPrunner(GradBasedPrunner):

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        activation = module.activation
        grad = module.grad

        ranks = torch.mean(activation * grad, dim=(1, 2, 3))
        # Normalize the rank by the filter dimensions
        # ranks /= activation.shape[1] * activation.shape[2] * activation.shape[3]

        return ranks


def get_candidates_to_prune(self, num_filters_to_prune):
    self.prunner.reset()

    self.train_epoch(rank_filters=True)

    self.prunner.normalize_ranks_per_layer()

    return self.prunner.get_prunning_plan(num_filters_to_prune)
