import torch
import numpy as np
from operator import itemgetter
from heapq import nsmallest
from modules.abstract_bonsai_classes import Prunable


class AbstractPrunner:

    def __init__(self, bonsai, normalize=False):
        self.bonsai = bonsai
        self.normalize = normalize

    def prunable_modules_iterator(self):
        """
        :return: iterator over module list filtered for prunable modules. holds tuples of (module index, module)
        """
        enumerator = enumerate(self.bonsai.model.module_list)
        return filter(lambda x: isinstance(x[1], Prunable), enumerator)

    def set_up(self):
        for _, module in self.prunable_modules_iterator():
            module.weights = module.get_weights()

    def reset(self):
        for _, module in self.prunable_modules_iterator():
            module.reset()

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        """
        attaches hooks for needed actions during forward / backward pass of prunable modules
        :return: None
        """
        raise NotImplementedError

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        """
        function that computes the rank of filters in conv2d and deconv2d layers for pruning.
        this function wil be called for prunable modules during prunning process
        :return:
        """
        raise NotImplementedError

    def compute_model_ranks(self):
        for _, module in self.prunable_modules_iterator():
            layer_current_ranks = self.compute_single_layer_ranks(module)
            module.ranking += layer_current_ranks

    @staticmethod
    def normalize_filter_ranks_per_layer(module: Prunable):
        """
        performs the layer-wise l2 normalization featured in:
        [1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference]
        :return: None
        """
        v = np.abs(module.ranking)
        v = v / np.sqrt(np.sum(v * v))
        module.ranking = v

    def normalize_ranks(self):
        for _, module in self.prunable_modules_iterator():
            self.normalize_filter_ranks_per_layer(module)

    def reset_ranks(self):
        for _, module in self.prunable_modules_iterator():
            module.reset()

    def lowest_ranking_filters(self, num):
        data = []
        for i, module in self.prunable_modules_iterator():
            for j, rank in enumerate(module.ranking):
                data.append((i, j, rank))
        return nsmallest(num, data, itemgetter(2))

    # TODO make sure filters are returned vectorized for slicing operations
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class WeightBasedPrunner(AbstractPrunner):

    def __init__(self, bonsai):
        super().__init__(bonsai)

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        pass

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError


class ActivationBasedPrunner(AbstractPrunner):

    def __init__(self, bonsai):
        super().__init__(bonsai)

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        module.activation = x

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError


class GradBasedPrunner(AbstractPrunner):

    def __init__(self, bonsai):
        super().__init__(bonsai)

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        module.activation = x
        x.register_hook(lambda grad: self._store_grad_in_module(module, grad))

    @staticmethod
    def _store_grad_in_module(module, grad):
        module.grad = grad

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError
