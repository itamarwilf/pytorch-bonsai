import torch
import numpy as np
import weakref
from typing import Iterator
from bonsai.modules.abstract_bonsai_classes import Prunable, Elementwise, BonsaiModule


class AbstractPrunner:

    def __init__(self, bonsai, normalize=False):
        self.bonsai = weakref.ref(bonsai)
        self.normalize = normalize
        self.pruning_residual = 0

    def get_bonsai(self):
        return self.bonsai()

    def prunable_modules_iterator(self) -> Iterator:
        """
        :return: iterator over module list filtered for prunable modules. holds tuples of (module index, module)
        """
        enumerator = enumerate(self.get_bonsai().model.module_list)
        return filter(lambda x: isinstance(x[1], Prunable), enumerator)

    def elementwise_modules_iterator(self):
        """
        Returns: iterator over module list filtered for elementwise modules. holds tuple of (module index, module)
        """
        enumerator = enumerate(self.get_bonsai().model.module_list)
        return filter(lambda x: isinstance(x[1], Elementwise), enumerator)

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
    def _compute_single_layer_ranks(module, *args, **kwargs):
        """
        function that computes the rank of filters in conv2d and deconv2d layers for pruning.
        this function wil be called for prunable modules during prunning process
        :return:
        """
        raise NotImplementedError

    def compute_model_ranks(self, engine=None):
        for _, module in self.prunable_modules_iterator():
            layer_current_ranks = self._compute_single_layer_ranks(module)
            module.ranking += layer_current_ranks.cpu()

    @staticmethod
    def _normalize_filter_ranks_per_layer(module: Prunable):
        """
        performs the layer-wise l2 normalization featured in:
        [1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference]
        :return: None
        """
        v = torch.abs(module.ranking)
        v = v / torch.sqrt(torch.sum(v * v))
        module.ranking = v

    def normalize_ranks(self):
        for _, module in self.prunable_modules_iterator():
            self._normalize_filter_ranks_per_layer(module)

    def _recursive_find_prunables_modules(self, base_module: BonsaiModule, module_idx: int) -> list:
        """
        performs search for prunable modules going into the elementwise module.
        it goes over layers going into elementwise module:
        1. if the module is prunable, stop searching
        2. if the module has "layers" in its config, apply this function on those modules
        3. if the module is non of the above, apply this function to the module before it

        this function will fail if it tries to perform pruning of both prunable and non prunable modules.
        Args:
            base_module: the module currently being checked
            module_idx: the modules index in the bonsai_model.module_list, used for finding the needed modules

        Returns: list of prunable modules going into the elementwise module
        """
        prunable_modules = []
        for layer_idx in base_module.module_cfg["layers"]:
            layer = self.get_bonsai().model.module_list[module_idx + layer_idx]
            if isinstance(layer, Prunable):
                prunable_modules += [layer]
            elif layer.module_cfg.get("layers"):
                prunable_modules += \
                    self._recursive_find_prunables_modules(layer, module_idx + layer_idx)
            else:
                new_idx = module_idx - 1
                prunable_modules += \
                    self._recursive_find_prunables_modules(self.get_bonsai().model.module_list[new_idx], new_idx)

        return prunable_modules

    def _equalize_single_elementwise(self, module: Elementwise, module_idx: int):
        """
        finds all prunable modules going into an elementwise module and equalize their pruning ranks
        Args:
            module: Elementwise module
            module_idx: the index in the bonsai_model.module list of the elementwise module

        Returns: None
        """
        prunable_modules = self._recursive_find_prunables_modules(module, module_idx)
        ranks = torch.stack([prunable_module.ranking for prunable_module in prunable_modules])
        # calculate algebraic mean of all the ranks going into that elementwise module
        new_ranks = ranks.mean(dim=0)
        for prunable_module in prunable_modules:
            prunable_module.ranking = new_ranks

    def equalize_elementwise(self):
        for module_idx, module in self.elementwise_modules_iterator():
            self._equalize_single_elementwise(module, module_idx)

    def lowest_ranking_filters(self, num_filters_to_prune):
        """
        Iterates over prunable modules for module index, filter index and filter ranks.
        In order to handle pruning of elementwise modules, pruning should be done simou

        Args:
            num_filters_to_prune:

        Returns:

        """
        print("pruning residual", self.pruning_residual)
        data = []
        for i, module in self.prunable_modules_iterator():
            for j, rank in enumerate(module.ranking):
                data.append((i, j, rank))
        data = sorted(data, key=lambda x: x[2])
        ranks = np.array([x[2] for x in data])
        desired_num_to_prune = num_filters_to_prune - self.pruning_residual
        max_prunable_rank = ranks[desired_num_to_prune]
        ranks_mask = ranks <= max_prunable_rank
        current_num_filters_to_prune = sum(ranks_mask)
        self.pruning_residual = current_num_filters_to_prune - desired_num_to_prune

        return data[:current_num_filters_to_prune]

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

        return filters_to_prune_per_layer

    def inverse_pruning_targets(self, pruning_targets):
        for i, module in self.prunable_modules_iterator():
            if i in pruning_targets.keys():
                pruning_targets[i] = [x for x in range(len(module.ranking)) if x not in pruning_targets[i]]
        return pruning_targets


class WeightBasedPrunner(AbstractPrunner):

    def __init__(self, bonsai, normalize=False):
        super().__init__(bonsai, normalize)

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        pass

    @staticmethod
    def _compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError


class ActivationBasedPrunner(AbstractPrunner):

    def __init__(self, bonsai, normalize=False):
        super().__init__(bonsai, normalize)

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        module.activation = x

    @staticmethod
    def _compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError


class GradBasedPrunner(AbstractPrunner):

    def __init__(self, bonsai, normalize=False):
        super().__init__(bonsai, normalize)

    def attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        module.activation = x
        x.register_hook(lambda grad: self._store_grad_in_module(module, grad))

    @staticmethod
    def _store_grad_in_module(module, grad):
        module.grad = grad

    @staticmethod
    def _compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError
