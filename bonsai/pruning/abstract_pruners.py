import torch
import numpy as np
import weakref
from typing import Iterator
from bonsai.modules.abstract_bonsai_classes import Prunable, Elementwise, BonsaiModule


class AbstractPruner:
    """
    Basic class for greedy neuron ranking for model pruning.
    It implements all needed methods for ranking and pruning all the prunable neurons in the model:
        1. sets up the hooks for layer rank calculation.
        2. calculates the prunable layer ranks.
        3. optional - normalize each layer ranks by their L2 norm, as seen in `Pruning Convolutional Neural Networks for Resource Efficient Inference <https://arxiv.org/abs/1611.06440>`_.
        4. equalizes the ranks of elementwise operations (such as residual connections) using algebraic mean.
        5. sorts all prunable neurons in the model by rank and returns which neurons should be pruned
        6. inverse pruning targets for implementation (which neurons to keep for each layer)
    """

    def __init__(self, bonsai, normalize=False):
        """
        Initializes the pruner.

        Args:
            bonsai (bonsai.main.Bonsai): The Bonsai object using the pruner. We hold a weak ref to it
            normalize (bool): whether to perform layer ranks normalization
        """
        self.bonsai = weakref.ref(bonsai)
        self.normalize = normalize
        self.pruning_residual = 0

    def _get_bonsai(self):
        return self.bonsai()

    def _prunable_modules_iterator(self) -> Iterator:
        """
        :return: iterator over module list filtered for prunable modules. holds tuples of (module index, module)
        """
        enumerator = enumerate(self._get_bonsai().model.module_list)
        return filter(lambda x: isinstance(x[1], Prunable), enumerator)

    def _elementwise_modules_iterator(self):
        """
        Returns: iterator over module list filtered for elementwise modules. holds tuple of (module index, module)
        """
        enumerator = enumerate(self._get_bonsai().model.module_list)
        return filter(lambda x: isinstance(x[1], Elementwise), enumerator)

    def set_up(self):
        """
        sets up the pruner for rank calculation by getting layers weights in desired shape (output x input x ...)
        """
        for _, module in self._prunable_modules_iterator():
            module.weights = module.get_weights().cpu()

    def reset(self):
        """
        resets all prunable layer ranks
        """
        for _, module in self._prunable_modules_iterator():
            module.reset()

    def _attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        """
        attaches hooks for needed actions during forward / backward pass of prunable modules
        :return: None
        """
        raise NotImplementedError

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        """
        method for calculating the rank of a single layer based on weights and activations and gradients if calculated.

        Args:
            module (bonsai.modules.Prunable): the target module for ranking
            *args: arguments needed for computation
            **kwargs: keyword arguments needed for computation

        Returns (torch.Tensor): calculated ranks in the shape (num_of_prunable_parameters,)
        """
        raise NotImplementedError

    def compute_model_ranks(self, _=None):
        for _, module in self._prunable_modules_iterator():
            layer_current_ranks = self.compute_single_layer_ranks(module)
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
        for _, module in self._prunable_modules_iterator():
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

        if base_module.module_cfg.get("layers"):
            for layer_idx in base_module.module_cfg["layers"]:
                layer = self._get_bonsai().model.module_list[module_idx + layer_idx]
                prunable_modules += \
                    self._recursive_find_prunables_modules(layer, module_idx + layer_idx)
        else:
            if isinstance(base_module, Prunable):
                prunable_modules += [base_module]
            else:
                new_idx = module_idx - 1
                prunable_modules += \
                    self._recursive_find_prunables_modules(self._get_bonsai().model.module_list[new_idx], new_idx)

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
        for module_idx, module in self._elementwise_modules_iterator():
            self._equalize_single_elementwise(module, module_idx)

    def _lowest_ranking_filters(self, num_filters_to_prune):
        """
        Iterates over prunable modules for module index, filter index and filter ranks.
        In order to handle pruning of elementwise modules, pruning should be done simou

        Args:
            num_filters_to_prune:

        Returns:

        """
        print("pruning residual", self.pruning_residual)
        data = []
        for i, module in self._prunable_modules_iterator():
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
        filters_to_prune = self._lowest_ranking_filters(num_filters_to_prune)

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
        for i, module in self._prunable_modules_iterator():
            if i in pruning_targets.keys():
                pruning_targets[i] = [x for x in range(len(module.ranking)) if x not in pruning_targets[i]]
        return pruning_targets


class WeightBasedPruner(AbstractPruner):
    """
    basic class for pruners that rank layers only based on the layer weights and requires no data or forward/backward
    operations for ranking the neurons
    """

    def __init__(self, bonsai, normalize=False):
        super().__init__(bonsai, normalize)

    def _attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        pass

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError


class ActivationBasedPruner(AbstractPruner):
    """
    basic class for pruners that rank layers based on the layer weights and activations with respect to given data and
     requires only forward of the model for ranking the neurons
    """

    def __init__(self, bonsai, normalize=False):
        super().__init__(bonsai, normalize)

    def _attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        module.activation = x.detach().cpu()

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError


class GradBasedPruner(AbstractPruner):
    """
    basic class for pruners that rank layers based on the layer weights, activations and gradients with respect to given
    data and requires both forward and backward pass of the model for ranking the neurons
    """

    def __init__(self, bonsai, normalize=False):
        super().__init__(bonsai, normalize)

    def _attach_hooks_for_rank_calculation(self, module: Prunable, x: torch.Tensor):
        module.activation = x.detach().cpu()
        x.register_hook(lambda grad: self._store_grad_in_module(module, grad))

    @staticmethod
    def _store_grad_in_module(module, grad):
        module.grad = grad.detach().cpu()

    @staticmethod
    def compute_single_layer_ranks(module, *args, **kwargs):
        raise NotImplementedError
