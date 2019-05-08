import torch
import numpy as np
from operator import itemgetter
from heapq import nsmallest


class AbstractPrunner:

    def __init__(self):
        self.filter_ranks = {}

    @classmethod
    def compute_filter_ranks(cls):
        """
        function that computes the rank of filters in conv2d and deconv2d layers for pruning.
        this function wil be called by individual prunable modules during prunning process
        :return:
        """
        raise NotImplementedError

    def reset(self):
        self.filter_ranks = {}

    def normalize_filter_ranks_per_layer(self):
        """
        performs the layer-wise l2 normalization featured in:
        [1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference]
        :return: None
        """
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    # TODO take filter ranks from bonsai modules
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

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

    def __init__(self):
        super().__init__()
        self.activation_needed = False
        self.grad_needed = False

    @classmethod
    def compute_filter_ranks(cls):
        raise NotImplementedError


class ActivationBasedPrunner(AbstractPrunner):

    def __init__(self):
        super().__init__()

    @classmethod
    def compute_filter_ranks(cls):
        raise NotImplementedError


class GradBasedPrunner(AbstractPrunner):

    def __init__(self):
        super().__init__()

    @classmethod
    def compute_filter_ranks(cls):
        raise NotImplementedError


