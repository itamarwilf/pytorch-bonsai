import torch
import numpy as np
from operator import itemgetter
from heapq import nsmallest
from torch.autograd import Variable


def compute_rank(self, grad):
    activation_index = len(self.activations) - self.grad_index - 1
    activation = self.activations[activation_index]
    values = \
        torch.sum((activation * grad), dim=0). \
            sum(dim=2).sum(dim=3)[0, :, 0, 0].data

    # Normalize the rank by the filter dimensions
    values = \
        values / (activation.size(0) * activation.size(2) * activation.size(3))

    if activation_index not in self.filter_ranks:
        self.filter_ranks[activation_index] = \
            torch.FloatTensor(activation.size(1)).zero_().cuda()

    self.filter_ranks[activation_index] += values
    self.grad_index += 1


def lowest_ranking_filters(self, num):
    data = []
    for i in sorted(self.filter_ranks.keys()):
        for j in range(self.filter_ranks[i].size(0)):
            data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

    return nsmallest(num, data, itemgetter(2))


def normalize_ranks_per_layer(self):
    for i in self.filter_ranks:
        v = torch.abs(self.filter_ranks[i])
        v = v / np.sqrt(torch.sum(v * v))
        self.filter_ranks[i] = v.cpu()


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


def train_batch(self, optimizer, batch, label, rank_filters):
    self.model.zero_grad()
    input = Variable(batch)

    if rank_filters:
        output = self.prunner.forward(input)
        self.criterion(output, Variable(label)).backward()
    else:
        self.criterion(self.model(input), Variable(label)).backward()
        optimizer.step()


def train_epoch(self, optimizer=None, rank_filters=False):
    for batch, label in self.train_data_loader:
        self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)


def get_candidates_to_prune(self, num_filters_to_prune):
    self.prunner.reset()

    self.train_epoch(rank_filters=True)

    self.prunner.normalize_ranks_per_layer()

    return self.prunner.get_prunning_plan(num_filters_to_prune)