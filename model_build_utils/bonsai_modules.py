import torch
from torch import nn
from inspect import getfullargspec
from bonsai import Bonsai
from typing import Dict
from model_build_utils.factories import BonsaiFactory, NonLinearFactory


def call_constructor_with_cfg(constructor, cfg: dict):
    """
    filters cfg based on cfg args and kwargs and creates instance
    :param constructor: function, should be used for class instance creation
    :param cfg: dict, containing keys for constructor
    :return: instance of class based on constructor and appropriate cfg
    """
    kwargs = getfullargspec(constructor)
    constructor_cfg = {k: v for (k, v) in cfg.items() if k in kwargs}
    return constructor(**constructor_cfg)


class BonsaiModule(nn.Module):

    def __init__(self, bonsai_model: Bonsai, module_cfg: Dict[str]):
        super(BonsaiModule, self).__init__()
        self.bonsai_model = bonsai_model
        self.module_cfg = module_cfg

    def forward(self, *layer_input):
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


class BonsaiConv2d(BonsaiModule, Prunable):

    def __init__(self, bonsai_model, module_cfg):
        super().__init__(bonsai_model)
        self.mtype = module_cfg.pop('type')
        self.cfg = module_cfg

        self.bn = None
        if 'batchnorm' in module_cfg and module_cfg['batchnorm']:
            self.bn = nn.BatchNorm2d(module_cfg['out_channels'])

        self.f = None
        if 'activation' in module_cfg:
            activation_creator = NonLinearFactory.get_creator(module_cfg['activation'])
            self.f = call_constructor_with_cfg(activation_creator, module_cfg)

        # take input channels from prev layer
        module_cfg['in_channels'] = bonsai_model.channels[-1]
        self.conv2d = call_constructor_with_cfg(nn.Conv2d, module_cfg)
        # pass output channels to next module using bonsai model
        bonsai_model.channels.append(module_cfg['out_channels'])

    def forward(self, *layer_input):
        x = self.conv2d(layer_input)

        if self.bonsai_model.prune:
            x.register_hook(self.bonsai_model.pruning_func)
            self.activation = x

        x = self.bn(x)
        x = self.f(x)

        return x

    def prune_output(self):
        pass

    def prune_input(self):
        pass


class BonsaiConcat(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg):
        super().__init__(bonsai_model, module_cfg)
        self.layers = [int(x) for x in module_cfg["layers"].split(",")]
        # sum all the channels of concatenated tensors
        out_channels = sum([bonsai_model.output_filters[layer_i] for layer_i in self.layers])
        # pass output channels to next module using bonsai model
        bonsai_model.channels.append(out_channels)

    def forward(self, *layer_input):
        return torch.cat(tuple(self.bonsai_model.layer_outputs.get(i) for i in self.layers), dim=1)


def create_bonsai_modules(bonsai_model: nn.Module) -> nn.ModuleList:
    module_list = nn.ModuleList()
    # number of input channels for next layer is taken from prev layer output channels (or model input)
    bonsai_model.channels = [int(bonsai_model.hyperparams['in_channels'])]
    # TODO remove counter for names once better naming is implemented
    counter = 1
    # iterate over module definitions to create and add modules to bonsai model
    for module_def in bonsai_model.module_defs:
        module_type = module_def['type']
        # get the module creator based on type
        module_creator = BonsaiFactory.get_creator(module_type)
        # create the module using the creator and module cfg
        module = module_creator(bonsai_model, module_def)
        # TODO - find better naming mechanism, maybe take names from original parsed model after jit traced parsing is
        # implemented
        module_name = module_type.join(str(counter))
        counter += 1
        module_list.add_module(module_name, module)
    return module_list
