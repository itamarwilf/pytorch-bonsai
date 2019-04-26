import torch
from torch import nn
from inspect import getfullargspec
from model_build_utils.abstract_bonsai_classes import BonsaiModule, Prunable
from model_build_utils.factories import NonLinearFactory


def call_constructor_with_cfg(constructor, cfg: dict):
    """
    utility functions for constructors, filters cfg based on cfg args and kwargs and creates instance
    :param constructor: function, should be used for class instance creation
    :param cfg: dict, containing keys for constructor
    :return: instance of class based on constructor and appropriate cfg
    """
    kwargs = getfullargspec(constructor)
    constructor_cfg = {k: v for (k, v) in cfg.items() if k in kwargs}
    return constructor(**constructor_cfg)


class BonsaiConv2d(BonsaiModule, Prunable):

    def __init__(self, bonsai_model, module_cfg):
        super(BonsaiConv2d).__init__(bonsai_model, module_cfg)

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
        super(BonsaiConcat).__init__(bonsai_model, module_cfg)
        self.layers = [int(x) for x in module_cfg["layers"].split(",")]
        # sum all the channels of concatenated tensors
        out_channels = sum([bonsai_model.output_filters[layer_i] for layer_i in self.layers])
        # pass output channels to next module using bonsai model
        bonsai_model.channels.append(out_channels)

    def forward(self, *layer_input):
        return torch.cat(tuple(self.bonsai_model.layer_outputs.get(i) for i in self.layers), dim=1)


class BonsaiDeconv2d(BonsaiModule, Prunable):

    def __init__(self, bonsai_model, module_cfg):
        super(BonsaiDeconv2d).__init__(bonsai_model, module_cfg)

        self.bn = None
        if 'batchnorm' in module_cfg and module_cfg['batchnorm']:
            self.bn = nn.BatchNorm2d(module_cfg['out_channels'])

        self.f = None
        if 'activation' in module_cfg:
            activation_creator = NonLinearFactory.get_creator(module_cfg['activation'])
            self.f = call_constructor_with_cfg(activation_creator, module_cfg)

        # takes number of input channels from prev layer
        module_cfg['in_channels'] = bonsai_model.channels[-1]
        self.conv2d = call_constructor_with_cfg(nn.ConvTranspose2d, module_cfg)
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

    def prune_input(self):
        pass

    def prune_output(self):
        pass
