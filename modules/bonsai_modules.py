from typing import Dict, Any

import torch
from torch import nn
from inspect import getfullargspec

from modules.abstract_bonsai_classes import BonsaiModule, Prunable
from modules.factories.non_linear_factory import NonLinearFactory
from modules.module_build_utils import parse_kernel_size


def call_constructor_with_cfg(constructor, cfg: dict):
    """
    utility functions for constructors, filters cfg based on cfg args and kwargs and creates instance
    :param constructor: function, should be used for class instance creation
    :param cfg: dict, containing keys for constructor
    :return: instance of class based on constructor and appropriate cfg
    """
    kwargs = getfullargspec(constructor).args
    constructor_cfg = {k: v for (k, v) in cfg.items() if k in kwargs}
    # print(constructor_cfg)
    return constructor(**constructor_cfg)


GLOBAL_MODULE_CFGS = ["type", "name", "output"]


# region conv2d

class AbstractBConv2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(AbstractBConv2d, self).__init__(bonsai_model, module_cfg)

        self.bn = None
        if 'batch_normalize' in module_cfg and module_cfg['batch_normalize']:
            self.bn = nn.BatchNorm2d(module_cfg['out_channels'])

        self.f = None
        if 'activation' in module_cfg and module_cfg['activation'] is not None:
            activation_creator = NonLinearFactory.get_creator(module_cfg['activation'])
            self.f = call_constructor_with_cfg(activation_creator, module_cfg)

        # take input channels from prev layer
        module_cfg['in_channels'] = bonsai_model.output_channels[-1]
        self.conv2d = call_constructor_with_cfg(nn.Conv2d, module_cfg)
        # pass output channels to next module using bonsai model
        bonsai_model.output_channels.append(module_cfg['out_channels'])

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        for k, v in module_cfg.items():
            if k in GLOBAL_MODULE_CFGS:
                pass
            elif k == "batch_normalize":
                try:
                    new_v = int(v)
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
            elif k == "out_channels":
                try:
                    new_v = int(v)
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
            elif k == "kernel_size":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "stride":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "padding":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "activation":
                if v == "none":
                    module_cfg[k] = None
            # TODO - separate activation parsing from module
            elif k == "negative_slope":
                try:
                    new_v = float(v)
                    assert 0 < new_v < 1, f"slope for leaky ReLU in {module_cfg['name']} is {new_v} while " \
                        f"it should be 0 < slope < 1"
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
            else:
                raise NotImplementedError(f"parsing of '{k}' for module '{module_cfg['type']}' is not implemented")
        return module_cfg

    def forward(self, layer_input):
        raise NotImplementedError


class BConv2d(AbstractBConv2d):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)

    def forward(self, layer_input):
        x = self.conv2d(layer_input)

        if self.f is not None:
            x = self.f(x)

        if self.bn is not None:
            x = self.bn(x)
        return x


class PBConv2d(AbstractBConv2d, Prunable):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)

    def get_weights(self) -> torch.Tensor:
        return self.conv2d.weight.data

    def forward(self, layer_input):
        x = self.conv2d(layer_input)

        if self.bonsai_model.to_rank:
            self.activation = x
            x.register_hook(self.bonsai_model.pruning_func)

        if self.f is not None:
            x = self.f(x)

        if self.bn is not None:
            x = self.bn(x)
        return x

    def prune_output(self):
        pass

    def prune_input(self):
        pass

    def _prune_params(self, grad):
        pass

# endregion


# region deconv2d

class AbstractBDeconv2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(AbstractBDeconv2d, self).__init__(bonsai_model, module_cfg)

        self.bn = None
        if 'batch_normalize' in module_cfg and module_cfg['batch_normalize']:
            self.bn = nn.BatchNorm2d(module_cfg['out_channels'])

        self.f = None
        if 'activation' in module_cfg and module_cfg['activation'] is not None:
            activation_creator = NonLinearFactory.get_creator(module_cfg['activation'])
            self.f = call_constructor_with_cfg(activation_creator, module_cfg)

        # takes number of input channels from prev layer
        module_cfg['in_channels'] = bonsai_model.output_channels[-1]
        self.deconv2d = call_constructor_with_cfg(nn.ConvTranspose2d, module_cfg)
        # pass output channels to next module using bonsai model
        bonsai_model.output_channels.append(module_cfg['out_channels'])

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        for k, v in module_cfg.items():
            if k in GLOBAL_MODULE_CFGS:
                pass
            elif k == "batch_normalize":
                try:
                    new_v = int(v)
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
            elif k == "out_channels":
                try:
                    new_v = int(v)
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
            elif k == "kernel_size":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "stride":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "padding":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "activation":
                if v == "none":
                    module_cfg[k] = None
            elif k == "negative_slope":
                try:
                    new_v = float(v)
                    assert 0 < new_v < 1, f"slope for leaky ReLU in {module_cfg['name']} is {new_v} while " \
                        f"it should be 0 < slope < 1"
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
            else:
                raise NotImplementedError(f"parsing of {k} for module {module_cfg['type']} is not implemented")
        return module_cfg

    def forward(self, layer_input):
        raise NotImplementedError


class BDeconv2d(AbstractBDeconv2d):

    def forward(self, layer_input):
        x = self.deconv2d(layer_input)

        if self.bn is not None:
            x = self.bn(x)
        if self.f is not None:
            x = self.f(x)

        return x


class PBDeconv2d(AbstractBDeconv2d, Prunable):

    def get_weights(self) -> torch.Tensor:
        return self.deconv2d.weight.data.permute(1, 0, 2, 3)

    def forward(self, layer_input):
        x = self.deconv2d(layer_input)

        if self.bonsai_model.to_rank:
            self.activation = x
            x.register_hook(self.bonsai_model.pruning_func)

        if self.bn is not None:
            x = self.bn(x)
        if self.f is not None:
            x = self.f(x)

        return x

    def prune_output(self):
        pass

    def prune_input(self):
        pass

# endregion


class BRoute(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BRoute, self).__init__(bonsai_model, module_cfg)
        # sum all the channels of concatenated tensors
        out_channels = sum([bonsai_model.output_channels[layer_i] for layer_i in self.module_cfg["layers"]])
        # pass output channels to next module using bonsai model
        bonsai_model.output_channels.append(out_channels)

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        for k, v in module_cfg.items():
            if k in GLOBAL_MODULE_CFGS:
                pass
            elif k == "layers":
                try:
                    new_v = [int(x) for x in v.split(',')]
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} should be integers separated by commas,"
                                     f"got {v}")
            else:
                raise NotImplementedError(f"parsing of {k} for module {module_cfg['type']} is not implemented")
        return module_cfg

    def forward(self, layer_input):
        return torch.cat(tuple(self.bonsai_model.layer_outputs[i] for i in self.module_cfg["layers"]), dim=1)


class BPixelShuffle(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BPixelShuffle, self).__init__(bonsai_model, module_cfg)
        self.pixel_shuffle = call_constructor_with_cfg(nn.PixelShuffle, self.module_cfg)
        # calc number of output channels using input channels and scaling factor
        out_channels = bonsai_model.output_channels[-1] / (self.module_cfg["upscale_factor"] ** 2)
        bonsai_model.output_channels.append(out_channels)

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        for k, v in module_cfg.items():
            if k in GLOBAL_MODULE_CFGS:
                pass
            elif k == "upscale_factor":
                try:
                    new_v = int(v)
                    assert new_v > 0, f"upscale factor should be a positive integer, got {new_v}"
                    module_cfg[k] = new_v
                except ValueError:
                    raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be a positive int")
            else:
                raise NotImplementedError(f"parsing of {k} for module {module_cfg['type']} is not implemented")
        return module_cfg

    def forward(self, layer_input):
        return self.pixel_shuffle(layer_input)


class BMaxPool(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BMaxPool, self).__init__(bonsai_model, module_cfg)
        self.maxpool = call_constructor_with_cfg(nn.MaxPool2d, self.module_cfg)
        # since max pooling doesn't change the tensor's number of channels, re append previous output channels
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        for k, v in module_cfg.items():
            if k in GLOBAL_MODULE_CFGS:
                pass
            elif k == "kernel_size":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "stride":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            else:
                raise NotImplementedError(f"parsing of {k} for module {module_cfg['type']} is not implemented")
        return module_cfg

    def forward(self, layer_input):
        return self.maxpool(layer_input)


class BAvgPool2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BAvgPool2d, self).__init__(bonsai_model, module_cfg)
        self.avgpool2d = call_constructor_with_cfg(nn.AvgPool2d, self.module_cfg)
        # since max pooling doesn't change the tensor's number of channels, re append previous output channels
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        for k, v in module_cfg.items():
            if k in GLOBAL_MODULE_CFGS:
                pass
            elif k == "kernel_size":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            elif k == "stride":
                module_cfg = parse_kernel_size(module_cfg, k, v)
            else:
                raise NotImplementedError(f"parsing of {k} for module {module_cfg['type']} is not implemented")
        return module_cfg

    def forward(self, layer_input):
        return self.avgpool2d(layer_input)


class BGlobalAvgPool(BonsaiModule):

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)

    @staticmethod
    def _parse_module_cfg(module_cfg: dict) -> dict:
        pass

    def forward(self, layer_input):
        torch.mean(layer_input, dim=(2, 3))


# TODO linear layer needs more complicated implementaion
# class BLinear(BonsaiModule):
#
#     def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
#         super().__init__(bonsai_model, module_cfg)
#         self.linear = call_constructor_with_cfg(nn.Linear, self.module_cfg)
#         bonsai_model.output_channels.append(bonsai_model.output_channels[-1])
#
#     @staticmethod
#     def _parse_module_cfg(module_cfg: dict) -> dict:
#         for k, v in module_cfg.items():
#             if k in GLOBAL_MODULE_CFGS:
#                 pass
#             elif k == "batch_normalize":
#                 try:
#                     new_v = int(v)
#                     module_cfg[k] = new_v
#                 except ValueError:
#                     raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
#             elif k == "out_features":
#                 try:
#                     new_v = int(v)
#                     module_cfg[k] = new_v
#                 except ValueError:
#                     raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
#
#             # TODO change cfg key to correct name for specific constructor, maybe should be done in bonsai_model method
#             elif k == "in_channels":
#                 module_cfg["in_features"] = parse_kernel_size(module_cfg, k, v)
#             elif k == "bias":
#                 try:
#                     new_v = int(v)
#                     assert 0 <= new_v <= 1, f"{k} in config of {module_cfg['name']} is {v}, should be an int of 0 or 1"
#                     module_cfg[k] = new_v
#                 except ValueError:
#                     raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, should be an int of 0 or 1")
#             elif k == "activation":
#                 if v == "none":
#                     module_cfg[k] = None
#             # TODO - separate activation parsing from module
#             elif k == "negative_slope":
#                 try:
#                     new_v = float(v)
#                     assert 0 < new_v < 1, f"slope for leaky ReLU in {module_cfg['name']} is {new_v} while " \
#                         f"it should be 0 < slope < 1"
#                     module_cfg[k] = new_v
#                 except ValueError:
#                     raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be an int")
#             else:
#                 raise NotImplementedError(f"parsing of '{k}' for module '{module_cfg['type']}' is not implemented")
#         return module_cfg
#
#     def forward(self, layer_input):
#         return self.linear(layer_input)
