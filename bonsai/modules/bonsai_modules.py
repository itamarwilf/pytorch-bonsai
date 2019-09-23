from typing import Dict, Any
import torch
from torch import nn
from itertools import chain
from bonsai.modules.abstract_bonsai_classes import BonsaiModule, Prunable, Elementwise
from bonsai.modules.factories.activation_factory import construct_activation_from_config
from bonsai.utils.construct_utils import call_constructor_with_cfg


def conv_layer_output_size(module_cfg, in_h, in_w):
    padding = module_cfg.get("padding", 0)
    kernel_size = module_cfg.get("kernel_size", 0)
    stride = module_cfg.get("stride", 1)
    out_h = ((in_h + 2 * padding - kernel_size) // stride) + 1
    out_w = ((in_w + 2 * padding - kernel_size) // stride) + 1
    return out_h, out_w


# region conv2d
class AbstractBConv2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(AbstractBConv2d, self).__init__(bonsai_model, module_cfg)

        self.bn = None
        if module_cfg.get('batch_normalize'):
            self.bn = nn.BatchNorm2d(module_cfg['out_channels'])

        self.f = None
        if module_cfg.get('activation'):
            self.f = construct_activation_from_config(module_cfg)

        # if 'in_channels' in module_cfg use it
        # if it isn't, try to use out channels of prev layer if not None
        # if None, raise error
        if "in_channels" in module_cfg.keys():
            pass
        else:
            prev_layer_output = bonsai_model.output_channels[-1]
            if prev_layer_output is None:
                raise ValueError
            module_cfg['in_channels'] = bonsai_model.output_channels[-1]
        self.conv2d = call_constructor_with_cfg(nn.Conv2d, module_cfg)
        # pass output channels to next module using bonsai model
        bonsai_model.output_channels.append(module_cfg['out_channels'])

    def forward(self, layer_input):
        raise NotImplementedError

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        out_c = self.module_cfg.get("out_channels")
        if in_h is not None and in_w is not None:
            out_h, out_w = conv_layer_output_size(self.module_cfg, in_h, in_w)
            return out_c, out_h, out_w
        else:
            return out_c, None, None

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        if "conv2d.weight" in module_name:
            return module_tensor[:, pruning_targets]
        else:
            return module_tensor

    def propagate_pruning_target(self, initial_pruning_targets=None):
        if initial_pruning_targets:
            return initial_pruning_targets
        elif initial_pruning_targets is None:
            return list(range(self.module_cfg["out_channels"]))


class BConv2d(AbstractBConv2d):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)

    def forward(self, layer_input):
        x = self.conv2d(layer_input)

        if self.bn is not None:
            x = self.bn(x)

        if self.f is not None:
            x = self.f(x)

        return x


class PBConv2d(AbstractBConv2d, Prunable):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)

    def get_weights(self) -> torch.Tensor:
        return self.conv2d.weight.data

    def forward(self, layer_input):
        x = self.conv2d(layer_input)

        if self.get_model().to_rank:
            self.get_model().get_bonsai().prunner._attach_hooks_for_rank_calculation(self, x)

        if self.bn is not None:
            x = self.bn(x)

        if self.f is not None:
            x = self.f(x)

        return x

    @staticmethod
    def prune_output(pruning_targets, module_name, module_tensor):
        if "num_batches_tracked" in module_name:
            return module_tensor
        else:
            return module_tensor[pruning_targets]

# endregion


# region deconv2d

class AbstractBDeconv2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(AbstractBDeconv2d, self).__init__(bonsai_model, module_cfg)

        self.bn = None
        if 'batch_normalize' in module_cfg and module_cfg['batch_normalize']:
            self.bn = nn.BatchNorm2d(module_cfg['out_channels'])

        self.f = None
        if module_cfg.get('activation'):
            self.f = construct_activation_from_config(module_cfg)

        # takes number of input channels from prev layer
        module_cfg['in_channels'] = bonsai_model.output_channels[-1]
        self.deconv2d = call_constructor_with_cfg(nn.ConvTranspose2d, module_cfg)
        # pass output channels to next module using bonsai model
        bonsai_model.output_channels.append(module_cfg['out_channels'])

    def forward(self, layer_input):
        raise NotImplementedError

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        out_c = self.module_cfg.get("out_channels")
        stride = self.module_cfg.get("stride", 1)
        padding = self.module_cfg.get("padding", 0)
        kernel_size = self.module_cfg.get("kernel_size")
        dilation = self.module_cfg.get("dilation", 1)
        out_padding = self.module_cfg.get("out_padding", 0)
        if in_h is not None and in_w is not None:
            out_h = (in_h - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1
            out_w = (in_w - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1
            return out_c, out_h, out_w
        else:
            return out_c, None, None

    def propagate_pruning_target(self, initial_pruning_targets=None):
        if initial_pruning_targets:
            return initial_pruning_targets
        elif initial_pruning_targets is None:
            return list(range(self.module_cfg["out_channels"]))

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        if "deconv2d.weight" in module_name:
            return module_tensor[pruning_targets]
        else:
            return module_tensor


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

        if self.get_model().to_rank:
            self.get_model().get_bonsai().prunner._attach_hooks_for_rank_calculation(self, x)

        if self.bn is not None:
            x = self.bn(x)

        if self.f is not None:
            x = self.f(x)

        return x

    @staticmethod
    def prune_output(pruning_targets, module_name, module_tensor):
        if "num_batches_tracked" in module_name:
            return module_tensor
        elif "deconv2d.weight" in module_name:
            return module_tensor[:, pruning_targets]
        else:
            return module_tensor[pruning_targets]


# endregion


class BRoute(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BRoute, self).__init__(bonsai_model, module_cfg)
        # sum all the channels of concatenated tensors
        if isinstance(self.module_cfg["layers"], int):
            self.module_cfg["layers"] = [self.module_cfg["layers"]]
        out_channels = sum([bonsai_model.output_channels[layer_i] for layer_i in self.module_cfg["layers"]])
        # pass output channels to next module using bonsai model
        bonsai_model.output_channels.append(out_channels)

    def forward(self, layer_input):
        return torch.cat(tuple(self.get_model().output_manager[i] for i in self.module_cfg["layers"]), dim=1)

    def calc_layer_output_size(self, input_size):
        prev_layers_output_sizes = [self.get_model().output_sizes[i] for i in self.module_cfg["layers"]]
        out_c = 0
        out_h = None
        out_w = None
        for output_size in prev_layers_output_sizes:
            layer_c, out_h, out_w = output_size
            out_c += layer_c
        return out_c, out_h, out_w

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    # TODO - add more documentation
    def propagate_pruning_target(self, initial_pruning_targets=None):
        # get indices of layers to concat
        layer_indices = [len(self.get_model().pruning_targets) + i for i in self.module_cfg["layers"]]
        # reset desired output and index buffer
        result = []
        buffer = 0
        # iterate over concatenated layers
        for layer_idx in layer_indices:
            # result
            # get module whose output is concatenated
            # module = self.get_model().module_list[layer_idx]
            # module_out_channels = module.module_cfg["out_channels"]
            module_out_channels = self.get_model().output_sizes[layer_idx + 1][0]

            # if layer_idx in self.keys():
            #     module_pruning_targets = self.get_model().pruning_targets[layer_idx]
            # else:
            #     module_pruning_targets = list(range(module_out_channels))
            # module_pruning_targets = [x + buffer for x in module_pruning_targets]
            result.extend([x + buffer for x in self.get_model().pruning_targets[layer_idx]])
            buffer += module_out_channels
        return result


class BPixelShuffle(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BPixelShuffle, self).__init__(bonsai_model, module_cfg)
        self.pixel_shuffle = call_constructor_with_cfg(nn.PixelShuffle, self.module_cfg)
        # calc number of output channels using input channels and scaling factor
        out_channels = bonsai_model.output_channels[-1] / (self.module_cfg["upscale_factor"] ** 2)
        bonsai_model.output_channels.append(out_channels)

    def forward(self, layer_input):
        return self.pixel_shuffle(layer_input)

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        up_factor = self.module_cfg.get("upscale_factor")
        assert in_c % (up_factor ** 2) == 0, \
            f"{self.module_cfg.get('name')} - upscale of {up_factor} isn't compatible with input size {input_size}"
        out_c = in_c // (up_factor ** 2)
        if in_h is not None and in_w is not None:
            out_h = in_h * up_factor
            out_w = in_w * up_factor
        else:
            out_h = None
            out_w = None
        return out_c, out_h, out_w

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    def propagate_pruning_target(self, initial_pruning_targets=None):
        pass
        # prev_targets = self.get_model.pruning_targets[-1]
        # self.get_model.pruning_targets.append(prev_targets)


class BMaxPool2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BMaxPool2d, self).__init__(bonsai_model, module_cfg)
        self.maxpool = call_constructor_with_cfg(nn.MaxPool2d, self.module_cfg)
        # since max pooling doesn't change the tensor's number of channels, re append previous output channels
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    def forward(self, layer_input):
        return self.maxpool(layer_input)

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        if in_h is not None and in_w is not None:
            out_h, out_w = conv_layer_output_size(self.module_cfg, in_h, in_w)
            return in_c, out_h, out_w
        else:
            return in_c, None, None

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    def propagate_pruning_target(self, initial_pruning_targets=None):
        return self.get_model().pruning_targets[-1]


class BAvgPool2d(BonsaiModule):

    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BAvgPool2d, self).__init__(bonsai_model, module_cfg)
        self.avgpool2d = call_constructor_with_cfg(nn.AvgPool2d, self.module_cfg)
        # since max pooling doesn't change the tensor's number of channels, re append previous output channels
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    def forward(self, layer_input):
        return self.avgpool2d(layer_input)

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        if in_h is not None and in_w is not None:
            out_h, out_w = conv_layer_output_size(self.module_cfg, in_h, in_w)
            return in_c, out_h, out_w
        else:
            return in_c, None, None

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    def propagate_pruning_target(self, initial_pruning_targets=None):
        return self.get_model().pruning_targets[-1]


class BGlobalAvgPool(BonsaiModule):

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)
        self.avgpool = call_constructor_with_cfg(nn.AdaptiveAvgPool2d, self.module_cfg)
        # since max pooling doesn't change the tensor's number of channels, re append previous output channels
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    def calc_layer_output_size(self, input_size):
        in_c, _, _ = input_size
        return in_c, 1, 1

    def forward(self, layer_input):
        return self.avgpool(layer_input)

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    def propagate_pruning_target(self, initial_pruning_targets=None):
        return self.get_model().pruning_targets[-1]


class BFlatten(BonsaiModule):

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super(BFlatten, self).__init__(bonsai_model, module_cfg)
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    def forward(self, layer_input):
        n,  _, _, _ = layer_input.size()
        return layer_input.view(n, -1)

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        self.module_cfg["resolution"] = in_h * in_w
        self.module_cfg["channels"] = in_c
        return in_c * in_h * in_w

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    def propagate_pruning_target(self, initial_pruning_targets=None):
        # if initial_pruning_targets:
        if self.get_model().pruning_targets[-1]:
            pruning_targets = []
            for i in self.get_model().pruning_targets[-1]:
                pruning_targets.append(list(range(i * self.module_cfg["resolution"],
                                                  (i + 1) * self.module_cfg["resolution"])))
            return list(chain.from_iterable(pruning_targets))
        else:
            return list(range(self.module_cfg["resolution"] * self.module_cfg["channels"]))


# region linear

class AbstractBLinear(BonsaiModule):

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)

        self.bn = None
        if 'batch_normalize' in module_cfg and module_cfg['batch_normalize']:
            self.bn = nn.BatchNorm1d(module_cfg['out_features'])

        self.f = None
        if module_cfg.get('activation'):
            self.f = construct_activation_from_config(module_cfg)

        # if 'in_features' in module_cfg use it
        # if it isn't, try to use out size of prev layer if not None
        # if None, raise error
        if "in_features" in module_cfg.keys():
            pass
        else:
            prev_layer_output = bonsai_model.output_sizes[-1]
            if prev_layer_output is None:
                raise ValueError
            module_cfg['in_features'] = prev_layer_output

        self.linear = call_constructor_with_cfg(nn.Linear, self.module_cfg)
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    def calc_layer_output_size(self, input_size):
        return self.module_cfg.get("out_features")

    def forward(self, layer_input):
        raise NotImplementedError

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        if "linear.weight" in module_name:
            return module_tensor[:, pruning_targets]
        else:
            return module_tensor

    def propagate_pruning_target(self, initial_pruning_targets=None):
        if initial_pruning_targets:
            return initial_pruning_targets
        elif initial_pruning_targets is None:
            return list(range(self.module_cfg["out_features"]))


class BLinear(AbstractBLinear):

    def forward(self, layer_input):
        x = self.linear(layer_input)

        if self.bn is not None:
            x = self.bn(x)

        if self.f is not None:
            x = self.f(x)

        return x


class PBLinear(AbstractBLinear, Prunable):

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.data

    def forward(self, layer_input):
        x = self.linear(layer_input)

        if self.get_model().to_rank:
            self.get_model().get_bonsai().prunner._attach_hooks_for_rank_calculation(self, x)

        if self.bn is not None:
            x = self.bn(x)

        if self.f is not None:
            x = self.f(x)

        return x

    @staticmethod
    def prune_output(pruning_targets, module_name, module_tensor):
        if "num_batches_tracked" in module_name:
            return module_tensor
        else:
            return module_tensor[pruning_targets]

# endregion


class BDropout(BonsaiModule):

    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super().__init__(bonsai_model, module_cfg)
        self.dropout = call_constructor_with_cfg(nn.Dropout, module_cfg)
        bonsai_model.output_channels.append(bonsai_model.output_channels[-1])

    def forward(self, layer_input):
        return self.dropout(layer_input)

    def calc_layer_output_size(self, input_size):
        return input_size

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    def propagate_pruning_target(self, initial_pruning_targets=None):
        return self.get_model().pruning_targets[-1]


class BElementwiseAdd(Elementwise):
    def __init__(self, bonsai_model, module_cfg: Dict[str, Any]):
        super(BElementwiseAdd, self).__init__(bonsai_model, module_cfg)

        out_channels = bonsai_model.output_channels[self.module_cfg["layers"][0]]
        # pass output channels to next module using bonsai model
        self.f = None
        if module_cfg.get('activation'):
            self.f = construct_activation_from_config(module_cfg)
        bonsai_model.output_channels.append(out_channels)

    def forward(self, layer_input):
        layers = self.module_cfg["layers"]
        output = self.get_model().output_manager[layers[0]]
        for layer in layers[1:]:
            output += self.get_model().output_manager[layer]
        if self.f:
            output = self.f(output)
        return output

    def calc_layer_output_size(self, input_size):
        return self.get_model().output_sizes[self.module_cfg["layers"][0]]

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        pass

    # TODO - add more documentation
    def propagate_pruning_target(self, initial_pruning_targets=None):
        return self.get_model().pruning_targets[self.module_cfg["layers"][0]]


class BBatchNorm2d(BonsaiModule):
    def __init__(self, bonsai_model: nn.Module, module_cfg: Dict[str, Any]):
        super(BBatchNorm2d, self).__init__(bonsai_model, module_cfg)
        # if 'in_channels' in module_cfg use it
        # if it isn't, try to use out channels of prev layer if not None
        # if None, raise error
        if "in_channels" in module_cfg.keys():
            pass
        else:
            prev_layer_output = bonsai_model.output_channels[-1]
            if prev_layer_output is None:
                raise ValueError
            module_cfg['in_channels'] = bonsai_model.output_channels[-1]

        self.bn = nn.BatchNorm2d(module_cfg["in_channels"])

        self.f = None
        if module_cfg.get('activation'):
            self.f = construct_activation_from_config(module_cfg)
        # TODO - the following line is important for model building, document well or find safer implementation
        bonsai_model.output_channels.append(module_cfg['in_channels'])

    def forward(self, layer_input):
        x = self.bn(layer_input)
        if self.f is not None:
            x = self.f(x)
        return x

    def calc_layer_output_size(self, input_size):
        in_c, in_h, in_w = input_size
        return in_c, in_h, in_w

    @staticmethod
    def prune_input(pruning_targets, module_name, module_tensor):
        if "num_batches_tracked" in module_name:
            return module_tensor
        else:
            return module_tensor[pruning_targets]

    def propagate_pruning_target(self, initial_pruning_targets=None):
        return self.get_model().pruning_targets[-1]
