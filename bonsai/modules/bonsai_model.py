import copy
import weakref
from collections import Counter, OrderedDict
from typing import List
import torch
from torch import nn
from bonsai.modules.abstract_bonsai_classes import Prunable
from bonsai.modules.receptive_field_calculation import calc_receptive_field
from bonsai.modules.factories.bonsai_module_factory import BonsaiFactory
from bonsai.modules.model_cfg_parser import basic_model_cfg_parsing

def sprint(tag, tensor):
    print(tag+': ', 'size: ', tensor.size(),'min: ', float(torch.min(tensor)), 'max: ', float(torch.max(tensor)), 'avg: ', float(torch.mean(tensor)))

class BonsaiModel(torch.nn.Module):
    """
    a model made of wrappers to pytorch modules. It functions as a regular nn.Module, but it can also be used for
    propagating pruning instructions between different modules.

    Args:
        cfg_path (str): a path to the model config file, look at example models for reference.
        bonsai : the model's parent Bonsai object
    """

    def __init__(self, cfg_path, bonsai=None):
        super(BonsaiModel, self).__init__()
        self.bonsai = None
        if bonsai:
            self.bonsai = weakref.ref(bonsai)
        self.device = None

        # TODO - remove output_channels and replace with output_sizes everywhere
        self.output_channels: List[int] = []
        self.output_sizes: List = []

        # self.layer_outputs: List[torch.Tensor] = []
        self.model_output = []

        self.pruning_targets = []
        self.to_rank = False

        self.full_cfg = basic_model_cfg_parsing(cfg_path)  # type: List[dict]
        self.module_cfgs = copy.deepcopy(self.full_cfg)
        self.hyperparams = self.module_cfgs.pop(0)  # type: dict

        self.module_list: nn.ModuleList = self._create_bonsai_modules()

        self.output_manager = self._create_output_manager()

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def get_bonsai(self):
        """
        Returns: the model's parent Bonsai object
        """
        return self.bonsai()

    def forward(self, model_input):
        """
        runs model on given tensor
        Args:
            model_input: tensor or list of tensors to run the model on

        Returns:
            model's output
        """

        if isinstance(model_input, list):
            x = None
        elif isinstance(model_input, torch.Tensor):
            x = model_input
        else:
            raise TypeError(f"Model input must be torch.Tensor or List[torch.Tensor], got {type(model_input)}")

        output = []
        self.output_manager.reset()

        for i, module in enumerate(self.module_list):
            if "input" in module.module_cfg.keys():
                x = module(model_input[module.module_cfg["input"]])
            else:
                x = module(x)
            sprint(module.module_cfg['name'], x)
            self.output_manager[module.module_cfg["name"]] = x
            if module.module_cfg.get("output"):
                output.append(x)
        self.output_manager.reset()
        return output

    def _create_bonsai_modules(self) -> nn.ModuleList:
        """
        Iterates over given module configs from the model config file.
        Calculates the number of channels/features going into each module for automatic model building.

        Returns:
            torch.nn.ModuleList containing all the modules of the model

        """
        module_list = nn.ModuleList()
        # number of input channels for next layer is taken from prev layer output channels (or model input)
        in_h = self.hyperparams.get("height")
        in_w = self.hyperparams.get("width")
        in_c = self.hyperparams.get("in_channels")
        self.output_channels.append(int(self.hyperparams['in_channels']))
        self.output_sizes.append((in_c, in_h, in_w))
        counter = Counter()
        # iterate over module definitions to create and add modules to bonsai model
        for layer_num, module_cfg in enumerate(self.module_cfgs):
            # TODO - maybe take names from original parsed model after jit traced parsing is implemented
            module_type = module_cfg['type']
            counter[module_type] += 1
            if not module_cfg.get("name"):
                module_name = module_type + "_" + str(counter[module_type])
                module_cfg["name"] = module_name

            # get the module creator based on type
            module_creator = BonsaiFactory.get_creator(module_type)
            # create the module using the creator and module cfg
            module_cfg["prev_out_size"] = self.output_sizes[-1]
            module = module_creator(self, module_cfg)
            self.output_sizes.append(module.calc_layer_output_size(self.output_sizes[-1]))

            module_list.append(module)
        return module_list

    def total_prunable_filters(self):
        """
        Returns: number of prunable channels + features in the model
        """
        filters = 0
        for module in self.module_list:
            if isinstance(module, Prunable):
                if module.module_cfg.get("out_channels"):
                    filters += int(module.module_cfg.get("out_channels"))
                elif module.module_cfg.get("out_features"):
                    filters += int(module.module_cfg.get("out_features"))
        return filters

    def propagate_pruning_targets(self, initial_pruning_targets):
        """
        Propagates each layer pruning targets across entire model to account for all place where this pruning has any
        effect on the model, such as pruning input channels to another layer,

        Args:
            initial_pruning_targets: dictionary with layer index as key and list of channels/feature indices that needs
            to be pruned at that layer

        Returns: None

        """
        self.pruning_targets = [list(range(self.output_channels[0]))]

        for i, module in enumerate(self.module_list):
            module_pruning_targets = None
            if i in initial_pruning_targets.keys():
                module_pruning_targets = initial_pruning_targets[i]
            current_target = module.propagate_pruning_target(module_pruning_targets)

            if current_target is None:
                current_target = []
            self.pruning_targets.append(current_target)

    def calc_receptive_field(self):
        """
        calculates convolutions receptive field at each layer of the model
        Returns: None
        """
        calc_receptive_field(self.module_cfgs)

    def _create_output_manager(self):
        """
        static analysis of layers, counting the number of times a layer output is used for memory management
        Returns:


        """
        counter = OrderedDict()
        for module_config in self.module_cfgs:

            # TODO - part of hack
            # if module_config["name"] in counter.keys():
            #     counter[module_config["name"]] += 1
            # else:
            counter[module_config["name"]] = 0

            if "input" in module_config.keys():
                if isinstance(module_config["input"], int):
                    name = list(counter.keys())[module_config["input"]]
                    counter[name] += 1
                elif isinstance(module_config["input"], str):
                    counter[module_config["input"]] += 1

            if module_config.get("layers"):
                for layer in module_config.get("layers"):
                    if isinstance(layer, int):
                        name = list(counter.keys())[layer - 1]
                        counter[name] += 1
                    elif isinstance(layer, str):
                        counter[layer] += 1

        return _OutputManager(counter)


class _OutputManager:

    def __init__(self, init_counter: OrderedDict):
        self.init_counter = init_counter
        self.counter = copy.deepcopy(self.init_counter)
        self.outputs = OrderedDict()

    def __setitem__(self, key, value):
        # if isinstance(key, int):
        #     key = list(self.outputs.keys())[key]
        if self.counter[key] > 0:
            self.outputs[key] = value
        # TODO - hack for maintaining correct list len
        else:
            self.outputs[key] = None

    def __getitem__(self, item):
        if isinstance(item, int):
            item = list(self.outputs.keys())[item]
        value = self.outputs[item]
        self.counter[item] -= 1
        if self.counter[item] == 0:
            self.outputs[item] = None
        return value

    def reset(self):
        self.counter = copy.deepcopy(self.init_counter)
        self.outputs = OrderedDict()
