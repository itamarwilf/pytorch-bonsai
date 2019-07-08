import copy
import weakref
from collections import Counter
from typing import List
import torch
from torch import nn
from modules.errors import NotBonsaiModuleError
from modules.abstract_bonsai_classes import Prunable
from modules.factories.bonsai_module_factory import BonsaiFactory
from modules.model_cfg_parser import basic_model_cfg_parsing


class BonsaiModel(torch.nn.Module):

    def __init__(self, cfg_path, bonsai):
        super(BonsaiModel, self).__init__()
        self.bonsai = weakref.ref(bonsai)
        self.device = None

        self.output_channels: List[int] = []
        self.output_sizes: List = []

        self.layer_outputs: List[torch.Tensor] = []
        self.model_output = []

        self.pruning_targets = []
        self.to_rank = False

        self.full_cfg = basic_model_cfg_parsing(cfg_path)  # type: List[dict]
        self.module_cfgs = copy.deepcopy(self.full_cfg)
        self.hyperparams = self.module_cfgs.pop(0)  # type: dict

        self.module_list = self._create_bonsai_modules()  # type: nn.ModuleList

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def get_bonsai(self):
        return self.bonsai()

    def _reset_forward(self):

        self.model_output = []
        self.layer_outputs = []

    def forward(self, x):

        self._reset_forward()

        for i, module in enumerate(self.module_list):
            x = module(x)
            self.layer_outputs.append(x)
            if module.module_cfg.get("output"):
                self.model_output.append(x)

        output = self.model_output
        self._reset_forward()
        return output

    # TODO needs design for nn.Linear construction, including feature map size, strides, kernel sizes, etc.
    def _create_bonsai_modules(self) -> nn.ModuleList:
        module_list = nn.ModuleList()
        # number of input channels for next layer is taken from prev layer output channels (or model input)
        in_h = self.hyperparams.get("height")
        in_w = self.hyperparams.get("width")
        in_c = self.hyperparams.get("in_channels")
        self.output_channels.append(int(self.hyperparams['in_channels']))
        self.output_sizes.append((in_c, in_h, in_w))
        counter = Counter()
        # iterate over module definitions to create and add modules to bonsai model
        for module_cfg in self.module_cfgs:
            # TODO - maybe take names from original parsed model after jit traced parsing is implemented
            module_type = module_cfg['type']
            counter[module_type] += 1
            module_name = module_type + "_" + str(counter[module_type])
            module_cfg["name"] = module_name

            # get the module creator based on type
            module_creator = BonsaiFactory.get_creator(module_type)
            # create the module using the creator and module cfg
            module = module_creator(self, module_cfg)
            self.output_sizes.append(module.calc_layer_output_size(self.output_sizes[-1]))

            module_list.append(module)
        return module_list

    # TODO - add docstring
    def total_prunable_filters(self):
        filters = 0
        for module in self.module_list:
            if isinstance(module, Prunable):
                filters += int(module.module_cfg.get("out_channels"))
        return filters

    # TODO - add docstring
    def propagate_pruning_targets(self, inital_pruning_targets):

        self.pruning_targets = [list(range(self.output_channels[0]))]

        for i, module in enumerate(self.module_list):
            module_pruing_targets = None
            if i in inital_pruning_targets.keys():
                module_pruing_targets = inital_pruning_targets[i]
            current_target = module.propagate_pruning_target(module_pruing_targets)

            if current_target is None:
                current_target = []
            self.pruning_targets.append(current_target)
