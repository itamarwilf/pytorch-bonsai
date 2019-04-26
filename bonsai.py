import torch
from torch import nn

from model_build_utils.factories import BonsaiFactory
from model_build_utils.model_parser import parse_model_cfg
from typing import List, Dict


class Bonsai(torch.nn.Module):

    def __init__(self, cfg_path):
        super(Bonsai, self).__init__()
        self.device = None
        self.output_channels = List[int]
        self.layer_outputs = Dict[int]
        self.model_output = []

        self.last_pruned = None
        self.prune = False

        self.module_cfgs = parse_model_cfg(cfg_path)  # type: List[dict]
        self.hyperparams = self.module_cfgs.pop(0)  # type: dict
        self.module_list = create_bonsai_modules(self)

    def forward(self, x):

        for i, (module_cfg, module) in enumerate(zip(self.module_cfgs, self.module_list)):

            x = module(x)
            # mtype = module_cfg['type']
            # if mtype in ['convolutional', 'upsample', 'maxpool']:
            #     x = module(x)
            # elif mtype == 'route':
            #     layer_i = [int(x) for x in module_cfg['layers'].split(',')]
            #     if len(layer_i) == 1:
            #         x = layer_outputs[layer_i[0]]
            #     else:
            #         x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            # elif mtype == 'shortcut':
            #     layer_i = int(module_cfg['from'])
            #     x = layer_outputs[-1] + layer_outputs[layer_i]
            # layer_outputs.append(x)
            if module_cfg.get("output") is True:
                self.model_output.append(x)

        return self.model_output


def create_bonsai_modules(bonsai_model: nn.Module) -> nn.ModuleList:
    module_list = nn.ModuleList()
    # number of input channels for next layer is taken from prev layer output channels (or model input)
    bonsai_model.output_channels.append(int(bonsai_model.hyperparams['in_channels']))
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
