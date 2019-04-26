import torch
from model_build_utils.model_parser import parse_model_cfg
from model_build_utils.bonsai_modules import create_bonsai_modules
from typing import List, Dict


class Bonsai(torch.nn.Module):

    def __init__(self, cfg_path):
        super(Bonsai, self).__init__()
        self.device = None
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
