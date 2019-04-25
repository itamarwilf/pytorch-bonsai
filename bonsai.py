import torch
from model_build_utils.model_parser import parse_model_cfg
from model_build_utils.bonsai_modules import create_bonsai_modules


class Bonsai(torch.nn.Module):

    def __init__(self, cfg_path):
        super(Bonsai, self).__init__()
        self.device = None
        self.layer_outputs = {}
        self.model_output = []

        self.last_pruned = None
        self.prune = False

        self.module_defs = parse_model_cfg(cfg_path)
        self.hyperparams = self.module_defs.pop(0)
        self.module_list = create_bonsai_modules(self)

    def forward(self, x, var=None):
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p
