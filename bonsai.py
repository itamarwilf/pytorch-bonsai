import torch
from torch import nn

from modules.factories.bonsai_module_factory import BonsaiFactory
from modules.model_cfg_parser import parse_model_cfg
from typing import List, Dict
from collections import Counter


class Bonsai(torch.nn.Module):

    def __init__(self, cfg_path):
        super(Bonsai, self).__init__()
        self.device = None
        self.output_channels: List[int] = []
        self.layer_outputs: List[torch.Tensor] = []
        self.model_output = []

        self.last_pruned = None
        self.prune = False

        self.module_cfgs = parse_model_cfg(cfg_path)  # type: List[dict]
        self.hyperparams = self.module_cfgs.pop(0)  # type: dict
        self.module_list = self._create_bonsai_modules()

    def forward(self, x):

        for i, module in enumerate(self.module_list):
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
            self.layer_outputs.append(x)
            if module.module_cfg.get("output"):
                self.model_output.append(x)

        return self.model_output

    def _create_bonsai_modules(self) -> nn.ModuleList:
        module_list = nn.ModuleList()
        # number of input channels for next layer is taken from prev layer output channels (or model input)
        self.output_channels.append(int(self.hyperparams['in_channels']))
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
            module_list.add_module(module_name, module)
        return module_list

    # TODO - needs rewriting
    def total_num_filters(self):
        filters = 0
        for mod_def in self.module_cfgs:
            if mod_def["type"] == "convolutional" and "avoid_pruning" not in mod_def.keys():
                filters += int(mod_def["filters"])
        return filters

    def prune(self, device, train_dataloader, eval_dataloader, optimizer, criterion, pruning_percentage=0.1,
              pruning_iterations=9):

        # TODO - remove writer? replace with pickling or graph for pruning?
        # writer = SummaryWriter()
        # init prunner and engines

        # train_dataloader, eval_dataloader = get_dataloaders()

        # _, device = set_cuda()

        # model = NNFactory(prune_cfg["model"])
        # original_state_dict = torch.load('final.pt')
        # new_state_dict = model.state_dict()
        # for k, v in zip(new_state_dict.keys(), original_state_dict.values()):
        #     new_state_dict[k] = v
        # model.load_state_dict(new_state_dict)
        self.to(device)

        num_filters_to_prune = pruning_percentage * total_num_filters(model)

        for iter in range(pruning_iterations):
            # run ranking engine on val dataset
            model.prune = True
            ranker_engine = create_supervised_ranker(model, criterion)
            ranker_engine.run(eval_dataloader, max_epochs=1)

            # prune model and init optimizer, etc
            model.normalize_ranks_per_layer()
            # TODO check for duplicates
            pruning_targets = model.get_prunning_plan(num_filters_to_prune)

            # run training engine on train dataset (and log recovery using val dataset and engine)

            optimizer = get_optimizer(model)
            trainer = create_supervised_trainer(model, optimizer, criterion)
            attach_trainer_events(trainer, model, scheduler=None)

            metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError()}

            val_evaluator = create_supervised_evaluator(model, metrics=metrics)
            attach_eval_events(trainer, val_evaluator, eval_dataloader, writer, "Val")

            trainer.run(train_dataloader, prune_cfg['recovery_epochs'])

