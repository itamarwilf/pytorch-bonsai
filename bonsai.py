import torch
from torch import nn
import numpy as np
from typing import List
from collections import Counter

from modules.abstract_bonsai_classes import Prunable, BonsaiModule
from modules.factories.bonsai_module_factory import BonsaiFactory
from modules.model_cfg_parser import parse_model_cfg

from pruning_process.pruning_engines import create_supervised_trainer, create_supervised_evaluator, \
    create_supervised_ranker
from pruning_process.abstract_prunner import AbstractPrunner


class Bonsai(torch.nn.Module):

    class _Mediator:
        """
        Used to mediate between Bonsai model and its modules, while avoiding circular referencing of torch.nn.Modules
        Also used to give pruning instructions to modules, and thus needs configuring of filter ranking function
        """
        def __init__(self):
            super().__init__()
            self.output_channels: List[int] = []
            self.layer_outputs: List[torch.Tensor] = []
            self.model_output = []

            self.to_prune = False
            self.ranking_function = None

    def __init__(self, cfg_path):
        super(Bonsai, self).__init__()
        self.device = None

        self._mediator = self._Mediator()

        self.last_pruned = None

        self.module_cfgs = parse_model_cfg(cfg_path)  # type: List[dict]
        self.hyperparams = self.module_cfgs.pop(0)  # type: dict
        self.module_list = self._create_bonsai_modules()

    def forward(self, x):

        for i, module in enumerate(self.module_list):
            x = module(x)
            self._mediator.layer_outputs.append(x)
            if module.module_cfg.get("output"):
                self._mediator.model_output.append(x)

        return self._mediator.model_output

    def total_num_filters(self):
        filters = 0
        for module in self.module_list:
            print(type(module))
            # if module["type"] == "convolutional" and "avoid_pruning" not in module.keys():
            if isinstance(module, (Prunable, BonsaiModule)):
                # print("AAA")
                # print(module.module_cfg["out_channels"])
                filters += module.module_cfg.get("out_channels")
        return filters

    def _create_bonsai_modules(self) -> nn.ModuleList:
        module_list = nn.ModuleList()
        # number of input channels for next layer is taken from prev layer output channels (or model input)
        self._mediator.output_channels.append(int(self.hyperparams['in_channels']))
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
            module = module_creator(self._mediator, module_cfg)
            module_list.append(module)
        return module_list

    def total_prunable_filters(self):
        filters = 0
        for module in self.module_list:
            if isinstance(module, Prunable):
                filters += int(module.module_cfg.get("out_channels"))
        return filters

    def register_ranking_function(self, ranking_function: AbstractPrunner):
        assert isinstance(ranking_function, AbstractPrunner)
        self._mediator.ranking_function = ranking_function

    def prune(self, train_dataloader, eval_dataloader, optimizer, criterion, pruning_percentage=0.1,
              pruning_iterations=9, device="cuda:0"):

        if self._mediator.ranking_function is None:
            raise NotImplementedError("you need to add a pruning function to Bonsai model to run pruning")

        # TODO - remove writer? replace with pickling or graph for pruning?
        # writer = SummaryWriter()
        # init prunner and engines

        # train_dataloader, eval_dataloader = get_dataloaders()

        self.to(device)

        num_filters_to_prune = np.floor(pruning_percentage * self.total_num_filters())

        for pruning_iteration in range(pruning_iterations):
            # run ranking engine on val dataset
            self._mediator.to_prune = True
            ranker_engine = create_supervised_ranker(self, criterion)
            ranker_engine.run(eval_dataloader, max_epochs=1)

            # prune model and init optimizer, etc
            model.normalize_ranks_per_layer()
            # TODO check for duplicates
            pruning_targets = model.get_prunning_plan(num_filters_to_prune)

            # run training engine on train dataset (and log recovery using val dataset and engine)

            # optimizer = get_optimizer(model)
            trainer = create_supervised_trainer(model, optimizer, criterion)
            attach_trainer_events(trainer, model, scheduler=None)

            metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError()}

            val_evaluator = create_supervised_evaluator(model, metrics=metrics)
            attach_eval_events(trainer, val_evaluator, eval_dataloader, writer, "Val")

            trainer.run(train_dataloader, prune_cfg['recovery_epochs'])

