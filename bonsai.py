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


class Bonsai:
    """
    main class of the library, which contains the following components:
    - model: built of simple, prunable pytorch layers
    - prunner: an object in charge of the pruning process

    the interaction between the components is as followed:
    the prunner includes instructions of performing the pruning

    """

    def __init__(self, model_cfg_path: str, prunner: AbstractPrunner = None):
        self.model = BonsaiModel(model_cfg_path, self)
        self.prunner = prunner

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to_prune = False

    def __call__(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def prune(self, train_dl, eval_dl, optimizer, criterion, prune_percent=0.1, iterations=9, device="cuda:0"):

        if self._mediator.ranking_function is None:
            raise NotImplementedError("you need to add a pruning function to Bonsai model to run pruning")

        # TODO - remove writer? replace with pickling or graph for pruning?
        # writer = SummaryWriter()
        # init prunner and engines

        # train_dataloader, eval_dataloader = get_dataloaders()

        self.model.to(device)

        num_filters_to_prune = np.floor(prune_percent * self.model.total_num_filters())

        for pruning_iteration in range(prune_percent):
            # run ranking engine on val dataset
            self.to_prune = True
            ranker_engine = create_supervised_ranker(self, criterion)
            ranker_engine.run(eval_dl, max_epochs=1)

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




class BonsaiModel(torch.nn.Module):

    class _Mediator:
        """
        Used to mediate between Bonsai model and its modules, while avoiding circular referencing of torch.nn.Modules
        """
        def __init__(self, bonsai=None):
            super().__init__()
            self.output_channels: List[int] = []
            self.layer_outputs: List[torch.Tensor] = []
            self.model_output = []

    def __init__(self, cfg_path):
        super(BonsaiModel, self).__init__()
        self.device = None

        self._mediator = self._Mediator()

        self.last_pruned = None

        self.module_cfgs = parse_model_cfg(cfg_path)  # type: List[dict]
        self.hyperparams = self.module_cfgs.pop(0)  # type: dict
        self.module_list = self._create_bonsai_modules()

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, x):

        for i, module in enumerate(self.module_list):
            x = module(x)
            self._mediator.layer_outputs.append(x)
            if module.module_cfg.get("output"):
                self._mediator.model_output.append(x)

        return self._mediator.model_output

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
