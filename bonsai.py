import torch
from torch import nn
import numpy as np
from typing import List
from collections import Counter
from ignite.engine import Events

from modules.abstract_bonsai_classes import Prunable, BonsaiModule
from modules.factories.bonsai_module_factory import BonsaiFactory
from modules.model_cfg_parser import parse_model_cfg
from pruning.pruning_engines import create_supervised_trainer, create_supervised_evaluator, \
    create_supervised_ranker
from pruning.abstract_prunners import AbstractPrunner, WeightBasedPrunner


class Bonsai:
    """
    main class of the library, which contains the following components:
    - model: built of simple, prunable pytorch layers
    - prunner: an object in charge of the pruning process

    the interaction between the components is as followed:
    the prunner includes instructions of performing the pruning

    """

    def __init__(self, model_cfg_path: str, prunner=None):
        self.model = BonsaiModel(model_cfg_path, self)
        if prunner is not None and isinstance(prunner(self), AbstractPrunner):
            self.prunner = prunner(self)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def rank(self, rank_dl, criterion):
        self.model.to_rank = True
        self.prunner.set_up()
        if not isinstance(self.prunner, WeightBasedPrunner):
            ranker_engine = create_supervised_ranker(self.model, self.prunner, criterion)
            # add event hook for accumulation of scores over the dataset
            ranker_engine.add_event_handler(Events.ITERATION_COMPLETED, self.prunner.compute_model_ranks)
            # add event hook for accumulation
            # ranker_engine.add_event_handler(Events.ITERATION_STARTED, self.prunner.reset)
            ranker_engine.run(rank_dl, max_epochs=1)
        else:
            self.prunner.compute_model_ranks()
        if self.prunner.normalize:
            self.prunner.normalize_ranks()

    def finetune(self, train_dl, optimizer, criterion, max_epochs=10):
        self.model.to_rank = False
        finetune_engine = create_supervised_trainer(self.model, optimizer, criterion, self.device)
        finetune_engine.run(train_dl, max_epochs=max_epochs)

    def prune(self, train_dl, eval_dl, optimizer, criterion, prune_percent=0.1, iterations=9, device="cuda:0"):

        if self.prunner is None:
            raise ValueError("you need a prunner object in the Bonsai model to run pruning")

        # TODO - remove writer? replace with pickling or graph for pruning?
        # writer = SummaryWriter()
        # init prunner and engines

        # train_dataloader, eval_dataloader = get_dataloaders()

        self.model.to(device)

        num_filters_to_prune = np.floor(prune_percent * self.model.total_num_filters())

        for iteration in range(iterations):
            # run ranking engine on val dataset
            self.model.to_rank = True
            if not isinstance(self.prunner, WeightBasedPrunner):
                ranker_engine = create_supervised_ranker(self.model, self.prunner, criterion)
                ranker_engine.run(eval_dl, max_epochs=1)

            # prune model and init optimizer, etc
            # TODO check for duplicates
            pruning_targets = self.prunner.get_prunning_plan(num_filters_to_prune)

            # run training engine on train dataset (and log recovery using val dataset and engine)

            # optimizer = get_optimizer(model)
            # trainer = create_supervised_trainer(model, optimizer, criterion)
            # attach_trainer_events(trainer, model, scheduler=None)
            #
            # metrics = {"L1": MeanAbsoluteError(), "L2": MeanSquaredError()}
            #
            # val_evaluator = create_supervised_evaluator(model, metrics=metrics)
            # attach_eval_events(trainer, val_evaluator, eval_dataloader, writer, "Val")
            #
            # trainer.run(train_dataloader, prune_cfg['recovery_epochs'])


class BonsaiModel(torch.nn.Module):

    class _Mediator:
        """
        Used to mediate between Bonsai model and its modules, while avoiding circular referencing of torch.nn.Modules
        """
        def __init__(self, model=None):
            super().__init__()
            self.model = model

        def __getattribute__(self, item):
            try:
                return super().__getattribute__(item)
            except AttributeError:
                return self.model.__getattribute__(item)

    def __init__(self, cfg_path, bonsai):
        super(BonsaiModel, self).__init__()
        self.bonsai = bonsai
        self.device = None

        self._mediator = self._Mediator(self)
        self.output_channels: List[int] = []
        self.feature_map_size: List[int] = []
        self.kernel_sizes: List[int] = []
        self.strides: List[int] = []

        self.layer_outputs: List[torch.Tensor] = []
        self.model_output = []

        self.last_pruned = None
        self.to_rank = False

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
                self.model_output.append(x)

        return self.model_output

    # TODO needs design for nn.Linear construction, including feature map size, strides, kernel sizes, etc.
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
            module = module_creator(self._mediator, module_cfg)
            # TODO take parsed cfg from module and accumulate strides, kernels, and calc feature map stride if possible
            # TODO have a list of layers that allow for receptive field calculations (fuck linear layers)
            # parsed_cfg = module.module_cfg
            # n_out = np.floor(n_in + 2 * padding - kernel_size / stride) -1 #activation map size
            # jump_out = jump_in * stride #jump in features (equivalent to the accumulated stride)
            # r_out = r_in + (kernel_size-1) * j_in
            # start_out = start_in + ( (kernel_size - 1) / 2 - p) * j_in # note: can be discarded

            module_list.append(module)
        return module_list

    def total_prunable_filters(self):
        filters = 0
        for module in self.module_list:
            if isinstance(module, Prunable):
                filters += int(module.module_cfg.get("out_channels"))
        return filters
