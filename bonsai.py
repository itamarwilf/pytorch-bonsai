from config import config
import os
import torch
import numpy as np
from ignite.engine import Events
# from ignite.contrib.handlers.tqdm_logger import ProgressBar as Progbar
from ignite.metrics import Accuracy
from modules.bonsai_model import BonsaiModel
from modules.model_cfg_parser import write_pruned_config
from utils.progress_bar import Progbar
from utils.efficiency_checks import speed_testing
from utils.engine_hooks import attach_eval_handlers, attach_train_handlers
from pruning.pruning_engines import create_supervised_trainer, create_supervised_evaluator, \
    create_supervised_ranker
from pruning.abstract_prunners import AbstractPrunner, WeightBasedPrunner
from pruning.optimizer_factory import optimizer_constructor_from_config
from torch.utils.tensorboard import SummaryWriter


class Bonsai:
    """
    main class of the library, which contains the following components:
    - model: built of simple, prunable pytorch layers
    - prunner: an object in charge of the pruning process

    the interaction between the components is as followed:
    the prunner includes instructions of performing the pruning

    """

    def __init__(self, model_cfg_path: str, prunner=None, normalize=False):
        self.model = BonsaiModel(model_cfg_path, self)
        if prunner is not None and isinstance(prunner(self), AbstractPrunner):
            self.prunner = prunner(self, normalize=normalize)  # type: AbstractPrunner
        elif config["pruning"]["type"].get():
            self.prunner = config["pruning"]["type"].get()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def rank(self, rank_dl, criterion, writer, iter_num):
        print("Ranking")
        self.model.to_rank = True
        self.prunner.set_up()
        if not isinstance(self.prunner, WeightBasedPrunner):
            ranker_engine = create_supervised_ranker(self.model, self.prunner, criterion, device=self.device)
            # add progress bar
            pbar = Progbar(rank_dl, metrics='none')
            ranker_engine.add_event_handler(Events.ITERATION_COMPLETED, pbar)
            # add event hook for accumulation of scores over the dataset
            ranker_engine.add_event_handler(Events.ITERATION_COMPLETED, self.prunner.compute_model_ranks)
            # ranker_engine.add_event_handler(Events.ITERATION_STARTED, self.prunner.reset)
            ranker_engine.run(rank_dl, max_epochs=1)
        else:
            self.prunner.compute_model_ranks()
        if self.prunner.normalize:
            self.prunner.normalize_ranks()

        if writer:
            histogram_name = f"layer ranks - iteration {iter_num}"
            for i, module in self.prunner.prunable_modules_iterator():
                writer.add_histogram(histogram_name, module.ranking, i)

    # TODO - eval should be called at the end of each fine tuning epoch to log recovery
    # TODO - eval should also return validation loss for early stopping of fine tuning
    def finetune(self, train_dl, criterion, writer):
        print("Recovery")
        self.model.to_rank = False
        finetune_epochs = config["pruning"]["finetune_epochs"].get()

        optimizer_constructor = optimizer_constructor_from_config(config)
        optimizer = optimizer_constructor(self.model.parameters())

        finetune_engine = create_supervised_trainer(self.model, optimizer, criterion, self.device)
        pbar = Progbar(train_dl, metrics='none')
        finetune_engine.add_event_handler(Events.ITERATION_COMPLETED, pbar)
        attach_train_handlers(trainer=finetune_engine, writer=writer)
        finetune_engine.run(train_dl, max_epochs=finetune_epochs)

    # TODO - eval metrics should not be hardcoded, maybe pass metrics as a dict to eval
    def eval(self, eval_dl, criterion, writer):
        print("Evaluation")
        val_evaluator = create_supervised_evaluator(self.model, criterion, device=self.device,
                                                    metrics={"acc": Accuracy(output_transform=lambda output:
                                                                             (output[0], output[1]))})

        # TODO - add eval handlers and plotting
        if writer:
            attach_eval_handlers(val_evaluator, writer=writer)

        # TODO - add more verbose debugging
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda x: print(x.state.metrics['acc']))

        pbar = Progbar(eval_dl, metrics='acc')
        val_evaluator.add_event_handler(Events.ITERATION_COMPLETED, pbar)

        val_evaluator.run(eval_dl, 1)

    # TODO - add docstring
    def prune_model(self, num_filters_to_prune, iter_num):
        pruning_targets = self.prunner.get_prunning_plan(num_filters_to_prune)
        filters_to_keep = self.prunner.inverse_pruning_targets(pruning_targets)
        # out_path = f"pruning_iteration_{iter_num}.cfg"
        os.makedirs(config["pruning"]["out_path"].get(), exist_ok=True)
        out_path = os.path.join(config["pruning"]["out_path"].get(), f"pruning_iteration_{iter_num}.cfg")
        write_pruned_config(self.model.full_cfg, out_path, filters_to_keep)

        self.model.propagate_pruning_targets(filters_to_keep)
        new_model = BonsaiModel(out_path, self)

        self.model.cpu()

        final_pruning_targets = self.model.pruning_targets
        for i, (old_module, new_module) in enumerate(zip(self.model.module_list, new_model.module_list)):
            pruned_state_dict = old_module.prune_weights(final_pruning_targets[i + 1], final_pruning_targets[i])
            new_module.load_state_dict(pruned_state_dict)

        self.prunner.reset()
        self.model = new_model

    def run_pruning_loop(self, train_dl, eval_dl, criterion, prune_percent=None, iterations=None):

        if self.prunner is None:
            raise ValueError("you need a prunner object in the Bonsai model to run pruning")

        if prune_percent is None:
            prune_percent = config["pruning"]["prune_percent"].get()
        if iterations is None:
            iterations = config["pruning"]["num_iterations"].get()

        if config["logging"]["use_tensorboard"].get():
            writer = SummaryWriter(log_dir=config["logging"]["logdir"].get())
        else:
            writer = None

        assert prune_percent * iterations < 1, f"prune_percent * iterations is bigger than entire model, " \
            f"can't prune that much"

        num_filters_to_prune = int(np.floor(prune_percent * self.model.total_prunable_filters()))

        for iteration in range(iterations):
            # run ranking engine on val dataset
            self.rank(eval_dl, criterion, writer, iteration)

            # prune model and init optimizer, etc
            self.prune_model(num_filters_to_prune, iteration)

            # eval performance loss
            self.eval(eval_dl, criterion, writer)

            self.finetune(train_dl, criterion, writer)
