import torch
import numpy as np
from ignite.engine import Events
# from ignite.contrib.handlers.tqdm_logger import ProgressBar as Progbar
from ignite.metrics import Accuracy
from modules.bonsai_model import BonsaiModel
from modules.model_cfg_parser import write_pruned_config
from utils.progress_bar import Progbar
from utils.engine_hooks import attach_eval_handlers, attach_train_handlers
from pruning.pruning_engines import create_supervised_trainer, create_supervised_evaluator, \
    create_supervised_ranker
from pruning.abstract_prunners import AbstractPrunner, WeightBasedPrunner
from torch.utils.tensorboard import SummaryWriter


class Bonsai:
    """
    main class of the library, which contains the following components:
    - model: built of simple, prunable pytorch layers
    - prunner: an object in charge of the pruning process

    the interaction between the components is as followed:
    the prunner includes instructions of performing the pruning

    """

    def __init__(self, model_cfg_path: str, prunner = None, normalize=False):
        self.model = BonsaiModel(model_cfg_path, self)
        if prunner is not None and isinstance(prunner(self), AbstractPrunner):
            self.prunner = prunner(self, normalize=normalize)  # type: AbstractPrunner

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def rank(self, rank_dl, criterion):
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

    # TODO - eval should be called at the end of each fine tuning epoch to log recovery
    # TODO - eval should also return validation loss for early stopping of fine tuning
    def finetune(self, train_dl, optimizer, criterion, writer, max_epochs=3):
        print("Recovery")
        self.model.to_rank = False
        finetune_engine = create_supervised_trainer(self.model, optimizer, criterion, self.device)
        pbar = Progbar(train_dl, metrics='none')
        finetune_engine.add_event_handler(Events.ITERATION_COMPLETED, pbar)
        attach_train_handlers(trainer=finetune_engine, writer=writer)
        finetune_engine.run(train_dl, max_epochs=max_epochs)

    # TODO - eval metrics should not be hardcoded, maybe pass metrics as a dict to eval
    def eval(self, eval_dl, criterion, writer):
        print("Evaluation")
        val_evaluator = create_supervised_evaluator(self.model, criterion, device=self.device,
                                                    metrics={"acc": Accuracy(output_transform=lambda output:
                                                    (output[0], output[1]))})

        # TODO - add eval handlers and plotting
        attach_eval_handlers(val_evaluator, writer=writer)

        pbar = Progbar(eval_dl, metrics='acc')
        val_evaluator.add_event_handler(Events.ITERATION_COMPLETED, pbar)

        val_evaluator.run(eval_dl, 1)

    # TODO - add docstring
    def prune_model(self, num_filters_to_prune, iter_num):
        pruning_targets = self.prunner.get_prunning_plan(num_filters_to_prune)
        filters_to_keep = self.prunner.inverse_pruning_targets(pruning_targets)
        out_path = f"pruning_iteration_{iter_num}.cfg"

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

    def run_pruning_loop(self, train_dl, eval_dl, optimizer, criterion, prune_percent=0.1, iterations=9,
                         device="cuda:0"):

        if self.prunner is None:
            raise ValueError("you need a prunner object in the Bonsai model to run pruning")

        # TODO - remove writer? replace with pickling or graph for pruning?
        writer = SummaryWriter()

        assert prune_percent * iterations < 1, f"prune_percent * iterations is bigger than entire model, " \
            f"can't prune that much"

        num_filters_to_prune = int(np.floor(prune_percent * self.model.total_prunable_filters()))

        for iteration in range(iterations):
            # run ranking engine on val dataset
            self.rank(eval_dl, criterion)

            # prune model and init optimizer, etc
            self.prune_model(num_filters_to_prune, iteration)

            # eval performance loss
            self.eval(eval_dl, criterion, writer)

            # run training engine on train dataset (and log recovery using val dataset and engine)
            # TODO - move optimizer to bonsai class?
            model_optimizer = optimizer(self.model.parameters())

            # TODO - fix hardcoded recovery epochs
            self.finetune(train_dl, model_optimizer, criterion, writer)
