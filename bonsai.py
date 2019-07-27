from config import config
import os
from typing import Callable
import torch
import numpy as np
from ignite.engine import Events
# from ignite.contrib.handlers.tqdm_logger import ProgressBar as Progbar
from ignite.metrics import Loss, Metric
from ignite.handlers import EarlyStopping, TerminateOnNan, ModelCheckpoint
from modules.bonsai_model import BonsaiModel
from modules.model_cfg_parser import write_pruned_config
from utils.progress_bar import Progbar
from utils.performance_utils import log_performance
from utils.engine_hooks import log_training_loss, log_evaluator_metrics, calc_model_speed, run_evaluator
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
        self.writer = None

        # metrics_list is used to store eval values across pruning procedure
        self.metrics_list = []

        self._eval_handlers = []
        self._finetune_handlers = []
        # _metrics is used to store the metrics the user wants to calculate besides the loss
        self._metrics = {}

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _rank(self, rank_dl, criterion, iter_num):
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

        self.prunner.equalize_elementwise()

        if self.writer:
            histogram_name = f"layer ranks - iteration {iter_num}"
            for i, module in self.prunner.prunable_modules_iterator():
                self.writer.add_histogram(histogram_name, module.ranking, i)

    # TODO - option for eval being called at the end of each fine tuning epoch to log recovery
    def _finetune(self, train_dl, val_dl, criterion, iter_num):
        print("Recovery")
        self.model.to_rank = False
        finetune_epochs = config["pruning"]["finetune_epochs"].get()

        optimizer_constructor = optimizer_constructor_from_config(config)
        optimizer = optimizer_constructor(self.model.parameters())

        finetune_engine = create_supervised_trainer(self.model, optimizer, criterion, self.device)
        # progress bar
        pbar = Progbar(train_dl, metrics='none')
        finetune_engine.add_event_handler(Events.ITERATION_COMPLETED, pbar)

        # log training loss
        if self.writer:
            finetune_engine.add_event_handler(Events.ITERATION_COMPLETED,
                                              lambda engine: log_training_loss(engine, self.writer))

        # terminate on Nan
        finetune_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        # model checkpoints
        checkpoint = ModelCheckpoint(config["pruning"]["out_path"].get(), require_empty=False,
                                     filename_prefix=f"pruning_iteration_{iter_num}", save_interval=1)
        finetune_engine.add_event_handler(Events.COMPLETED, checkpoint, {"weights": self.model.cpu()})

        # add early stopping
        validation_evaluator = create_supervised_evaluator(self.model, device=self.device,
                                                           metrics=self._metrics)

        if config["pruning"]["early_stopping"].get():
            def _score_function(evaluator):
                return -evaluator.state.metrics["loss"]
            early_stop = EarlyStopping(config["pruning"]["patience"].get(), _score_function, finetune_engine)
            validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stop)

        finetune_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda engine:
                                          run_evaluator(engine, validation_evaluator, val_dl))

        for handler_dict in self._finetune_handlers:
            finetune_engine.add_event_handler(handler_dict["event_name"], handler_dict["handler"],
                                              *handler_dict["args"], **handler_dict["kwargs"])

        # run training engine
        finetune_engine.run(train_dl, max_epochs=finetune_epochs)

    def _eval(self, eval_dl):
        print("Evaluation")

        evaluator = create_supervised_evaluator(self.model, device=self.device,
                                                metrics=self._metrics)

        # TODO - add logger
        if self.writer:
            evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                        lambda engine: log_evaluator_metrics(engine, self.writer))

        input_size = [1] + list(eval_dl.dataset[0][0].size())
        evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    lambda engine: calc_model_speed(engine, self, input_size))

        pbar = Progbar(eval_dl, None)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, pbar)

        for handler_dict in self._eval_handlers:
            evaluator.add_event_handler(handler_dict["event_name"], handler_dict["handler"],
                                        *handler_dict["args"], **handler_dict["kwargs"])

        evaluator.run(eval_dl, 1)

    # TODO - add docstring
    def _prune_model(self, num_filters_to_prune, iter_num):
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

    def run_pruning(self, train_dl, val_dl, test_dl, criterion, prune_percent=None, iterations=None):

        if self.prunner is None:
            raise ValueError("you need a prunner object in the Bonsai model to run pruning")
        self.metrics_list = []
        self._metrics["loss"] = Loss(criterion)

        if prune_percent is None:
            prune_percent = config["pruning"]["prune_percent"].get()
        if iterations is None:
            iterations = config["pruning"]["num_iterations"].get()
        assert prune_percent * iterations < 1, f"prune_percent * iterations is bigger than entire model, " \
            f"can't prune that much"
        num_filters_to_prune = int(np.floor(prune_percent * self.model.total_prunable_filters()))

        if config["logging"]["use_tensorboard"].get():
            self.writer = SummaryWriter(log_dir=config["logging"]["logdir"].get())

        self._eval(test_dl)

        for iteration in range(1, iterations+1):
            print(iteration)
            # run ranking engine on val dataset
            self._rank(val_dl, criterion, iteration)

            # prune model and init optimizer, etc
            self._prune_model(num_filters_to_prune, iteration)

            self._finetune(train_dl, val_dl, criterion, iteration)

            # eval performance loss
            self._eval(test_dl)

        log_performance(self.metrics_list, self.writer)

    def attach_handler_to_eval(self, event: Events, handler: Callable, *args, **kwargs):
        self._eval_handlers.append({"event_name": event, "handler": handler, "args": args, "kwargs": kwargs})

    def attach_handler_to_finetune(self, event: Events, handler: Callable, *args, **kwargs):
        self._finetune_handlers.append({"event_name": event, "handler": handler, "args": args, "kwargs": kwargs})

    def attach_metric_to_eval(self, metric_name: str, metric: Metric):
        self._metrics[metric_name] = metric
