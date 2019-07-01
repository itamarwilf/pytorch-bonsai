from ignite.handlers import EarlyStopping, TerminateOnNan
from ignite.engine import Engine, Events
from torch.utils.tensorboard import SummaryWriter
import config

config.Config("../config_default.yaml")


def attach_train_handlers(trainer: Engine, writer: SummaryWriter):

    log_interval = config["logging"]["train_log_interval"].get()

    def log_training_loss(engine):
        # iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if engine.state.iteration % log_interval == 0:
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    # def log_training_results(engine, evaluator):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll = metrics['nll']
    #     print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #           .format(engine.state.epoch, avg_accuracy, avg_nll))
    #     writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
    #     writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)
    #
    #
    # def log_validation_results(engine, evaluator):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll = metrics['nll']
    #     print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #           .format(engine.state.epoch, avg_accuracy, avg_nll))
    #     writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
    #     writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)


def attach_eval_handlers(evaluator: Engine, writer: SummaryWriter):

    def log_eval_metrics(engine):
        metrics = engine.state.metrics
        for metric_name, metric_value in metrics.items():
            writer.add_scalar("val_" + metric_name, metric_value)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_eval_metrics)
