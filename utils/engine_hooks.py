from ignite.handlers import EarlyStopping, TerminateOnNan
from ignite.engine import Engine, Events
from torch.utils.tensorboard import SummaryWriter
from utils.performance_utils import speed_testing
from config import config


def attach_train_handlers(trainer: Engine, writer: SummaryWriter):

    log_interval = config["logging"]["train_log_interval"].get()

    def log_training_loss(engine):
        # iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if engine.state.iteration % log_interval == 0:
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
    if writer:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)


def attach_eval_handlers(evaluator: Engine, writer: SummaryWriter, bonsai, input_size):

    def log_eval_metrics(engine):
        metrics = engine.state.metrics
        if writer:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar("val_" + metric_name, metric_value)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_eval_metrics)

    def test_model_speed(engine):
        metrics = engine.state.metrics
        metrics["avg_time"] = speed_testing(bonsai.model, input_size, verbose=False)
        bonsai.metrics_list.append(metrics)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, test_model_speed)
