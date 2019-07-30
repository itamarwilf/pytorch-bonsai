from ignite.engine import Engine
from torch.utils.tensorboard import SummaryWriter
from bonsai.utils.performance_utils import speed_testing
from bonsai.config import config

log_interval = config["logging"]["train_log_interval"].get()


def log_training_loss(engine: Engine, writer: SummaryWriter):
    # iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if engine.state.iteration % config["logging"]["train_log_interval"].get() == 0:
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)


def log_evaluator_metrics(engine: Engine, writer: SummaryWriter):
    metrics = engine.state.metrics
    if writer:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar("val_" + metric_name, metric_value)


def calc_model_speed(engine: Engine, bonsai, input_size):
    metrics = engine.state.metrics
    metrics["avg_time"] = speed_testing(bonsai, input_size, verbose=False)
    bonsai.metrics_list.append(metrics)


def run_evaluator(engine: Engine, evaluator: Engine, dataloader):
    evaluator.run(dataloader, max_epochs=1)
