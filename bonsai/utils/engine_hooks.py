from ignite.engine import Engine
from torch.utils.tensorboard import SummaryWriter
from bonsai.utils.performance_utils import speed_testing
from bonsai.config import config
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

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


def calc_model_speed(engine: Engine, bonsai, input_size, iterations):
    metrics = engine.state.metrics
    metrics["avg_time"] = speed_testing(bonsai, input_size, verbose=False, iterations=iterations)
    bonsai.metrics_list.append(metrics)


def run_evaluator(engine: Engine, evaluator: Engine, dataloader):
    evaluator.run(dataloader, max_epochs=1)


class BonsaiLoss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).

    """

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: x.shape[0]):
        super(BonsaiLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self._sum = 0
        self._num_examples = 0

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        if isinstance(self._loss_fn, list):
            assert len(y_pred) == len(y) == len(self._loss_fn), \
                "If loss_fn is a list, its length should match the number of outputs and labels"
            average_loss = sum(self._loss_fn[i](y_pred[i], y[i], **kwargs) for i in range(len(self._loss_fn)))
        else:
            average_loss = sum([self._loss_fn(y_pred[i], y, **kwargs) for i in range(len(y_pred))])

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        n = self._batch_size(y)
        self._sum += average_loss.item() * n
        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples
