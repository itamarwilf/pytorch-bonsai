from config import config
from typing import List
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import time
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def speed_testing(model, input_size, iterations=1000, verbose=True):
    """
    Test the inference time of a model on a single input

    Args:
        model: torch.nn.Module
        input_size: tuple of ints representing the model input size (1xCxHxW for example)
        iterations: number of iterations to average over
        verbose (bool): whether or not to print results

    Returns: average inference time of model given the input

    """

    # cuDnn configurations
    cudnn.benchmark = True
    cudnn.deterministic = True

    if verbose:
        print("Speed testing")
    model = model.cuda()
    random_input = torch.randn(*input_size).cuda()

    model.eval()

    time_list = []
    for _ in tqdm.tqdm(range(iterations + 1)):
        torch.cuda.synchronize()
        tic = time.time()
        model(random_input)
        torch.cuda.synchronize()
        time_list.append(time.time()-tic)

    # the first iteration time cost much higher, so exclude the first iteration
    time_list = time_list[1:]
    average_time = sum(time_list) / iterations
    fps = 1 / average_time
    if verbose:
        print(f"Done {iterations} iterations inference !")
        print(f"Average time cost: {average_time}")
        print(f"Frame Per Second: {fps}")
    return average_time


def _make_figure(x, y, x_name, y_name):
    fig, ax = plt.subplots()
    sns.lineplot(x, y, alpha=0.5, ax=ax)
    sns.scatterplot(x, y, ax=ax)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    prune_percent = config["pruning"]["prune_percent"].get()
    for i, _ in enumerate(x):
        ax.annotate(f"{100 - (i * 100 * prune_percent)}%", (x[i], y[i]))
    return fig


def log_performance(metrics: List[dict], writer: SummaryWriter):
    """

    Args:
        metrics:
        writer:

    Returns:

    """

    avg_time = [metric_dict.pop("avg_time") for metric_dict in metrics]
    fps = [1 / t for t in avg_time]
    if writer:
        for metric_name in metrics[0].keys():
            metric_values = [metric_dict[metric_name] for metric_dict in metrics]
            avg_time_fig = _make_figure(avg_time, metric_values, "avg_time", metric_name)
            fps_fig = _make_figure(fps, metric_values, "FPS", metric_name)
            writer.add_figure(f"avg time VS {metric_name}", avg_time_fig)
            writer.add_figure(f"FPS VS {metric_name}", fps_fig)
