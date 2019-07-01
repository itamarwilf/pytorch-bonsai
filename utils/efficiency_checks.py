import torch
from torch.backends import cudnn
import time
import tqdm


def speed_testing(model, input_size, iterations=1000):
    """
    Test the inference time of a model on a single input
    :param model: torch.nn.Module
    :param input_size: tuple of ints representing the model input size (1xCxHxW for example)
    :param iterations: average inference time over num iterations
    :return: average inference time in seconds
    """

    # cuDnn configurations
    cudnn.benchmark = True
    cudnn.deterministic = True

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
    print(f"Done {iterations} iterations inference !")
    print(f"Average time cost: {average_time}")
    print(f"Frame Per Second: {fps}")
    return average_time
