import torch
from torch.backends import cudnn
import time
import tqdm


def speed_testing(model, input_size, iterations=1000):

    # cuDnn configurations
    cudnn.benchmark = True
    cudnn.deterministic = True

    name = "Bonsai"
    print("     + {} Speed testing... ...".format(name))
    model = model.cuda()
    random_input = torch.randn(*input_size).cuda()

    model.eval()

    time_list = []
    for _ in tqdm.tqdm(range(iterations)):
        torch.cuda.synchronize()
        tic = time.time()
        model(random_input)
        torch.cuda.synchronize()
        # the first iteration time cost much higher, so exclude the first iteration
        # print(time.time()-tic)
        time_list.append(time.time()-tic)
    time_list = time_list[1:]
    print("     + Done 10000 iterations inference !")
    print("     + Total time cost: {}s".format(sum(time_list)))
    print("     + Average time cost: {}s".format(sum(time_list)/10000))
    print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/10000)))
