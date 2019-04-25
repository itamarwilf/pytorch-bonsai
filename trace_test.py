from u_net import UNet
from torchviz import make_dot, make_dot_from_trace
import torch


model = UNet(4, 12)

temp = torch.rand(1,4,256, 256)

res = model(temp)

named_params = dict(model.named_parameters())

make_dot(res, params=named_params)

trace = torch.jit.trace(model, (temp,))

with torch.onnx.set_training(model, False):
    trace, _ = torch.jit.get_trace_graph(model, args=(temp,))

make_dot_from_trace(trace)
