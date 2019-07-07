import torch
import torch.nn as nn
from u_net import UNet, InConv, DoubleConv, OutConv, Down, Up

# Graphviz shenanigans
from graphviz import Digraph

# renaming modules
name_dict = {'UnsafeViewBackward': '-',
             'CloneBackward': '-',
             'PermuteBackward': 'pixel_shuffle',  # this is actually not completely true, and quite problematic
             'AsStridedBackward': '-',
             'MkldnnConvolutionBackward': 'conv2d',
             'LeakyReluBackward0': '-',
             'ThnnConvTranspose2DBackward': 'deconv2d',
             'MaxPool2DWithIndicesBackward': 'maxpool',
             'CatBackward': 'route',
             'ThnnConv2DBackward': 'conv2d',
             'ViewBackward': '-'}

activation_name_dict = {nn.ReLU: 'relu',
                        nn.Sigmoid: 'sigmoid',
                        nn.LeakyReLU: 'leaky_relu',
                        nn.Hardtanh: 'htanh',
                        nn.Tanh: 'tanh'}


# Memory management
# These classes used to handle and parse the graph during the make dot operation

# a single layer
class BackwardModule():
    def __init__(self, module_id: int):
        self.module_id = module_id
        self.module_cnt = -1
        self.module_name = None
        self.parents = []
        self.parent_cnts = []
        self.children_cnts = []

    def add_parent(self, parent_id):
        if parent_id not in self.parents:
            self.parents.append(parent_id)

    def remove_parent(self, parent_id):
        if parent_id in self.parents:
            idx = self.parents.index(parent_id)
            self.parent_cnts.remove(idx)
            self.parents.remove(idx)

    def update_parent_counts(self, parent_list):
        self.parent_cnts = parent_list

    def update_children_counts(self, children_list):
        self.children_cnts.extend(children_list)

    def set_name(self, module_name):
        self.module_name = module_name

    def update_count(self, module_cnt):
        if module_cnt > self.module_cnt:
            self.module_cnt = module_cnt

    def __repr__(self):
        return f'[{self.module_cnt}] {name_dict[self.module_name]}, parents: {self.parent_cnts}'

    def children_str(self):
        return f'[{self.module_cnt}] {name_dict[self.module_name]}, children: {self.children_cnts}'

    def __str__(self):
        return self.__repr__()


# the complete model
class RoutingMemory():
    def __init__(self):
        self.memory = {}

    def create(self, i):
        self.memory[i] = BackwardModule(i)

    def add_parent(self, i, parent_id):
        if i not in self.memory.keys():
            self.create(i)
        self.memory[i].add_parent(parent_id)

    def set_name(self, i, module_name):
        if i not in self.memory.keys():
            self.create(i)
        self.memory[i].set_name(module_name)

    def update_count(self, i, module_cnt):
        if i not in self.memory.keys():
            self.create(i)
        if module_cnt > self.memory[i].module_cnt:
            self.memory[i].module_cnt = module_cnt

    def ids_to_counts(self):
        out_mem = {}
        for mod_id, module in self.memory.items():
            # in addition we reverse the list since we are backwards now becuase of grad_fn
            module.update_parent_counts([len(self.memory) - self.memory[x].module_cnt for x in module.parents])
        for mod_id, module in self.memory.items():
            module.module_cnt = len(self.memory) - module.module_cnt
            out_mem[module.module_cnt] = module
        self.cnt_mem = out_mem

    # from now on we assume ids_to_counts was already applied,
    # and work with self.cnt_mem instead of self.memory
    def delete_module(self, module_cnt):
        assert self.cnt_mem
        deleted_module = self.cnt_mem[module_cnt]

        for module_id, module in self.cnt_mem.items():
            # module.remove_parent(del_id)
            # decrease by 1 all indices after the removed module, and update parents
            module.update_parent_counts([x - 1 if x > module_cnt else x for x in module.parent_cnts])
            if module.module_cnt > module_cnt:
                module.module_cnt -= 1
        # handling the layer before the deleted layer: now it has the parents
        prev_mod = self.cnt_mem[module_cnt - 1]
        prev_mod.update_parent_counts(list(set(prev_mod.parent_cnts + deleted_module.parent_cnts)))
        del self.cnt_mem[module_cnt]
        self.cnt_mem = {mod.module_cnt: mod for mod_cnt, mod in self.cnt_mem.items()}

    # deleting all layers that are redundant
    # including activation functions and unnecessary blocks defined by grad_fn
    # alter name_dict to decide what you may consider redundant
    def delete_all_redundants(self):
        def exists_redundant():
            for key, mod in self.cnt_mem.items():
                if mod.module_name not in name_dict.keys() or name_dict[mod.module_name] == '-':
                    return key
            return None

        red = exists_redundant()
        while red:
            self.delete_module(red)
            red = exists_redundant()

    # getting the reverse funciton: for each node, which nodes are using it
    def parents_to_children(self):
        for mod_cnt, module in self.cnt_mem.items():
            children = [other_mod.module_cnt for other_mod in list(self.cnt_mem.values()) if
                        module.module_cnt in other_mod.parent_cnts]
            module.update_children_counts(children)

    # for display (tostring)
    def __repr__(self):
        output = ''
        if self.cnt_mem is not None:
            mem = self.cnt_mem
        else:
            mem = self.memory
        for _, module in sorted(mem.items()):
            output += str(module)
            output += '\n'
        return output

    def children_view(self):
        output = ''
        for _, module in sorted(self.cnt_mem.items()):
            output += module.children_str()
            output += '\n'
        print(output)

    def routing_layers(self):
        output = {}
        for key, mod in self.cnt_mem.items():
            if name_dict[mod.module_name] == 'route':
                output[mod.module_cnt] = mod.children_cnts
        return output


#Graph walk

def make_dot(model, var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="35,35"))
    seen = set()

    routing_mem = RoutingMemory()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var, counter=0, parent=None, parent_counter=-1):
        if True:  # var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                dot.node(str(id(var)), f'C{counter}\n{size_to_str(u.size())}', fillcolor='lightblue')
            else:
                layer_name = str(type(var).__name__)
                routing_mem.set_name(str(id(var)), layer_name)
                routing_mem.update_count(str(id(var)), counter)
                if parent:
                    routing_mem.add_parent(str(id(var)), str(id(parent)))
                if layer_name == 'CatBackward':
                    dot.node(str(id(var)), layer_name + f' C{counter}', fillcolor='red')
                elif str(type(parent).__name__) == 'CatBackward':
                    dot.node(str(id(var)), layer_name + f' C{counter}', fillcolor='yellow')
                else:
                    dot.node(str(id(var)), layer_name + f' C{counter}')
                counter += 1
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0], counter, parent=var, parent_counter=counter - 1)
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t, counter, parent=var, parent_counter=counter - 1)

    add_nodes(var.grad_fn)
    return routing_mem, dot


# classes that describe a cfg file

# a single layer
class CFGModule():
    def __init__(self, type_name: str):
        self.type_name = type_name
        self.params = {}

    def add(self, key, val):
        if key not in self.params.keys():
            self.params[str(key)] = str(val)

    def __str__(self):
        output = f'[{self.type_name}]\n'
        for k, v in self.params.items():
            output += f'{k}={v}\n'
        return output

    def __repr__(self):
        return self.__str__()


# the complete model
class CFGMemory():
    def __init__(self):
        self.modules = []

    def append_module(self, in_str: str):
        new_module = CFGModule(in_str)
        self.modules.append(new_module)

    def insert_module(self, idx: int, in_str: str):
        new_module = CFGModule(in_str)
        self.modules.insert(idx, new_module)

    def add_param(self, idx: int, key, value):
        if idx >= len(self.modules):
            raise ValueError(f'index {idx} out of bounds for list of length {len(self.modules)}')
        else:
            self.modules[idx].add(key, value)

    def __len__(self):
        return len(self.modules)

    def summary(self):
        output = ''
        for i, s in enumerate(self.modules):
            output += f'[{i}][{s.type_name}]'
            output += '\n'
        print(output)

    def __str__(self):
        output = ''
        for i, s in enumerate(self.modules):
            if i > 0:
                output += f'# layer {i}\n'
            output += str(s)
            output += '\n'
        return output

    def __repr__(self):
        return self.__str__()

    def save_cfg(self, path):
        cfg_str = self.__str__()
        with open(path, 'w') as f:
            f.write(cfg_str)


# from pytorch model to a CFGMemory object
# you may add layers as required

# model to cfg

# saves 2d parameters such as the kernel
def write_2d_params(layer, cfg_mem):
    for var, name in {layer.kernel_size: 'kernel_size',
                      layer.stride: 'stride',
                      layer.padding: 'padding',
                      layer.dilation: 'dilation'}.items():

        if type(var) == tuple and len(var) == 2:
            cfg_mem.add_param(-1, name, str(var)[1:-1])  # removing braces
        elif type(var) == int:
            cfg_mem.add_param(-1, name, var)
        else:
            raise ValueError(f'{name} {var} should be one or two ints')


def is_activation(layer):
    return type(layer) in [nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.Hardtanh, nn.Tanh, nn.ReLU6]


def model_to_cfg(model, input_shape):
    cfg_mem = CFGMemory()
    cfg_mem.append_module('net')
    cfg_mem.add_param(-1, 'width', input_shape[-1])
    cfg_mem.add_param(-1, 'height', input_shape[-2])
    cfg_mem.add_param(-1, 'in_channels', input_shape[-3])

    inner_model_to_cfg(model, cfg_mem, file_path='cfg.txt')

    return cfg_mem


def inner_model_to_cfg(model, cfg_mem, file_path):
    # with open(file_path, 'a') as f:
    for i, layer in enumerate(model.children()):

        if type(layer) in [torch.nn.Sequential, InConv, DoubleConv, OutConv, Down, Up]:  # handling recursive calls
            inner_model_to_cfg(layer, cfg_mem, file_path)

        elif type(layer) in activation_name_dict.keys():  # handling non linearities
            cfg_mem.add_param(-1, 'activation', activation_name_dict[type(layer)])
            if type(layer) == nn.LeakyReLU:
                cfg_mem.add_param(-1, 'negative_slope', layer.negative_slope)

        # handling each layer type
        elif type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
            type_value = ('prunable_conv2d', 'prunable_deconv2d')[type(layer) == nn.ConvTranspose2d]
            cfg_mem.append_module(type_value)
            cfg_mem.add_param(-1, 'in_channels', layer.in_channels)
            cfg_mem.add_param(-1, 'out_channels', layer.out_channels)

            write_2d_params(layer, cfg_mem)

            cfg_mem.add_param(-1, 'groups', layer.groups)
            # cfg_mem.add_param(-1,'padding_mode', layer.padding_mode)



        elif type(layer) in [nn.MaxPool2d, nn.AvgPool2d]:
            type_value = ('maxpool', 'avgpool')[type(layer) == nn.AvgPool2d]
            cfg_mem.append_module(type_value)

            write_2d_params(layer, cfg_mem)

            cfg_mem.add_param(-1, 'ceil_mode', int(layer.ceil_mode))

        elif type(layer) in [nn.PixelShuffle]:
            cfg_mem.append_module('pixel_shuffle')
            cfg_mem.add_param(-1, 'upscale_factor', layer.upscale_factor)


# a complete model
# combines the pytorch parser with the graph parser (to get routes)
def model_to_cfg_w_routing(model, input_shape, model_output):
    cfg_mem = model_to_cfg(model, input_shape) #cfg without routing
    routing_mem, graph = make_dot(model, model_output)
    routing_mem.ids_to_counts()
    routing_mem.delete_all_redundants()
    routing_mem.parents_to_children()
    #routing_mem.children_view()
    routing_layers = routing_mem.routing_layers()
    for k,v in sorted(routing_layers.items()):
        cfg_mem.insert_module(int(k), 'route')
        cfg_mem.add_param(int(k), 'layers', str(v))
    return cfg_mem
