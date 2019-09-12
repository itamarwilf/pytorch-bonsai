import torch.nn as nn

from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
import torch.utils.tensorboard._pytorch_graph as pg


class BonsaiParsedModule:
    def __init__(self, type_name: str):
        """
        Represents a single parsed layer from a pytorch model
        Args:
            type_name: name of the module type to be specified on creation.
                       e.g. prunable_conv2d, batch_normalization2d, pixel_shuffle, etc.
        """

        self.type_name = type_name
        self.params = {}
        self.weight_names = []

    def add(self, key, val):
        """
        Add a property to the module.
        Args:
            key: property name, e.g. kernel_size
            val: property value
        """
        if key not in self.params.keys():
            self.params[str(key)] = str(val)

    def add_weight_name(self, weight_name: str):
        """
        Add a weight name to the module, usually lowercase letters and numbers separated by dots.
        Args:
            weight_name: the weight's name. e.g. conv1.conv.bias
        """
        if weight_name not in self.weight_names:
            self.weight_names.append(weight_name)

    def __str__(self):
        output = f'[{self.type_name}]\n'
        for k, v in self.params.items():
            output += f'{k}={v}\n'
        return output

    def __repr__(self):
        return self.__str__()


class BonsaiParsedModel:
    def __init__(self):
        """
        Represents a complete pytorch model after parsing
        """
        self.modules = []

    # Adding modules to the model

    def append_module(self, in_str: str):
        new_module = BonsaiParsedModule(in_str)
        self.modules.append(new_module)

    def insert_module(self, idx: int, in_str: str):
        new_module = BonsaiParsedModule(in_str)
        self.modules.insert(idx, new_module)

    # Adding parameters to the modules

    def insert_param(self, idx: int, key, value):
        if idx >= len(self.modules):
            raise ValueError(f'Module index {idx} out of bounds for list of length {len(self.modules)}')
        else:
            self.modules[idx].add(key, value)

    def insert_weight_name(self, idx: int, weight_name: str):
        if idx >= len(self.modules):
            raise ValueError(f'Module index {idx} out of bounds for list of length {len(self.modules)}')
        else:
            self.modules[idx].add_weight_name(weight_name)

    def add_param(self, key, value):
        self.insert_param(-1, key, value)

    def add_weight_name(self, weight_name: str):
        self.insert_weight_name(-1, weight_name)

    def add_weight_name_by_prefix(self, layer_name: str, prefix: str):
        if prefix is None or len(prefix) == 0:
            param_name = layer_name
        else:
            param_name = prefix + '.' + layer_name

        self.add_weight_name(param_name)

    def get_weight_names(self):
        output = []
        for module in self.modules:
            output.extend(module.weight_names)
        return output

    # Getting layer numbers by the weight names specified beforehand
    # Useful for generating route layers

    def get_layer_by_weight(self, weight_name):
        for i, module in enumerate(self.modules):
            for weight in module.weight_names:
                if weight_name in weight:
                    return i
        raise ValueError('Weight not found!')

    def get_layers_by_weights(self, weight_name_list):
        result_list = []
        for weight_name in weight_name_list:
            result_list.append(self.get_layer_by_weight(weight_name))
        return result_list

    def __len__(self):
        return len(self.modules)

    def summary(self):
        """
        Display a summary of the model, since although printing the complete model may be useful, it is very redundant.
        """
        output = ''
        for i, s in enumerate(self.modules):
            output += f'[{i}][{s.type_name}]'
            output += '\n'
        print(output)

    def __str__(self):
        """
        A complete model display.
        """
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
        """
        Saving the model string to a cfg file
        Args:
            path: the path to be stored
        """
        file_contents = self.__str__()
        with open(path, 'w') as f:
            f.write(file_contents)


def write_2d_params(layer, bonsai_parsed_model):
    """
    Adding 2d parameters conveniently, e.g. kernel size, stride and more.
    Args:
        layer: the layer to be added to
        bonsai_parsed_model:
    Returns:

    """
    for var, name in {layer.kernel_size: 'kernel_size',
                      layer.stride: 'stride',
                      layer.padding: 'padding',
                      layer.dilation: 'dilation'}.items():

        if type(var) == tuple and len(var) == 2:
            bonsai_parsed_model.add_param(name, str(var)[1:-1])  # removing braces
        elif type(var) == int:
            bonsai_parsed_model.add_param(name, var)
        else:
            raise ValueError(f'{name} {var} should be one or two ints')


def parse_simple_model(model, input_shape):
    """
    Parse a model before inserting the complicated routes, such as residual connections, concatenations and more.
    Args:
        model: a pytorch model to be parsed
        input_shape: the shape of the input the model expects

    Returns:
        bonsai_parsed_model: BonsaiParsedModel, a parsed model
    """
    bonsai_parsed_model = BonsaiParsedModel()
    bonsai_parsed_model.append_module('net')
    bonsai_parsed_model.add_param('width', input_shape[-1])
    bonsai_parsed_model.add_param('height', input_shape[-2])
    bonsai_parsed_model.add_param('in_channels', input_shape[-3])

    inner_model_parser(model, bonsai_parsed_model, prefix='')

    return bonsai_parsed_model


def inner_model_parser(model, bonsai_parsed_model, prefix=''):
    """
    Inner function that parses the model recursively
    Args:
        model: the model to be parsed
        bonsai_parsed_model: parsed model so far, which would be edited
        prefix: prefix accumulated down the recursion tree, takes part in generating a weight's name
    """



    for i, named_child in enumerate(model.named_children()):
        layer_name = named_child[0]
        layer = named_child[1]

        # Handling recursive modules, such as nn.Sequential, BasicBlock (ResNet), Up & Down (U-net) etc.
        if len(list(layer.children())) > 0:
            if prefix == '':
                called_prefix = layer_name
            else:
                called_prefix = prefix + '.' + layer_name
            inner_model_parser(layer, bonsai_parsed_model, prefix=called_prefix)

        # handling non linear functions
        elif getattr(type(layer), '__module__') == 'torch.nn.modules.activation':
            bonsai_parsed_model.add_param('activation', repr(type(layer)).split("\'")[1].split('.')[-1])
            if type(layer) == nn.LeakyReLU:
                bonsai_parsed_model.add_param('negative_slope', layer.negative_slope)

        # handling each layer type
        # TODO: add more modules here OR connect to the bonsai factories OR generalize to any module
        elif type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
            type_value = ('prunable_conv2d', 'prunable_deconv2d')[type(layer) == nn.ConvTranspose2d]

            bonsai_parsed_model.append_module(type_value)
            bonsai_parsed_model.add_param('in_channels', layer.in_channels)
            bonsai_parsed_model.add_param('out_channels', layer.out_channels)

            bonsai_parsed_model.add_weight_name_by_prefix(layer_name, prefix)

            write_2d_params(layer, bonsai_parsed_model)

            bonsai_parsed_model.add_param('groups', layer.groups)
            # bonsai_parsed_model.add_param('padding_mode', layer.padding_mode) #TBA

        elif type(layer) in [nn.MaxPool2d, nn.AvgPool2d]:
            type_value = ('maxpool', 'avgpool')[type(layer) == nn.AvgPool2d]
            bonsai_parsed_model.append_module(type_value)

            bonsai_parsed_model.add_weight_name_by_prefix(layer_name, prefix)

            write_2d_params(layer, bonsai_parsed_model)

            bonsai_parsed_model.add_param('ceil_mode', int(layer.ceil_mode))

        elif type(layer) in [nn.PixelShuffle]:
            bonsai_parsed_model.append_module('pixel_shuffle')
            bonsai_parsed_model.add_param('upscale_factor', layer.upscale_factor)

            bonsai_parsed_model.add_weight_name_by_prefix(layer_name, prefix)

        elif type(layer) in [nn.BatchNorm2d]:
            bonsai_parsed_model.append_module('batch_normalization2d')
            bonsai_parsed_model.add_param('num_features', layer.num_features)
            bonsai_parsed_model.add_param('eps', layer.eps)
            bonsai_parsed_model.add_param('momentum', layer.momentum)
            bonsai_parsed_model.add_param('affine', layer.affine)
            bonsai_parsed_model.add_param('track_running_stats', layer.track_running_stats)

            bonsai_parsed_model.add_weight_name_by_prefix(layer_name, prefix)

        elif type(layer) in [nn.AdaptiveAvgPool2d]:
            bonsai_parsed_model.append_module('adaptive_avgpool2d')
            bonsai_parsed_model.add_param('output_size', layer.output_size)

            bonsai_parsed_model.add_weight_name_by_prefix(layer_name, prefix)

        elif type(layer) in [nn.Linear]:
            bonsai_parsed_model.append_module('linear')
            bonsai_parsed_model.add_param('in_features', layer.in_features)
            bonsai_parsed_model.add_param('out_features', layer.out_features)
            bonsai_parsed_model.add_param('bias', layer.bias is not None)

            bonsai_parsed_model.add_weight_name_by_prefix(layer_name, prefix)


def get_node_name(node_str):
    """
    Gets the weight/node name from the full node name supplied by tensorflow's GraphDef
    Args:
        node_str: the full string name

    Returns: the summarized name, separated by dots
    """
    total_str = node_str
    pure_names = [x[:x.index(']')] for x in total_str.split('[') if ']' in x]
    return '.'.join(pure_names)


def find_real_ancestors(curr_weights, predecessors, real_nodes):
    """
    A recursive function that replaces a layer's parents in the graph with real layers,
    instead of intermediate tensor names.
    Args:
        curr_weights: current weights left to be processed
        predecessors: the graph connections
        real_nodes: list of real layer names

    Returns:
        A list of weight names that relate to 'real' layers
    """
    if len(curr_weights) == 0:
        return []
    elif curr_weights[0] in real_nodes:
        return [curr_weights[0]] + find_real_ancestors(curr_weights[1:], predecessors, real_nodes)
    else:
        curr_weights = predecessors[curr_weights[0]] + curr_weights[1:]

        return find_real_ancestors(curr_weights, predecessors, real_nodes)


def get_real_predecessors(predecessors, real_nodes):
    """
    Replace intermediate value predecessors with real ancestor layers that represent graph connectivity truthfully
    Args:
        predecessors: graph connections
        real_nodes: list of real layer names

    Returns:
         A list of weight names that relate to 'real' layers
    """
    real_prevs = {weight: [] for weight, _ in predecessors.items()}
    for weight, prev_weights in predecessors.items():
        real_prevs[weight] = find_real_ancestors(prev_weights, predecessors, real_nodes)
    return real_prevs


def bonsai_parser(model, model_in):
    """
    Full parsing function, handling the route layers
    Args:
        model: pytorch model to be processed
        model_in: model input

    Returns:
        A complete BonsaiParsedModel
    """

    # Simple parsing, without the routing layers
    bonsai_parsed_model = parse_simple_model(model, model_in.size())

    # Getting the graph that represents the underlying network connectivity
    gd = pg.graph(model, args=(model_in,))
    name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(gd[0])

    # Convert node numbers to their short weight name
    graph_layers_to_weights = {get_node_name(k):v for k,v in name_to_seq_num.items()}

    # Route layers
    route_layers = {k: v.op.split('::')[1] for k,v in name_to_node.items() if v.op in ['onnx::Concat', 'onnx::Add']}
    route_weight_names = [get_node_name(x) for x in route_layers]

    # matching node (full name, node shortened name, and previous nodes connected by the graph)
    raw_predecessors = {(k, get_node_name(k)): [get_node_name(x) for x in in_names_list] for k, in_names_list in name_to_input_name.items()}

    # removing duplicates and empty strings
    predecessors = {weight: list(set([x for x in weight_list if len(x) > 0])) for (name, weight), weight_list in raw_predecessors.items()}

    # removing nodes that are intermediate values, they dont correspond to graph layers
    real_nodes = list(set(bonsai_parsed_model.get_weight_names())) + route_weight_names
    real_predecessors = get_real_predecessors(predecessors.copy(), real_nodes)

    # getting the relevant layers for the routing computation
    route_real_predecessors = {k:v for k,v in real_predecessors.items() if k in route_weight_names}
    route_predecessors_layers = {k:bonsai_parsed_model.get_layers_by_weights(v) for k,v in route_real_predecessors.items()}

    # computing the layer number of the generated route layer
    # here we set it to be 1 after the node that is previous to it in the GraphDef graph
    graph_connection = {graph_layers_to_weights[k]: [graph_layers_to_weights[x] for x in v] for k, v in route_real_predecessors.items()}
    prev_index = {k: v.index(int(k)-1) for k, v in graph_connection.items()}
    res_layers = {v[prev_index[graph_layers_to_weights[k]]] + 1: v for k, v in route_predecessors_layers.items()}

    # keeping in mind that layer indices shift when we add new layers
    shifted_values = {k: [val + len([x for x in res_layers if int(x) < int(val)]) for val in v] for k, v in res_layers.items()}
    final_layers = {int(k) + len([x for x in shifted_values if int(x) < int(k)]): v for k, v in shifted_values.items()}

    # adding the layers to the model
    for (k, v), operation in zip(final_layers.items(), route_layers.values()):
        if operation == 'Concat':
            bonsai_parsed_model.insert_module(int(k), 'route')
            bonsai_parsed_model.insert_param(int(k), 'layers', str(v))
        elif operation == 'Add':
            bonsai_parsed_model.insert_module(int(k), 'residual_add')
            bonsai_parsed_model.insert_param(int(k), 'layers', str(v))

    return bonsai_parsed_model
