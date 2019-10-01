import torch


def rchop(string, suffix):
    """
    String manipulation. Remove a suffix if it is present.
    Args:
        string: string to be chopped
        suffix: suffix to be removed

    Returns: shortened string

    """
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string


def lchop(string, prefix):
    """
    String manipulation. Remove a prefix if it is present.
    Args:
        string: string to be chopped
        prefix: prefix to be removed

    Returns: shortened string

    """
    if string.startswith(prefix):
        return string[len(prefix):]
    return string


def recursive_layer_parser(model, name, mem=None, start=True):
    """
    Recursive parser of layers. Used to determine attributes of nn.Conv2d and nn.Linear layers
    Args:
        model: model to iterate over
        name: layer name to store (e.g. Conv, Linear)
        mem: memory storing the located layers
        start: boolean for recursion termination

    Returns:
        mem: list of nn.Module objects of the desired type, defined by name
    """
    if mem is None:
        mem = []
    if start:
        mem = []
    for layer in model.children():
        if len(list(layer.children())) > 0:
            recursive_layer_parser(layer, name, mem, start=False)
        elif name in str(layer):
            mem.append(layer)
    if start:
        return mem


def parse_trace_code(model, model_in):
    """
    Given a model and input format generate command list and its corresponding tensor names required for execution.
    First we split the code to the signature block and the command block.
    Then we split each block into its components - tensors and commands.
    The commands are parsed in order to generate the cfg file.
    Args:
        model: nn.Module, model to read
        model_in: model input, usually torch.Tensor

    Returns:
        command_list: list of strings representing the commands
        input_tensor_names: tensors that the commands use (seldom weight names)
    """
    signature, commands = get_code_string(model, model_in)
    input_tensor_names, command_list = parse_code_blocks(signature, commands)
    return input_tensor_names, command_list


def get_code_string(model, model_in):
    """
    Trace a model and generate a code string.
    Args:
        model: nn.Module, model to read
        model_in: model input, usually torch.Tensor

    Returns:
        signature: string of the format 'def forward( x, tensors names...)'
        commands: code string of commands separated by newlines
    """
    model.eval()
    traced = torch.jit.trace(model, (model_in,), check_trace=False)
    code_string = traced.code
    start_identifier = '-> Tensor:\n'
    declaration_end = code_string.find(start_identifier) + len(start_identifier)
    # delete all lines between declaration end and first command with 'input'

    lines = code_string[declaration_end + 1:].split('\n')
    while 'input' not in lines[0]:
        del lines[0]
    code_string = code_string[:declaration_end] + '\n' + '\n'.join(lines)

    signature = code_string[:declaration_end]
    commands = code_string[declaration_end:]

    return signature, commands


def parse_code_blocks(signature, commands):
    """
    Split the signature block into tensor names, and the commands block to the different commands
    Args:
        signature: signature block code string
        commands: commands block code string

    Returns:
        input_tensor_names: list of tensor names that appear as input
        command_list: list of commands

    """
    signature = [x.strip() for x in signature.split('\n') if len(x.strip()) > 0]
    input_tensor_names = [rchop(slot, ': Tensor,') for slot in signature]
    input_tensor_names[0] = lchop(input_tensor_names[0], 'def forward(')[:-1]
    input_tensor_names[-1] = rchop(input_tensor_names[-1], ': Tensor) -> Tensor:')

    command_list = [x.strip() for x in commands.split('\n') if len(x) > 0]

    return input_tensor_names, command_list


def get_command_set(commands):
    """
    Get list of used commands in the model.
    Use this functions to check if all used functions appear in the supported function list.
    Args:
        commands: list of commands string

    Returns:
        command_set: list of command names' string
    """
    commands = [x.strip() for x in commands.split('\n') if len(x) > 0]
    command_set = list(set([x[x.find('=') + 2:x.find('(')] for x in commands if '=' in x]))
    return command_set


def is_variable_name(x):
    """
    Check if a string is a valid variable name
    Args:
        x: input string

    Returns:
        True if x is a valid variable name
    """
    return (x.isidentifier() and x not in ['False', 'True', 'None']) or '=' in x


def clear_argument_string(input_string):
    """
    Remove redundant string parts from a command
    Args:
        input_string: command string

    Returns:
        shortened string
    """
    redundant_strings = ['[', ']', ')', 'annotate(', 'int(', 'torch.t(', 'torch.size(']
    for red_string in redundant_strings:
        input_string = input_string.replace(red_string, '')
    input_string = input_string.strip()
    return input_string


def split_command(command):
    """
    Split a command into function, arguments and result variable
    Args:
        command: string.

    Returns:
        result, function, argument list
    """
    result, func = None, None

    if '=' in command:
        result = command[:command.find('=')].strip()
        func = command[command.find('=') + 1: command.find('(')].strip()

    elif 'return' in command:
        # command is 'return -tensor-'
        if command.split(' ')[-1].isidentifier():
            return command.split(' ')[-1], None, None

        # command is 'return func(args)'
        result = 'final_output'
        func = command[command.find('return') + len('return') + 1: command.find('(')].strip()

    args = command[command.find('('):].strip()
    args = [clear_argument_string(x) for x in args[1:-1].split(',')]
    args = ','.join(['"' + x + '"' if is_variable_name(x) else x for x in args]).replace(',,', ',')
    return result, func, args


def parse_command(bonsai_parsed_model, conv_list, linear_list, func, prevs, args):
    """
    Parse the command string and add it to the final model
    Args:
        bonsai_parsed_model: parse model representing a cfg file
        conv_list: list of nn.Conv2d modules and their parameters
        linear_list: list of nn.Linear modules and their parameters
        func: function name
        prevs: previous layers
        args: function arguments

    Returns:
        Updated conv_list and linear_list (0 or 1 element may be removed)
    """
    if 'NumToTensor' in func:
        return conv_list, linear_list
    func = func.replace('torch.', '')
    args = eval(args)
    prev_layer_nums = bonsai_parsed_model.get_layers_by_result_dict(prevs)

    if func not in ['cat', 'add_'] and prev_layer_nums[0] is not None:
        if not (len(prev_layer_nums) == 1 and prev_layer_nums[0] == len(bonsai_parsed_model.modules) - 1):
            bonsai_parsed_model.append_module('route')
            relative_layer_nums = [int(x) - len(bonsai_parsed_model.modules) + 1 for x in prev_layer_nums]
            bonsai_parsed_model.add_param('layers', str(relative_layer_nums)[1:-1])

    if func == 'batch_norm':
        bonsai_parsed_model.append_module('batchnorm2d')

    elif func == 'max_pool2d':
        bonsai_parsed_model.append_module('maxpool')
        bonsai_parsed_model.add_2d_param('kernel_size', args[1])
        bonsai_parsed_model.add_2d_param('stride', args[3])

    elif func in ['adaptive_avg_pool2d', 'avg_pool2d']:
        bonsai_parsed_model.append_module('avgpool2d')
        bonsai_parsed_model.add_2d_param('kernel_size', args[1])

    elif func == 'leaky_relu':
        bonsai_parsed_model.add_param('activation', 'LeakyReLU')
        bonsai_parsed_model.add_param('negative_slope', args[1])

    elif func in ['relu_', 'relu']:
        bonsai_parsed_model.add_param('activation', 'ReLU')

    elif func in ['reshape', 'view']:
        bonsai_parsed_model.append_module('flatten')

    elif func == '_convolution':
        if str(args[9]).strip() == 'True':
            bonsai_parsed_model.append_module('prunable_deconv2d')
        else:
            bonsai_parsed_model.append_module('prunable_conv2d')

        bonsai_parsed_model.add_param('out_channels', conv_list[0].out_channels)

        bonsai_parsed_model.add_2d_param('kernel_size', conv_list[0].kernel_size)
        bonsai_parsed_model.add_2d_param('stride', conv_list[0].stride)
        bonsai_parsed_model.add_2d_param('padding', conv_list[0].padding)
        bonsai_parsed_model.add_param('bias', conv_list[0].bias is not None)

        conv_list = conv_list[1:]
    elif func == 'cat':
        bonsai_parsed_model.append_module('route')
        prev_layer_nums = bonsai_parsed_model.get_layers_by_result_dict(prevs)
        relative_layer_nums = [int(x) - len(bonsai_parsed_model.modules) + 1 for x in prev_layer_nums]
        bonsai_parsed_model.add_param('layers', str(relative_layer_nums)[1:-1])

    elif func == 'add_':
        bonsai_parsed_model.append_module('residual_add')
        prev_layer_nums = bonsai_parsed_model.get_layers_by_result_dict(prevs)
        relative_layer_nums = [int(x) - len(bonsai_parsed_model.modules) + 1 for x in prev_layer_nums]
        bonsai_parsed_model.add_param('layers', str(relative_layer_nums)[1:-1])

    elif func == 'pixel_shuffle':
        bonsai_parsed_model.append_module('pixel_shuffle')
        bonsai_parsed_model.add_param('upscale_factor', args[1])

    elif func == 'addmm':
        bonsai_parsed_model.append_module('linear')
        bonsai_parsed_model.add_param('in_features', linear_list[0].in_features)
        bonsai_parsed_model.add_param('out_features', linear_list[0].out_features)
        linear_list = linear_list[1:]
    else:
        raise ValueError('Layer not implemented!' + str(func))

    return conv_list, linear_list


def add_layer(tensor_memory, layer_memory, result, func, args):
    """
    Add layer to memory, used for previous layer computation
    Args:
        tensor_memory: slot memory
        layer_memory: module memory
        result: result slot name
        func: function name
        args: function arguments

    Returns:
        dict of previous layer names
    """
    if 'NumToTensor' in func:
        return
    args = [arg.replace('"', '') for arg in args.split(',')]
    prevs = {}
    for arg in args:
        for k in tensor_memory.keys():
            if arg in tensor_memory[k]:
                if k not in prevs.keys():
                    prevs[k] = []
                prevs[k].append(arg)

    prevs = {k: [var for var in v if var not in tensor_memory[0] or var == 'input'] for k, v in prevs.items()}
    prevs = {k: v for k, v in prevs.items() if len(v) > 0}

    if len(layer_memory) not in tensor_memory.keys():
        tensor_memory[len(layer_memory)] = []
    tensor_memory[len(layer_memory)].append(result)
    layer_memory[len(layer_memory)] = (str(result), str(func), prevs)

    return prevs


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

    def add(self, key, val):
        """
        Add a property to the module.
        Args:
            key: property name, e.g. kernel_size
            val: property value
        """
        self.params[str(key)] = str(val)

    def __str__(self):
        output = f'[{self.type_name}]\n'
        for k, v in self.params.items():
            if k != 'result':
                output += f'{k}={v}\n'
        return output

    def __repr__(self):
        return self.__str__()


class BonsaiParsedModel:
    def __init__(self, input_shape):
        """
        Represents a complete Pytorch model after parsing
        Args:
            input_shape: Iterable, shape of model input
        """
        self.modules = []

        self.append_module('net')
        self.add_param('width', input_shape[-1])
        self.add_param('height', input_shape[-2])
        self.add_param('in_channels', input_shape[-3])
        self.add_param('result', 'input')

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

    def add_param(self, key, value):
        self.insert_param(-1, key, value)

    def __len__(self):
        return len(self.modules)

    def get_layer_by_result(self, result_name):
        for i, module in enumerate(self.modules):
            #         print('module.params', 'result_name', result_name, module.params)
            if 'result' in module.params.keys():
                if module.params['result'] == result_name:
                    return i

    def get_layers_by_result_dict(self, result_dict):
        layer_numbers = []
        for _, layer_list in result_dict.items():
            for layer in layer_list:
                layer_numbers.append(self.get_layer_by_result(layer))
        return layer_numbers

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

    def add_2d_param(self, name, var):
        """
        Adding 2d parameters conveniently, e.g. kernel size, stride and more.
        Args:
            name: parameter name
            var: parameter content
        Returns:

        """

        if type(var) == tuple and len(var) == 2:
            # TODO: implement 2d params (just return var instead of var[0] and act accordingly)
            self.add_param(name, var[0])
        elif type(var) == int:
            self.add_param(name, var)
        else:
            raise ValueError(f'{name} {var} should be one or two ints')


def bonsai_parser(model, model_in):
    """
    Bonsai Parser turns a Pytorch nn.Module into a *.cfg file used to create BonsaiModel objects.
    The .cfg is easy to parse and quite self explainable, allowing the representation of complex model intuitively.
    Args:
        model: nn.Module model
        model_in: model input, usually torch.Tensor

    Returns:
        Parsed model object, you may call save_cfg to save to a convenient path.
    """
    # set up parsed model
    bonsai_parsed_model = BonsaiParsedModel(model_in.shape)

    # trace the model into code string and parse it
    input_tensor_names, command_list = parse_trace_code(model, model_in)

    # set up for iterative parsing
    tensor_memory = {0: input_tensor_names}
    layer_memory = {0: ('input', 'net', {})}

    # get nn.Conv2d and nn.Linear layers from model for parameter parsing
    conv_list = recursive_layer_parser(model, 'Conv')
    linear_list = recursive_layer_parser(model, 'Linear')

    # parse commands iteratively
    for c in command_list:
        res, func, args = split_command(c)
        if func is None:
            break
        prevs = add_layer(tensor_memory, layer_memory, res, func, args)
        conv_list, linear_list = parse_command(bonsai_parsed_model, conv_list, linear_list, func, prevs, args)
        bonsai_parsed_model.add_param('result', res)

    # add final layer to output
    bonsai_parsed_model.add_param('output', 1)
    return bonsai_parsed_model

# if __name__ == '__main__':
#     model = models.resnet18()
#     model_in = torch.zeros((1, 3, 224, 224))
#     bpm = bonsai_parser(model, model_in)
#     print(bpm.summary())
