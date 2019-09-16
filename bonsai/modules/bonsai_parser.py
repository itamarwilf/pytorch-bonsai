import torch
from torchvision import models


def layer_parser(model, name):
    l = []
    inner_layer_parser(model, l, name)
    return l


def inner_layer_parser(model, l, name):
    for layer in model.children():
        if len(list(layer.children())) > 0:
            inner_layer_parser(layer, l, name)
        elif name in str(layer):
            l.append(layer)


def rchop(thestring, ending):
    if thestring.endswith(ending):
        return thestring[:-len(ending)]
    return thestring


def lchop(thestring, start):
    if thestring.startswith(start):
        return thestring[len(start):]
    return thestring


def parse_declaration(code_string, declaration_end):
    declaration = code_string[:declaration_end]
    declaration = [x.strip() for x in declaration.split('\n') if len(x.strip()) > 0]
    slot_names = [rchop(slot, ': Tensor,') for slot in declaration]
    slot_names[0] = lchop(slot_names[0], 'def forward(')[:-1]
    slot_names[-1] = rchop(slot_names[-1], ': Tensor) -> Tensor:')

    commands = code_string[declaration_end:]
    commands = [x.strip() for x in commands.split('\n') if len(x) > 0]

    return slot_names, commands


def get_command_set(code_string, declaration_end):
    commands = code_string[declaration_end:]
    commands = [x.strip() for x in commands.split('\n') if len(x) > 0]
    command_set = list(set([x[x.find('=') + 2:x.find('(')] for x in commands if '=' in x]))
    return command_set


def evalable(x):
    try:
        eval(x)
    except:
        return False
    return True


def cond(x):
    return (x.isidentifier() and x not in ['False', 'True', 'None']) or '=' in x


def split_command(c):

    if '=' in c:
        result = c[:c.find('=')].strip()
        func = c[c.find('=') + 1: c.find('(')].strip()
        args = c[c.find('('):].strip()

        args = [
            x.replace('[', '').replace(']', '').replace(')', '').replace('annotate(', '')
                .replace('int(', '').replace('torch.t(', '').replace('torch.size(', '')
                .strip() for x in args[1:-1].split(',')]
        evalable_args = ','.join(['"' + x + '"' if cond(x) else x for x in args]).replace(',,', ',')
        return result, func, evalable_args

    elif 'return' in c:
        if c.split(' ')[-1].isidentifier():
            return c.split(' ')[-1], None, None

        result = 'final_output'
        func = c[c.find('return') + len('return') + 1: c.find('(')].strip()
        args = c[c.find('('):].strip()

        args = [
            x.replace('[', '').replace(']', '').replace(')', '').replace('int(', '').replace('torch.t(', '').replace(
                'torch.size(', '').strip() for x in args[1:-1].split(',')]
        evalable_args = ','.join(['"' + x + '"' if cond(x) else x for x in args])
        return result, func, evalable_args


def init_layer():
    return ('input', 'net', None)


class BonsaiMemory:
    def __init__(self, tensor_list=[]):
        self.tensor_memory = {0: tensor_list}
        self.layer_memory = {0: init_layer()}

    def add_route(self, layers):
        self.layer_memory.append('route', layers)

    def add(self, result, func, args):
        if 'NumToTensor' in func:
            return
        args = [arg.replace('"', '') for arg in args.split(',')]
        prevs = {}
        for arg in args:
            for k in self.tensor_memory.keys():
                if arg in self.tensor_memory[k]:
                    if k not in prevs.keys():
                        prevs[k] = []
                    prevs[k].append(arg)

        prevs = {k: [var for var in v if var not in self.tensor_memory[0] or var == 'input'] for k, v in prevs.items()}
        prevs = {k: v for k, v in prevs.items() if len(v) > 0}

        if len(self.layer_memory) not in self.tensor_memory.keys():
            self.tensor_memory[len(self.layer_memory)] = []
        self.tensor_memory[len(self.layer_memory)].append(result)
        self.layer_memory[len(self.layer_memory)] = (result, func, prevs)

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
            layer: the layer to be added to
            bonsai_parsed_model:
        Returns:

        """

        if type(var) == tuple and len(var) == 2:
            #TODO: implement 2d params (just return var instead of var[0] and act accordingly)
            self.add_param(name, var[0])  # removing braces
        elif type(var) == int:
            self.add_param(name, var)
        else:
            raise ValueError(f'{name} {var} should be one or two ints')



def bonsai_parser(model, model_in):
    model.eval()
    traced = torch.jit.trace(model, (model_in,), check_trace=False)
    code_string = traced.code
    start_identifier = '-> Tensor:\n'
    declaration_end = code_string.find(start_identifier) + len(start_identifier)
    # delete all lines between declaration end and first command with 'input'

    lines = code_string[declaration_end+1:].split('\n')
    while 'input' not in lines[0]:
        del lines[0]
    code_string = code_string[:declaration_end] + '\n' + '\n'.join(lines)

    input_shape = model_in.shape
    bonsai_parsed_model = BonsaiParsedModel()
    bonsai_parsed_model.append_module('net')
    bonsai_parsed_model.add_param('width', input_shape[-1])
    bonsai_parsed_model.add_param('height', input_shape[-2])
    bonsai_parsed_model.add_param('in_channels', input_shape[-3])
    bonsai_parsed_model.add_param('result', 'input')

    conv_list = layer_parser(model, 'Conv')
    linear_list = layer_parser(model, 'Linear')
    slot_names, commands = parse_declaration(code_string, declaration_end)
    bm = BonsaiMemory(slot_names)
    for slot in slot_names:
        exec(slot + ' = torch.zeros((1,1))')

    for c in commands:
        # print('ccc', c)

        res, func, args = split_command(c)
        # print('tag', res, func, args )
        if func is None:
            break
        exec(res + ' = torch.zeros((1,1))')
        prevs = bm.add(res, func, args)
        conv_list, linear_list = parse_command(bonsai_parsed_model, conv_list, linear_list, res, func, prevs, args)
        bonsai_parsed_model.add_param('result', res)

    bonsai_parsed_model.add_param('output', 1)
    return bonsai_parsed_model


def parse_command(bonsai_parsed_model, conv_list, linear_list, res, func, prevs, args):
    if 'NumToTensor' in func:
        return conv_list, linear_list
    func = func.replace('torch.', '')
    args = eval(args)
    prev_layer_nums = bonsai_parsed_model.get_layers_by_result_dict(prevs)

    #   print('prev_layer_nums', prev_layer_nums)

    if func not in ['cat', 'add_'] and prev_layer_nums[0] is not None:
        if not (len(prev_layer_nums) == 1 and prev_layer_nums[0] == len(
                bonsai_parsed_model.modules) - 1):  # TODO: add a case we route one layer but still routing
            bonsai_parsed_model.append_module('route')
            relative_layer_nums = [int(x) - len(bonsai_parsed_model.modules) + 1 for x in prev_layer_nums]
            bonsai_parsed_model.add_param('layers', str(relative_layer_nums)[1:-1])

    if func == 'batch_norm':
        bonsai_parsed_model.append_module('batchnorm2d')

    elif func == 'max_pool2d':
        bonsai_parsed_model.append_module('maxpool')
        bonsai_parsed_model.add_2d_param('kernel_size', args[1])
        bonsai_parsed_model.add_2d_param('stride', args[3])

    elif func in ['adaptive_avg_pool2d','avg_pool2d']:
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
        raise ValueError('Layer not implemented!'+str(func))

    return conv_list, linear_list


if __name__ == '__main__':
    model = models.resnet18()
    model_in = torch.zeros((1, 3, 224, 224))
    bpm = bonsai_parser(model, model_in)
    print(bpm.summary())