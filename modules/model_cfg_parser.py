from typing import List
from modules.errors import ModuleConfigError

GLOBAL_MODULE_CFGS = ["type", "name", "output"]


def basic_model_cfg_parsing(path: str) -> List[dict]:
    """
    Parses the model configuration file and returns module definitions
    :param path: path to model cfg file
    :return: list of dictionaries, usually passed to NN constructor
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


# TODO - add more checks regarding linear layer
def validate_model_cfg(model_cfg: List[dict]) -> None:
    hyper_params = model_cfg.pop(0)

    height = hyper_params.get("height")
    width = hyper_params.get("width")
    in_channels = hyper_params.get("in_channels")

    if any([module_cfg.get("type") in ["linear", "flatten"] for module_cfg in model_cfg]):
        if not all([height, width, in_channels]):
            raise ModuleConfigError("model has 'linear' or 'flatten' layer but initial input size isn't specified")

        prev_linear = False
        for module_cfg in model_cfg:
            if module_cfg.get("type") in ["linear", "flatten"]:
                prev_linear = True
            elif module_cfg.get("type") not in ["linear", "flatten"] and prev_linear:
                raise ModuleConfigError("'conv2d' or similar layer after 'linear' or 'flatten' is not supported yet")


def calc_receptive_field(model_cfg: List[dict]) -> None:

    total_stride = 1

    layer_sizes = []

    for i, module_cfg in enumerate(model_cfg):
        if module_cfg.get("type") in ["linear", "flatten", "pixel_shuffle"]:
            print("Encountered layer for which receptive field cannot be calculated")
            break

        if module_cfg.get("type") == "route":
            # rf = max([compute_rf(layer_sizes[:layer_idx]) for layer_idx in module_cfg.get("layers")])
            pass

        else:
            stride = int(module_cfg.get("stride"))
            total_stride *= stride
            # TODO - calc receptive field for non square kernel sizes
            kernel_size = int(module_cfg.get("kernel_size"))
            layer_sizes.append((kernel_size, stride))
            layer_rf = compute_rf(layer_sizes)
            print(f"receptive field at layer #{i + 1} is {layer_rf}")


def compute_layer_output_size(module_cfg, in_h, in_w):
    padding = module_cfg.get("padding", 0)
    kernel_size = module_cfg.get("kernel_size", 0)
    stride = module_cfg.get("stride", 1)
    out_h = ((in_h + 2 * padding - kernel_size) // stride) + 1
    out_w = ((in_w + 2 * padding - kernel_size) // stride) + 1
    return out_h, out_w


def compute_single_layer_rf(out, f, s):
    """
    compute receptive field of single layer
    :param out: number of neurons in output
    :param f: kernel size
    :param s: stride
    :return:
    """
    return s * (out - 1) + f


def compute_rf(layers):
    """

    :param layers: list of tuples containing (kernel size, stride)
    :return: receptive field of entire network
    """
    out = 1
    for f, s in reversed(layers):
        out = compute_single_layer_rf(out, f, s)
    return out


def write_pruned_model_cfg(mod_defs, pruning_targets, file_path: str):
    """
    write new model configuration file based on a built model configuration and pruning targets
    :param mod_defs: a list of dictionaries containing model configuration
    :param pruning_targets: dictionary. keys are layer indices and values are list of channel indices to prune
    :param file_path: path for new model configuration file
    :return:
    """
    with open(file_path, 'w') as f:
        for i, mod in enumerate(mod_defs):
            # first, write model type
            type_value = mod.pop('type')
            f.write('[' + type_value + ']')
            for k, v in mod.items():
                if k == 'filters' and i - 1 in pruning_targets.keys():
                    f.write(k + '=' + str(int(v) - len(pruning_targets[i - 1])))
                else:
                    f.write(k + '=' + str(v))
                f.write('\n')
            f.write('\n')
