"""
Utils for reading, parsing and writing model configuration files. Used for writing the pruned models instructions
"""

from typing import List
from bonsai.modules.errors import ModuleConfigError


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
            module_defs[-1]['type'] = line[1:-1].replace(" ", "")
        else:
            key, value = line.split("=")
            value = value.replace(" ", "")
            value = _convert_module_cfg_value(value)
            module_defs[-1][key.replace(" ", "")] = value
    return module_defs


def _convert_module_cfg_value(val: str):
    """
    change val into the correct data type
    :param val: value to convert
    :return: str, int, float, or a list containing these types
    """
    val = val.split(",")
    try:
        val = [int(x) for x in val]
    except ValueError:
        try:
            val = [float(x) for x in val]
        except ValueError:
            pass
    if len(val) == 1:
        return val[0]
    try:
        if val == "False":
            return False
        elif val == "True":
            return True
    except ValueError:
        raise ValueError
    return val


# TODO - add more checks regarding linear layer
def validate_model_cfg(model_cfg: List[dict]) -> None:
    """
    validates that the given model configuration can be constructed an is not missing information
    Args:
        model_cfg: list of dictionaries containing modules parameters

    Returns: None
    """
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


def write_pruned_config(full_cfg: List[dict], output_path: str, pruning_targets: dict):
    """
    After each pruning stage, write the pruned model configuration so it could be used for next iteration or by user.
    Args:
        full_cfg: The old model configuration.
        output_path: where to write the new model's configuration
        pruning_targets: dictionary with the current iterations pruning targets.

    Returns: None
    """
    write_layer_num = False
    with open(output_path, 'w')as f:
        for i, block in enumerate(full_cfg):
            if write_layer_num:
                f.write(f"#{i - 1}\n")
            for k, v in block.items():
                if k == 'type':
                    f.write('[' + v + ']')
                elif (k == 'out_channels' or k == 'out_features') and i - 1 in pruning_targets.keys():
                    f.write(k + '=' + str(len(pruning_targets[i - 1])))
                else:
                    if isinstance(v, list):
                        f.write(k + "=" + ",".join(str(x) for x in v))
                    else:
                        f.write(k + '=' + str(v))
                f.write('\n')
            f.write('\n')
            write_layer_num = True
