from typing import List


def parse_model_cfg(path: str) -> List[dict]:
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
