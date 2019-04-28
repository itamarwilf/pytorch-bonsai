"""
utils
"""


def parse_kernel_size(module_cfg, k, v):
    """
    parses the kernel size string and returns list of integers.
    also works for stride, padding, dilation, etc.
    :param module_cfg:
    :param k:
    :param v:
    :return:
    """
    try:
        new_v = [int(x) for x in v.split(',')]
        assert 1 <= len(new_v) <= 2, f"number of values for {k} should be 1 or 2 for conv2d," \
            f" got {len(new_v)}"
        if len(new_v) == 1:
            new_v *= 2
        module_cfg[k] = new_v
    except ValueError:
        raise ValueError(f"{k} in config of {module_cfg['name']} is {v}, it should be ints separated "
                         f"by commas")
    return module_cfg
