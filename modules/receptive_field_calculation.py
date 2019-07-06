from typing import List


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


def compute_rf(layers):
    """

    :param layers: list of tuples containing (kernel size, stride)
    :return: receptive field of entire network
    """
    out = 1
    for f, s in reversed(layers):
        out = compute_single_layer_rf(out, f, s)
    return out


def compute_single_layer_rf(out, f, s):
    """
    compute receptive field of single layer
    :param out: number of neurons in output
    :param f: kernel size
    :param s: stride
    :return:
    """
    return s * (out - 1) + f
