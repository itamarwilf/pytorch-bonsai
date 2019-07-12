from torch import nn
from utils.construct_utils import call_constructor_with_cfg


def construct_activation_from_config(config: dict):
    """
    wrapper function returning activation module based on bonsai module config
    Args:
        config: configuration holding activation params

    Returns:
        activation module
    """
    activation_type = config.get("activation")
    try:
        activation_constructor = getattr(nn, activation_type)
    except AttributeError:
        raise ValueError(f"activation type '{activation_type}' is not a recognized torch.nn Module")
    return call_constructor_with_cfg(activation_constructor, config)
