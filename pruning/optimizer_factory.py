from torch import optim
from confuse import Configuration
from utils.construct_utils import call_constructor_with_cfg


def optimizer_constructor_from_config(config: Configuration):
    """
    wrapper function returning optimizer constructor based on configuration for every model parameters
    Args:
        config: configuration holding optimizer params

    Returns:
        constructor for optimizer given the model params
    """
    optimizer_dict = config["optimizer"].get()
    optimizer_type = optimizer_dict.get("type")
    try:
        optimizer_constructor = getattr(optim, optimizer_type)
    except AttributeError:
        raise ValueError(f"optimizer type '{optimizer_type}' is not a recognized torch.Optimizer")

    def model_optimizer(params):
        optimizer_dict["params"] = params
        return call_constructor_with_cfg(optimizer_constructor, optimizer_dict)

    return model_optimizer
