from inspect import getfullargspec


def call_constructor_with_cfg(constructor, cfg: dict):
    """
    utility functions for constructors, filters cfg based on cfg args and kwargs and creates instance
    :param constructor: function, should be used for class instance creation
    :param cfg: dict, containing keys for constructor
    :return: instance of class based on constructor and appropriate cfg
    """
    kwargs = getfullargspec(constructor).args
    constructor_cfg = {k: v for (k, v) in cfg.items() if k in kwargs}
    # print(constructor_cfg)
    return constructor(**constructor_cfg)
