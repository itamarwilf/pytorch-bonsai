from torch import nn


class NonLinearFactory:
    """
    used to encapsulate creators of all bonsai activation functions, maybe factory isn't the best name?
    """
    _creators = {}

    @classmethod
    def register_new_creator(cls, activation_name: str, activation):
        if activation_name in cls._creators:
            raise ValueError(f"'{activation_name}' already used as a key in the activation function creator dictionary")
        else:
            cls._creators[activation_name] = activation

    @classmethod
    def get_creator(cls, activation_name: str):
        creator = cls._creators.get(activation_name)
        if not creator:
            raise ValueError(f"{activation_name} is an unrecognized activation function")
        return creator


# region non-linear registration
NonLinearFactory.register_new_creator("relu", nn.ReLU)
NonLinearFactory.register_new_creator("sigmoid", nn.Sigmoid)
NonLinearFactory.register_new_creator("htanh", nn.Hardtanh)
NonLinearFactory.register_new_creator("tanhh", nn.Tanh)
NonLinearFactory.register_new_creator("leaky_relu", nn.LeakyReLU)
# endregion
