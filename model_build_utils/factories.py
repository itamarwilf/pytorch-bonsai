from model_build_utils.bonsai_modules import *


class BonsaiFactory:
    """
    used to encapsulate all bonsai module creators, maybe factory isn't the best name?
    """
    _creators: {}

    @staticmethod
    def register_new_creator(module_name: str, module: BonsaiModule):
        if module_name in BonsaiFactory._creators:
            raise ValueError(f"'{module_name}' already used as a key in the module creator dictionary")
        else:
            BonsaiFactory._creators[module_name] = module

    @staticmethod
    def get_creator(module_name: str):
        creator = BonsaiFactory._creators.get(module_name)
        if not creator:
            raise ValueError(f"{module_name} is an unrecognized module creator key")
        return creator


BonsaiFactory.register_new_creator('conv2d', BonsaiConv2d)
BonsaiFactory.register_new_creator('concat', BonsaiConcat)
BonsaiFactory.register_new_creator('deconv2d', BonsaiDeconv2d)
BonsaiFactory.register_new_creator('maxpool', BonsaiMaxpool)
BonsaiFactory.register_new_creator('pixel_shuffle', BonsaiPixelShuffle)


class NonLinearFactory:
    """
    used to encapsulate creators of all bonsai activation functions, maybe factory isn't the best name?
    """
    _creators: {}

    @staticmethod
    def register_new_creator(activation_name: str, activation):
        if activation_name in NonLinearFactory._creators:
            raise ValueError(f"'{activation_name}' already used as a key in the activation function creator dictionary")
        else:
            NonLinearFactory._creators[activation_name] = activation

    @staticmethod
    def get_creator(activation_name: str):
        creator = NonLinearFactory._creators.get(activation_name)
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
