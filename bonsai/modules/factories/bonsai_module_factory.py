from bonsai.modules.bonsai_modules import *


class BonsaiFactory:
    """
    used to encapsulate all bonsai module creators, maybe factory isn't the best name?
    """
    _creators = {}

    @classmethod
    def register_new_creator(cls, module_name: str, module: BonsaiModule):
        if module_name in BonsaiFactory._creators:
            raise ValueError(f"'{module_name}' already used as a key in the module creator dictionary")
        else:
            BonsaiFactory._creators[module_name] = module

    @classmethod
    def get_creator(cls, module_name: str):
        creator = BonsaiFactory._creators.get(module_name)
        if not creator:
            raise ValueError(f"{module_name} is an unrecognized module creator key")
        return creator

    @classmethod
    def get_all_creator_names(cls):
        return list(BonsaiFactory._creators.keys())


# TODO - change to iterator going over BonsaiModule children who ar not abstract in modules.bonsai_modules
BonsaiFactory.register_new_creator('conv2d', BConv2d)
BonsaiFactory.register_new_creator('prunable_conv2d', PBConv2d)
BonsaiFactory.register_new_creator('deconv2d', BDeconv2d)
BonsaiFactory.register_new_creator('prunable_deconv2d', PBDeconv2d)
BonsaiFactory.register_new_creator('route', BRoute)
BonsaiFactory.register_new_creator('maxpool', BMaxPool2d)
BonsaiFactory.register_new_creator('avgpool2d', BAvgPool2d)
BonsaiFactory.register_new_creator('pixel_shuffle', BPixelShuffle)
BonsaiFactory.register_new_creator('flatten', BFlatten)
BonsaiFactory.register_new_creator('linear', BLinear)
BonsaiFactory.register_new_creator('prunable_linear', PBLinear)
BonsaiFactory.register_new_creator('dropout', BDropout)
BonsaiFactory.register_new_creator('residual_add', BElementwiseAdd)
BonsaiFactory.register_new_creator('batchnorm2d', BBatchNorm2d)
