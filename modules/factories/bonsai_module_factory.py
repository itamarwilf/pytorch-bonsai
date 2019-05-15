from modules.bonsai_modules import *


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


BonsaiFactory.register_new_creator('conv2d', BConv2d)
BonsaiFactory.register_new_creator('prunable_conv2d', PBConv2d)
BonsaiFactory.register_new_creator('deconv2d', BDeconv2d)
BonsaiFactory.register_new_creator('prunable_deconv2d', PBDeconv2d)
BonsaiFactory.register_new_creator('route', BRoute)
BonsaiFactory.register_new_creator('maxpool', BMaxPool)
BonsaiFactory.register_new_creator('pixel_shuffle', BPixelShuffle)
