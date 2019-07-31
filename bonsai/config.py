"""Configuration for the package is handled in this wrapper for confuse."""
import logging

import confuse


class Config(object):
    """This is a wrapper for the python confuse package, which handles setting and getting configuration variables via
    various ways (notably via argparse and kwargs).
    """

    config = None
    """The confuse.Configuration object."""

    def __init__(self, config_path="config_default.yaml"):
        if self.config is None:
            self.config = confuse.Configuration("BonsaiPruning", __name__)
            self.config.set_file(config_path)
            logging.debug(
                "The config constructor should be called only once, you should see this message only once."
            )

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key].set(value)


config = Config()
