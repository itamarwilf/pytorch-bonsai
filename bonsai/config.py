"""Configuration for the package is handled in this wrapper for confuse."""
import logging
import yaml
import warnings
import os
import confuse


class Config(object):
    """This is a wrapper for the python confuse package, which handles setting and getting configuration variables via
    various ways (notably via argparse and kwargs).
    """

    config = None
    """The confuse.Configuration object."""

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self, config_path: str = None):
        if config_path is None or not os.path.exists(config_path):
            warnings.warn(f"provided bonsai config file: '{config_path}', is not found. using default config...")
            config_path = "config_default.yml"
            generate_default_config(config_path)

        self.config = confuse.Configuration("BonsaiPruning", __name__)
        self.config.set_file(config_path)
        logging.debug(
            "The config constructor should be called only once, you should see this message only once."
        )

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key].set(value)


def generate_default_config(path: str):

    default_dict = {"pruning": {"num_iterations": 9,
                                "prune_percent": 0.1,
                                "finetune_epochs": 3,
                                "patience": 2,
                                "out_path": "pruning_results",
                                "early_stopping": True
                                },

                    "optimizer": {"type": "Adam",
                                  "lr": 0.0001,
                                  "momentum": 0.9
                                  },

                    "logging": {"use_tensorboard": "yes",
                                "logdir": "runs",
                                "train_log_interval": 1
                                },
                    "evaluate":
                        {"eval_speed": 5}  # inference iterations to average when measuring inference time, 0 to cancel
                    }

    with open(path, "w") as f:
        yaml.safe_dump(default_dict, f)


config = Config()
# config = None
