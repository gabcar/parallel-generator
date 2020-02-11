"""config_loader.py

Loads config from .yml files.
"""
import sys
import os

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

DEFAULT_CONFIG_PATH = 'configs/default.yml'

def print_configs(configs):
    """
    Pretty print for a dict/CommentedMap. 
    This function is called every time a 
    config file is loaded.
    """
    _, terminal_width = os.popen('stty size', 'r').read().split()
    print(terminal_width)
    yaml = YAML()
    print('=' * int(terminal_width))
    yaml.dump(configs, sys.stdout)
    print('=' * int(terminal_width))
       

def load_config_as_dict(path):
    """
    Loads config file from .yml format
    and returns it as a dict.

    If the specified path is None the function
    returns the default config path.

    :in path: Path to config file
    """
    if path:
        config_path = path
    else:
        config_path = DEFAULT_CONFIG_PATH
    with open(config_path, 'r') as file:
        yaml = YAML()
        configs =  yaml.load(file)

    return configs


def log_config(out_path, config):
    """
    Saves config to log 
    """
    yaml = YAML()

    with open(out_path, 'w') as file:
        yaml.dump(config, file)
