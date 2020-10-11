"""
Implement many pipe function with form:
    fn(**config:dict) -> config(dict)
Class Pipe also supported:
    class A:
        def __call__(**config:dict) -> config(dict)
"""
import logging

from prlab.gutils import convert_to_obj_or_fn

logger = logging.getLogger(__name__)


def simple_model(**config):
    config['model'] = convert_to_obj_or_fn(config['model']).to(config.get('device', 'cuda'))
    return config


def opt_func_load(**config):
    # TODO fix opt load, maybe wrap to object to predefined params, such as lr, ...
    opt_fn = convert_to_obj_or_fn(config['opt_func'])
    config['opt_func'] = opt_fn(params=config['model'].parameters(), lr=config.get('lr'))
    return config


def model_general_setup(**config):
    """
    General setup after model build, including opt_func, loss_func, metrics
    :param config:
    :return:
    """
    config['opt_func'] = convert_to_obj_or_fn(config['opt_func'], params=config['model'].parameters())
    config['loss_func'] = convert_to_obj_or_fn(config['loss_func'], **config)
    config['metrics'] = convert_to_obj_or_fn(config['metrics'], **config)

    return config
