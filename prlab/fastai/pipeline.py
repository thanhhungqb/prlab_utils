"""
Implement general pipeline for train/valid/test processes.
Can be easy to use to train a new model just by provided dict/JSON configure with some information including:
functions to call in pipeline, object to make,...
The main idea is make a standard pipeline that most of training process share, all variable will be save in configure,
and dump to file to easy track results later.

Note:
    - all pipeline process must be return at least 2 variable learn, config (change from old, only learn), input is
    learn and **config also
    - change use of data to data_train for clear meaning (and not confuse with data in fastai), data_test for testing
    (TODO keep data for older version use, but remove in future)
    - see `config/general.json` to basic configure for pipeline
"""

from fastai.vision import *

from prlab.fastai.utils import general_configure
from prlab.gutils import load_func_by_name


def pipeline_control(**kwargs):
    """
    For general control overall pipeline
    See `prlab.emotion.ferplus.sr_stn_vgg_8classes.train_test_control`
    :param kwargs:
    :return:
    """
    with open('config/general.json') as f:
        config = json.load(f)
    config.update(**kwargs)
    config = general_configure(**config)
    learn = None

    # load data func here, both train and test in data_train, data_test
    _, config, *_ = load_func_by_name(config['data_train'])[0](learn=learn, **config)

    process_pipeline = [load_func_by_name(o)[0] if isinstance(o, str) else o for o in config['process_pipeline']]
    for fn in process_pipeline:
        learn, config, *_ = fn(**config)
        config['learn'] = learn

    return learn, config


def model_build(**config):
    """
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    model_func, _ = load_func_by_name(config['model_func'])
    learn, layer_groups, *_ = model_func(**config)
    (config['cp'] / "model.txt").open('a').write(str(learn.model))
    config.update({
        'model': learn.model, 'layer_groups': layer_groups
    })

    return learn, config


def data_load_folder(**config):
    """
    Follow Pipeline Process template.
    Load from folder by name: train/val/test
    :param config:
    :return: None, new_config (None for learner)
    """
    data_train = (
        ImageList.from_folder(config['path'])
            .split_by_folder()
            .label_from_func(config['data_helper'].y_func, label_cls=FloatList)
            .transform(config['tfms'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    config['data_train'], config['data'] = data_train, data_train
    print('data train', data_train)

    # load test to valid to control later, ONLY USE AT TEST STEP
    data_test = (
        ImageList.from_folder(config['path'])
            .split_by_folder(valid='test')
            .label_from_func(config['data_helper'].y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    config['data_test'] = data_test
    print('data test', data_test)

    return None, config


def data_load_folder_df(**config):
    """
    Load from same folder, split by df (func)
    Follow Pipeline Process template
    TODO not implement yet
    :param config:
    :return: None, new_config (None for learner)
    """

    return None, config
