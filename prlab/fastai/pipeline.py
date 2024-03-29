"""
Implement general pipeline for train/valid/test processes.
Can be easy to use to train a new model just by provided dict/JSON configure with some information including:
functions to call in pipeline, object to make,...
The main idea is make a standard pipeline that most of training process share, all variable will be save in configure,
and dump to file to easy track results later.

Note:
    - all pipeline process must be received config as input and return new config as output
        form:  def func(**config): return new_config
    - change use of data to data_train for clear meaning (and not confuse with data in fastai), data_test for testing
    (TODO keep data for older version use, but remove in future)
    - see `config/general.json` to basic configure for pipeline
"""
import json
from functools import partial
from pathlib import Path

import deprecation
import fastai
import nltk
import sklearn
import torch
import torch.nn as nn
from fastai.learner import Learner
from fastai.tabular.learner import tabular_learner
from fastai.vision import models
from fastai.vision.core import imagenet_stats
from fastai.vision.learner import create_cnn_model, unet_learner, ImageDataLoaders, cnn_learner
from fastcore.basics import listify, defaults
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim import AdamW

from outside.scikit.plot_confusion_matrix import plot_confusion_matrix
from outside.stn import STN
from outside.super_resolution.srnet import SRNet3
from prlab.common import *
from prlab.common.dl import pipeline_control_multi
from prlab.common.utils import load_func_by_name, set_if, npy_arr_pretty_print, encode_and_bind, lazy_object_fn_call
# from prlab.fastai.image_data import SamplerImageList
from prlab.fastai.utils import general_configure, base_arch_str_to_obj
from prlab.torch.functions import fc_exchange_label

pipeline_control_multi  # just for old reference, new call should be in prlab.common.dl.pipeline_control_multi

ImageList = ImageDataLoaders
is_fastai_v1 = fastai.__version__ < "2"


def pipeline_control(**kwargs):
    """
    For general control overall pipeline
    See `prlab.emotion.ferplus.sr_stn_vgg_8classes.train_test_control`
    :param kwargs:
    :return:
    """
    config = {}
    try:
        with open('config/general.json') as f:
            config = json.load(f)
    except Exception as e:
        print('no general.json', e)

    config.update(**kwargs)
    config = general_configure(**config)

    process_pipeline = [load_func_by_name(o)[0] if isinstance(o, str) else o for o in config['process_pipeline']]
    for fn in process_pipeline:
        config = fn(**config)

    return config


# ************* model *************************************
def model_build(**config):
    """
    Follow Pipeline Process template.
    TODO model_func will follow Pipeline Process template too
    :param config:
    :return:
    """
    model_func, _ = load_func_by_name(config['model_func'])
    learn, layer_groups, *_ = model_func(**config)
    config.update({
        'model': learn.model, 'layer_groups': layer_groups, 'learn': learn
    })

    return config


def learn_general_setup(**config):
    """
    Follow Pipeline Process template.
    :param config: contains learn and all configure
    :return:
    """
    learn = config['learn']
    if config.get('loss_func', None) is not None:
        learn.loss_func = config['loss_func']
    if config.get('callback_fn', None) is not None:
        learn.callback_fns = learn.callback_fns[:1] + listify(config['callback_fn']())

    if config.get('metrics', None) is not None:
        learn.metrics = listify(config['metrics'])

    if config.get('layer_groups', None) is not None:
        learn.layer_groups = config['layer_groups']

    (config['cp'] / "model.txt").open('a').write(str(learn.model))
    (config['cp'] / "model.txt").open('a').write(
        "\n".join([str(learn.metrics), str(learn.loss_func), str(learn.callback_fns)]))

    return config


# ************* model func *******************
def self_created_model(**config):
    model = config['model']
    model = lazy_object_fn_call(model, **config)
    layer_groups = model.layer_groups() if hasattr(model, 'layer_groups') else [model]

    learn = Learner(config['data_train'], model=model, metrics=config['metrics'],
                    layer_groups=layer_groups,
                    model_dir=config['cp'])

    config.update({'learn': learn, 'model': learn.model, 'layer_groups': learn.layer_groups})

    return config


def basic_model_build(**config):
    """
    Build basic model: vgg, resnet, ...
    :param config:
    :return: learn and layer_groups
    """
    base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))
    learn = cnn_learner(data=config['data_train'], base_arch=base_arch, model_dir=config['cp'])
    learn.unfreeze()
    return learn, learn.layer_groups


def tabular_dnn_learner_build(**config):
    """
    Follow Pipeline Process template.
    Build a DNN model (for tabular data)
    :param config:
    :return: new config with update learner
    """
    data_train = config['data_train']
    learn = tabular_learner(data_train,
                            layers=config['dnn_layers'],
                            ps=config.get('ps', None),
                            emb_drop=config.get('emb_drop', 0.),
                            use_bn=config.get('use_bn', True),
                            emb_szs=config['emb_szs'],
                            model_dir=config.get('cp', 'models'))

    config.update({
        'learn': learn, 'model': learn.model, 'layer_groups': learn.layer_groups
    })

    return config


def create_obj_model(**config):
    """
    Follow Pipeline Process template.
    Make a leaner and set to config, update and return the new one.
    Note: by new style, it is not need to use `prlab.fastai.pipeline.basic_model_build` to load this function,
    directly add to pipeline instead.
    :param config: contains model_class or model_func (old)
    :return: new config with update learn, model and layer_groups
    """
    set_if(config, 'model_class', config.get('model_func', None))  # for back support, but not necessary needed
    model_class, _ = load_func_by_name(config['model_class'])
    model = model_class(**config)
    if hasattr(model, 'load_weights'):
        model.load_weights(**config)
    layer_groups = model.layer_groups() if hasattr(model, 'layer_groups') else [model]
    opt = partial(optim.SGD, momentum=config.get('momentum', 0.9)) \
        if config.get('opt_func', None) is not None and config['opt_func'] == 'SGD' else AdamW

    learn = Learner(config['data_train'], model=model,
                    opt_func=opt,
                    layer_groups=layer_groups,
                    model_dir=config['cp'])

    config.update({
        'learn': learn,
        'model': model,
        'layer_groups': layer_groups,
    })

    return config


def stn_based(**config):
    """
    model_func need return at least learn, model and layer_groups
    add STN on top of the base model.
    Set loss_func if have.
    Set callback if have.
    :param config:
    :return:
    """
    cp = config['cp']
    base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))

    model = nn.Sequential(
        STN(img_size=config['img_size']),
        create_cnn_model(base_arch, nc=config['n_classes'])
    )
    layer_groups = [model[0], model[1]]

    learn = Learner(config['data_train'], model=model, metrics=config['metrics'], layer_groups=layer_groups,
                    model_dir=cp)

    return learn, layer_groups


def sr_xn_stn(**config):
    """
    model_func need return at least learn and layer_groups
    Note:
        - load state_dict for model[0] (SRNet3) may be need, use `prlab.fastai.pipeline.srnet3_weights_load`
    :param config:
    :return:
    """
    cp = config['cp']
    xn = config.get('xn', 2)

    base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))

    model = nn.Sequential(
        SRNet3(xn),
        STN(img_size=config['img_size'] * xn),
        create_cnn_model(base_arch, nc=config['n_classes'])
    )
    layer_groups = [model[0], model[1], model[2]]

    learn = Learner(config['data_train'], model=model, metrics=config['metrics'], layer_groups=layer_groups,
                    model_dir=cp)

    return learn, layer_groups


def stn_sr_xn(**config):
    """
    Similar with `prlab.fastai.pipeline.sr_xn_stn` but stn before sr.
    The idea here is while stn could make some noise, sr should correct it in natural way
    model_func need return at least learn and layer_groups
    Note:
        - load state_dict for model[0] (SRNet3) may be need, use `prlab.fastai.pipeline.srnet3_weights_load`
    :param config:
    :return:
    """
    cp = config['cp']
    xn = config.get('xn', 2)

    base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))

    model = nn.Sequential(
        STN(img_size=config['img_size']),
        SRNet3(xn),
        create_cnn_model(base_arch, nc=config['n_classes'])
    )
    layer_groups = [model[0], model[1], model[2]]

    learn = Learner(config['data_train'], model=model, model_dir=cp)

    return learn, layer_groups


def build_unet_model(**config):
    config['learn'] = unet_learner(
        data=config['data_train'],
        arch=config.get('arch', models.resnet34),
        metrics=config.get('metrics', None),
        wd=config.get('wd', 1e-2),
        model_dir=config['cp'])

    return config


def wrap_pipeline_style_fn(old_func):
    """
    Wrap the old_func to new style fn(**config): new_config and then can add to pipeline directly
    :param old_func: fn(**config): learn, layer_groups
    :return: a function fn(**config): new_config
    """

    def _fn(**config):
        learn, layer_groups = old_func(**config)
        config.update({
            'learn': learn,
            'model': learn.model,
            'layer_groups': layer_groups,
        })
        return config

    return _fn


basic_model_build_ = wrap_pipeline_style_fn(basic_model_build)
stn_based_ = wrap_pipeline_style_fn(stn_based)
sr_xn_stn_ = wrap_pipeline_style_fn(sr_xn_stn)
stn_sr_xn_ = wrap_pipeline_style_fn(stn_sr_xn)


def wrap_model(**config):
    """
    Load from model make by model_func
    :param config:
    :return:
    """
    fn, _ = load_func_by_name(config['model_func'])
    model = fn(**config)
    learn = Learner(config['data_train'], model=model, model_dir=config['cp'])

    config.update({
        'learn': learn,
        'model': learn.model,
        'layer_groups': model.layer_groups() if hasattr(model, 'layer_groups') else None
    })
    return config


# *************** DATA **********************************
def data_load_folder(**config):
    """
    Follow Pipeline Process template.
    Load from folder by name:
        valid_pct None: train/val/test
        valid_pct (for train/valid in path) and test in test_path
    :param config:
    :return: None, new_config (None for learner)
    """
    assert is_fastai_v1, "only work with fastai v1"
    print('starting load train/valid')
    train_load = SamplerImageList.from_folder(config['path'])
    train_load = train_load.filter_by_func(config['data_helper'].filter_func) \
        if hasattr(config['data_helper'], 'filter_func') else train_load
    if config.get('valid_pct', None) is not None:
        train_load = train_load.split_by_rand_pct(valid_pct=config['valid_pct'], seed=config.get('seed', None))
    elif config.get('is_valid_fn', None) is not None:
        train_load = train_load.split_by_valid_func(config['data_helper'].valid_func)
    else:
        train_load = train_load.split_by_folder(train=config.get('train_folder', 'train'),
                                                valid=config.get('valid_folder', 'valid'))

    data_train = (
        train_load
            .label_from_func(config['data_helper'].y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'], size=config['img_size'])
            .databunch(bs=config['bs'], sampler_super=config.get('sampler_super', None))
    ).normalize(imagenet_stats)
    config['data_train'], config['data'] = data_train, data_train
    print('data train', data_train)

    # load test to valid to control later, ONLY USE AT TEST STEP
    print('starting load test')
    if config.get('valid_pct', None) is not None:
        # in this case, merge parent folder and just get test
        # to make sure train is not empty and not label filter out in test set
        # test_path should be a parent folder of path (or similar meaning)
        # if training size is big, test_path may simulate training but smaller size to quicker load
        test_load = ImageList.from_folder(config['test_path'])
    elif config.get('is_valid_fn', None) is not None:
        test_load = ImageList.from_folder(config['test_path'])
    else:
        test_load = ImageList.from_folder(config['path'])

    test_load = test_load.filter_by_func(config['data_helper'].filter_func) \
        if hasattr(config['data_helper'], 'filter_func') else test_load
    test_load = test_load.split_by_folder(train=config.get('train_folder', 'train'),
                                          valid=config.get('test_folder', 'test'))

    data_test = (
        test_load
            .label_from_func(config['data_helper'].y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'], size=config['img_size'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    config['data_test'] = data_test
    print('data test', data_test)

    return config


def data_load_folder_df(**config):
    """
    Load from same folder, split by df (func)
    Follow Pipeline Process template
    Note: get image path from data_helper instead config
    :param config:
    :return: None, new_config (None for learner)
    """
    # processor
    data_helper = config['data_helper']
    data_train = (
        ImageList.from_folder(data_helper.path)
            .filter_by_func(data_helper.filter_train_fn)
            .split_by_valid_func(data_helper.split_valid_fn)
            .label_from_func(data_helper.y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'], size=config['img_size'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    print('Load data done for train', data_train)

    data_test = (
        ImageList.from_folder(data_helper.path)
            .split_by_valid_func(data_helper.filter_test_fn)
            .label_from_func(data_helper.y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'], size=config['img_size'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    print('Load data done for test', data_test)

    config.update({
        'data_train': data_train,
        'data': data_train,
        'data_test': data_test
    })

    return config


def data_load_folder_balanced(**config):
    """
    Follow Pipeline Process template.
    Load from folder by name:
        valid_pct None: train/val/test
        valid_pct (for train/valid in path) and test in test_path
    :param config:
    :return: None, new_config (None for learner)
    """
    assert is_fastai_v1, "only work with fastai v1"
    image_list_cls = BalancedLabelImageList
    train_load = image_list_cls.from_folder(path=config['path'], data_helper=config['data_helper'],
                                            each_class_num=config['each_class_num'])
    train_load = train_load.filter_by_func(config['data_helper'].filter_func)
    if config.get('valid_pct', None) is None:
        train_load = train_load.split_by_folder(train=config.get('train_folder', 'train'),
                                                valid=config.get('valid_folder', 'valid'))
    else:
        train_load = train_load.split_by_rand_pct(valid_pct=config['valid_pct'], seed=config.get('seed', None))

    data_train = (
        train_load
            .label_from_func(config['data_helper'].y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'], size=config['img_size'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    config['data_train'], config['data'] = data_train, data_train
    print('data train', data_train)

    # load test to valid to control later, ONLY USE AT TEST STEP
    print('starting load test')
    if config.get('valid_pct', None) is None:
        test_load = ImageList.from_folder(config['path'])
        test_load = test_load.filter_by_func(config['data_helper'].filter_func)
        test_load = test_load.split_by_folder(train=config.get('train_folder', 'train'),
                                              valid=config.get('test_folder', 'test'))
    else:
        # in this case, merge parent folder and just get test
        # to make sure train is not empty and not label filter out in test set
        test_load = ImageList.from_folder(config['test_path'])
        test_load = test_load.filter_by_func(config['data_helper'].filter_func)
        test_load = test_load.split_by_folder(train=config.get('train_folder', 'train'),
                                              valid=config.get('test_folder', 'test'))

    data_test = (
        test_load
            .label_from_func(config['data_helper'].y_func, label_cls=config['data_helper'].label_cls)
            .transform(config['tfms'], size=config['img_size'])
            .databunch(bs=config['bs'])
    ).normalize(imagenet_stats)
    config['data_test'] = data_test
    print('data test', data_test)

    return config


# ********** Segmentation data loader ************
def load_seg_data(**config):
    assert is_fastai_v1, "only work with fastai v1"
    dh = config['data_helper']
    bs = config.get('bs', 16)
    classes = config.get('classes', None)
    seg_item_list_cls = config.get('seg_item_list_cls', SegmentationItemList)
    src = (seg_item_list_cls.from_folder(config['path'],
                                         convert_mode=config.get('convert_mode', 'RGB'))
           .filter_by_func(dh.filter_func)
           .split_by_valid_func(dh.valid_fn)
           .label_from_func(dh.y_func, classes=classes))

    config['data'] = (src.transform(config['tfms'], size=config['img_size'], tfm_y=True)
                      .databunch(bs=bs)
                      .normalize(imagenet_stats))
    config['data_train'] = config['data']

    # load test set if given in path_test
    if config.get('path_test', None) is not None:
        src_test = (seg_item_list_cls.from_folder(config['path_test'],
                                                  convert_mode=config.get('convert_mode', 'RGB'))
                    .filter_by_func(dh.filter_func)
                    .split_by_valid_func(lambda x: True)
                    .label_from_func(dh.y_func, classes=classes))
        config['data_test'] = (src_test.transform(config['tfms'], size=config['img_size'], tfm_y=True)
                               .databunch(bs=bs)
                               .normalize(imagenet_stats))
        print(config['data_test'])

    return config


# *************** TRAINING PROCESS **********************************
def training_simple(**config):
    """
    very simple training process
    `Pipeline Process template`
    :param config:
    :return:
    """
    learn = config['learn']
    learn.data = config['data_train']
    learn.save(config.get('best_name', 'best'))

    lr = config.get('lr', 1e-3)
    learn.fit_one_cycle(config.get('epochs', 30), max_lr=lr)

    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    return config


def training_simple_2_steps(**config):
    """
    Follow Pipeline Process template.
    Train with two steps: first with large lr for some epochs and then smaller lr with next some epochs
    :param config:
    :return: new config
    """
    learn = config['learn']
    learn.save(config.get('best_name', 'best'))

    # for large lr
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 30)
    learn.fit_one_cycle(epochs, max_lr=lr)

    # smaller lr, if not given then lr/10
    lr_2 = config.get('lr_2', lr)
    epochs_2 = config.get('epochs_2', epochs)
    learn.fit_one_cycle(epochs_2, max_lr=lr_2)

    return config


def train_seg_two_steps(**config):
    learn = config['learn']

    lr = config.get('lr', 1e-3)
    epochs = config.get('epochs', 10)
    learn.fit_one_cycle(epochs, slice(lr), pct_start=0.9)  # train model

    learn.save('freeze')  # save model

    learn.unfreeze()  # unfreeze all layers

    # find and plot lr again
    learn.unfreeze()
    learn.recorder.plot()

    lr_2 = config.get('lr_2', lr / 4)
    epochs_2 = config.get('epochs_2', epochs)
    learn.fit_one_cycle(epochs_2, slice(lr_2 / 100, lr_2), pct_start=0.8)

    learn.save(config.get('best_name', 'best'))

    return config


def training_adam_sgd(**config):
    """
    A training process that use both adam and sgd, adam for first epochs and later is sgd.
    This process seems quicker to sgd at the begin, but also meet the best result.
    Note, for this process, `best` model could not be load correctly, then resume should be careful and addition work to
    load weights only.
    Note: at the beginning, `learn` mostly configure with ADAM
    :param config:
    :return:
    """
    learn = config['learn']
    best_name = config.get('best_name', 'best')
    data_train, model, layer_groups = config['data_train'], config['model'], config['layer_groups']

    # TODO see note in header
    learn.save(best_name)  # TODO why need in the newer version of pytorch

    learn.data = data_train

    epochs = config.get('epochs', 30)
    lr = config.get('lr', 5e-3)
    learn.fit_one_cycle(epochs, max_lr=lr)

    learn.save('best-{}'.format(epochs))

    torch.save(learn.model.state_dict(), config['cp'] / 'e_{}.w'.format(epochs))
    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    # SGD optimize
    opt = partial(optim.SGD, momentum=0.9)
    learn = Learner(data_train, model=model, opt_func=opt, metrics=config['metrics'],
                    layer_groups=layer_groups,
                    model_dir=config['cp'])
    learn.model.load_state_dict(torch.load(config['cp'] / 'e_{}.w'.format(epochs)))
    config['learn'] = learn  # new leaner

    config = learn_general_setup(**config)
    learn.data = data_train

    epochs = config.get('epochs_2', epochs)
    lr = config.get('lr_2', lr)
    learn.fit_one_cycle(epochs, max_lr=lr)

    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    return config


def training_freeze(**config):
    """
    training with some diff
    :param config:
    :return:
    """
    learn = config['learn']
    learn.data = config['data_train']
    learn.save(config.get('best_name', 'best'))  # TODO why need in the newer version of pytorch

    lr = config.get('lr', 5e-3)
    learn.freeze_to(config.get('freeze_to', -1))
    learn.fit_one_cycle(config.get('epochs', 30), max_lr=lr)
    torch.save(learn.model.state_dict(), config['cp'] / 'freeze.w')

    learn.unfreeze()
    lr_2 = config.get('lr_2', lr)
    learn.fit_one_cycle(config.get('epochs_2', 30), max_lr=lr_2)

    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    return config


def two_step_train_saliency(**config):
    learn = config['learn']

    learn.save(config['best_name'])

    learn.data = config['data_train']

    lr = config.get('lr', 5e-4)
    if not isinstance(lr, list):
        lr = [lr / 500, lr / 100, lr / 100, lr]
    learn.model.p = 0
    epochs = config.get('epochs', 30)
    learn.fit_one_cycle(epochs, max_lr=lr)

    torch.save(learn.model.state_dict(), config['cp'] / 'e_{}.w'.format(config.get('epochs', 30)))
    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    lr_2 = config.get('lr_2', lr)
    if not isinstance(lr_2, list):
        lr_2 = [lr_2 / 500, lr_2 / 100, lr_2 / 100, lr_2]
    learn.model.p = 1
    epochs_2 = config.get('epochs_2', epochs)
    learn.fit_one_cycle(epochs_2, max_lr=lr_2)
    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    return config


# *************** REPORT **********************************
def backup_learner_data_decorator(fn):
    def fn_(**config):
        learn = config['learn']

        # make backup of current data of learn
        data_current = learn.data
        if hasattr(learn.model, 'is_testing'):
            learn.model.is_testing = True
        learn.data = config['data_test']

        config = fn(**config)

        # roll back for data in learn
        learn.data = data_current
        if hasattr(learn.model, 'is_testing'):
            learn.model.is_testing = False

        return config

    return fn_


@backup_learner_data_decorator
def make_report_cls(**config):
    """
    Newer, simpler version of `prlab.emotion.ferplus.sr_stn_vgg_8classes.run_report`,
    better support `Pipeline Process template`
    This is for CLASSIFICATION, report on accs and f1 score.
    y labels could be one-hot or prob mode.
    Run TTA 3 times and make report to screen, reports.txt and results.npy
    :param config: contains data_test store test in valid mode, tta_times (if have)
    :return: as description of `Pipeline Process template` including learn and config (not update in this func)
    """
    assert is_fastai_v1, "only work with fastai v1"
    print('starting report')
    learn = config['learn']
    cp = config['cp']

    print(config['data_test'])
    if hasattr(config['data_test'], 'classes') and config['data_test'].classes is not None:
        classes = np.array(config['data_test'].classes)
    else:
        classes = np.array(config['label_names']) if config.get('label_names', None) is not None else None

    # when we want to override classes in data_test to another from configure, one flag need
    if config.get('replace_classes', False):
        classes = np.array(config.get('label_names', None))

    print('classes (order)', classes)

    items = np.array([str(o) for o in config['data_test'].valid_ds.items])

    accs, f1s, to_save = [], [], {}
    uas = []
    for run_num in range(config.get('tta_times', 3)):
        ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=config.get('test_scale', 1.10))

        ys_npy, y_npy = ys.numpy(), y.numpy()
        ys_labels = np.argmax(ys_npy, axis=-1)
        #  support both one-hot and prob mode
        y_labels = y_npy if isinstance(y_npy.flat[0], (np.int64, np.int)) else np.argmax(y_npy, axis=-1)

        accs.append(nltk.accuracy(ys_labels, y_labels))
        f1 = sklearn.metrics.f1_score(y_labels, ys_labels, average='macro')  # micro macro
        to_save['time_{}'.format(run_num)] = {'ys': ys_npy, 'y': y_npy, 'acc': accs[-1], 'f1': f1}
        to_save['time_{}'.format(run_num)]['items'] = items
        print('run', run_num, accs[-1], 'f1', f1)

        _, fig = plot_confusion_matrix(y_labels, ys_labels,
                                       classes=classes,
                                       normalize=config.get('normalize_cm', True),
                                       title='Confusion matrix')
        fig.savefig(cp / 'run-{}.png'.format(run_num))
        cm = confusion_matrix(y_labels, ys_labels)
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        uas.append(np.trace(cm_n) / len(cm_n))
        (config['cp'] / "cm.txt").open('a').write(
            'acc: {:.4f}\tf1: {:.4f}\n{}\n{}\n\n'.format(accs[-1], f1,
                                                         npy_arr_pretty_print(cm, fm='{:>8}'),
                                                         npy_arr_pretty_print(cm_n)))
        f1s.append(f1)

    stats = [np.average(accs), np.std(accs), np.max(accs), np.median(accs)]

    accs_str = ' '.join(['{0:.4f}'.format(o) for o in accs])
    stats_str = ' '.join(['{0:.4f}'.format(o) for o in stats])
    f1s_str = ' '.join(['{:.4f}'.format(o) for o in f1s])
    uas_str = ' '.join(['{:.4f}'.format(o) for o in uas])
    (config['model_path'] / "reports.txt").open('a').write(
        '{}\t{}\tuas: {}\tstats: {}\tf1: {}\n'.format(cp, accs_str, uas_str, stats_str, f1s_str))
    print('3 results', accs_str, 'stats', stats_str)

    np.save(cp / "results", to_save)

    return config


@backup_learner_data_decorator
def make_report_general(**config):
    """
    Report for regression or some general case, where just focus on metrics
    Follow Pipeline Process template.
    :param config: contains data_test store test in valid mode, tta_times (if have)
    :return: new config
    """
    assert is_fastai_v1, "only work with fastai v1"
    print('starting report for regression')
    learn = config['learn']
    cp = config['cp']

    print(config['data_test'])

    metric_outs, to_save = [], {}
    metrics = config['metrics']
    metrics = metrics if isinstance(metrics, list) else [metrics]

    items = np.array([str(o) for o in config['data_test'].valid_ds.items])

    for run_num in range(config.get('tta_times', 3) if not config.get('NON_TTA', False) else 1):
        ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=config.get('test_scale', 1.10)) \
            if not config.get('NON_TTA', False) else learn.get_preds()

        outs = [o(ys, y) for o in metrics]

        print('run', run_num, outs)

        outs = [o.numpy().tolist() for o in outs]
        to_save['time_{}'.format(run_num)] = {'outs': outs, 'ys': ys.numpy(), 'y': y.numpy()}
        to_save['time_{}'.format(run_num)]['items'] = items

        metric_outs.append(outs)
        tmp_s = npy_arr_pretty_print(np.array(outs))
        (config['cp'] / "test-out.txt").open('a').write('?: {}\n\n'.format(tmp_s))

    # metric_outs shape [run_num, len(metrics)]
    outs_npy = np.array(metric_outs)
    stats = [np.average(outs_npy, axis=1), np.std(outs_npy, axis=1),
             np.min(outs_npy, axis=1), np.max(outs_npy, axis=1),
             np.median(outs_npy, axis=1)]

    # accs list of list (1+)
    metric_outs_str = npy_arr_pretty_print(outs_npy)
    stats_str = npy_arr_pretty_print(np.array(stats))

    (config['model_path'] / "reports.txt").open('a').write('{}\t{}\n{}\n\n'.format(cp, stats_str, metric_outs_str))
    print('3 results', stats_str)

    np.save(cp / "results", to_save)

    return config


make_report_regression = make_report_general


@backup_learner_data_decorator
def make_report_simple(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    :param config:
    :return: new config with output
    """
    assert is_fastai_v1, "only work with fastai v1"
    learn = config['learn']
    preds, gt = learn.get_preds(DatasetType.Valid)

    metrics = learn.metrics
    metrics = metrics if isinstance(metrics, list) else [metrics]

    to_ret = [o(preds, gt) for o in metrics]
    to_ret = [o.cpu().numpy() for o in to_ret]

    config['output'] = to_ret
    return config


@backup_learner_data_decorator
def predict_and_save_simple(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    :param config:
    :return: new config with output
    """
    assert is_fastai_v1, "only work with fastai v1"
    learn = config['learn']
    cp = config['cp']
    preds, gt = learn.get_preds(DatasetType.Valid)

    preds_npy, gt_npy = preds.numpy(), gt.numpy()

    items = np.array([str(o) for o in config['data_test'].valid_ds.items])
    to_save = {'ys': preds_npy, 'y': gt_npy, 'items': items}

    np.save(cp / "result-raw", to_save)
    print(preds_npy[:10])

    return config


@backup_learner_data_decorator
def make_fold_dnn_report(**config):
    """
    Follow Pipeline Process template in pipeline_control_multi.
    Should be called after report, where config['output'] available
    :param config:
    :return: do nothing, just return config, write fold output to text file
    """
    to_ret = config['output']
    print('this fold', to_ret)
    (config['cp'] / "folds.txt").open('a').write("fold {}: {}\n".format(config.get('test_fold', 0), to_ret))
    (config['model_path'] / "reports.txt").open('a').write(
        '{}\thidden:{}\t{}\n'.format(config['cp'],
                                     'x'.join([str(o) for o in config['dnn_layers']]),
                                     to_ret))
    return config


@backup_learner_data_decorator
def make_predict_seg(**config):
    """ make some sample for segmentation return list[(input, output, gt)]. Also save to file """
    print('get (img, gt, pred) and save to config to use later. e.g. slide_bar_show, save to file, viz')

    ys, gt = config['learn'].get_preds()
    xs = ys.argmax(dim=1)
    gt, xs = torch.squeeze(gt, dim=1), torch.squeeze(xs, dim=1)

    test_ds = config['data_test'].valid_ds
    imgs = [torch.squeeze(test_ds.get(i).data, dim=0) for i in range(len(test_ds))]

    config['out'] = {
        'imgs': imgs,
        'gt': gt,
        'pred': xs,
    }
    # e.g. multi_slide_bar([imgs, gt, xs])

    return config


@backup_learner_data_decorator
def predict_and_save(**config):
    assert is_fastai_v1, "only work with fastai v1"
    print('starting predict_and_save')

    learn = config['learn']
    cp = config['cp']

    if hasattr(learn.model, 'is_testing'):
        learn.model.is_testing = config.get('is_testing', True)

    print(config['data_test'])

    to_save = {}
    items = np.array([str(o) for o in config['data_test'].valid_ds.items])
    for run_num in range(config.get('tta_times', 3)):
        ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=config.get('test_scale', 1.10))

        ys_npy, y_npy = ys.numpy(), y.numpy()

        to_save['time_{}'.format(run_num)] = {'ys': ys_npy, 'y': y_npy}
        to_save['time_{}'.format(run_num)]['items'] = items

    np.save(cp / "result-raw", to_save)

    return config


# *************** WEIGHTS LOAD **********************************
def srnet3_weights_load(**config):
    """
    Use together with `prlab.fastai.pipeline.sr_xn_stn` to load pre-trained weights for
    `outside.super_resolution.srnet.SRNet3`
    Note: should call after model build.
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    learn = config['learn']
    xn = config.get('xn', 2)
    xn_weights_path = '/ws/models/super_resolution/facial_x{}.pth'.format(xn)
    xn_weights_path = xn_weights_path if config.get('xn_weights_path', None) is None else config['xn_weights_path']
    for i in range(3):
        if isinstance(learn.model[i], SRNet3):
            out = learn.model[i].load_state_dict(torch.load(xn_weights_path))
            print('load srnet2 out:', xn_weights_path, out)

    return config


def load_weights(**config):
    learn = config['learn']
    out = learn.model.load_state_dict(torch.load(config['cp'] / 'final.w'))
    print('load weights', out)
    return config


def resume_learner(**config):
    """
    Resume, load weight from final.w or best_name in this order.
    Note: best_name maybe newer than final.w, then will override if both found
    Order: final.w, best_name.pth
    :param config:
    :return:
    """
    print('resume step')
    learn = config['learn']
    best_name = config.get('best_name', 'best')

    if (config['cp'] / 'final.w').is_file():
        print('resume from weights')
        try:
            learn.model.load_state_dict(torch.load(config['cp'] / config.get('final_w', 'final.w')), strict=False)
        except Exception as e:
            print(e)

    if (config['cp'] / f'{best_name}.pth').is_file():
        print('resume from checkpoint')
        try:
            learn.load(best_name)
        except Exception as e:
            print(e)

    return config


def transfer_weight_load(**config):
    """
    Follow Pipeline Process template.
    order: weight_transfer, learner_transfer
    :param config:
    :return: new config (update learn)
    """
    learn = config['learn']

    weight_path = config.get('weight_transfer', None)
    if weight_path and Path(weight_path).is_file():
        print('transfer from weights', weight_path)
        try:
            learn.model.load_state_dict(torch.load(weight_path), strict=False)
        except Exception as e:
            print(e)

    learner_cp_path = config.get('learner_transfer', None)
    if learner_cp_path:
        print('transfer from checkpoint', learner_cp_path)
        try:
            learn.load(learner_cp_path)
        except Exception as e:
            print(e)

    return config


@deprecation.deprecated(
    details='use `prlab.fastai.pipeline.base_weights_load` for general case, note: change the params')
def vgg16_weights_load(**config):
    """
    Use together with `prlab.fastai.pipeline.sr_xn_stn` to load pre-trained weights for
    `outside.super_resolution.srnet.SRNet3`
    Note: should call after model build.
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    vgg16_weights_path = '/ws/models/ferplus/vgg16_bn_quick/final.w'
    config = set_if(config, 'xvgg16_weights_path', vgg16_weights_path)
    config = set_if(config, 'base_weights_path', config['xvgg16_weights_path'])

    return base_weights_load(**config)


def base_weights_load(**config):
    """
    Pipeline Process template.
    Load for vgg, resnet, ... which in the latest layer (classifier layer)
    Mostly use with `prlab.fastai.pipeline.stn_sr_xn`, `prlab.fastai.pipeline.sr_xn_stn`
    should be CALL after build model and before training step
    :param config:
    :return:
    """
    learn = config['learn']
    base_weights_path = config.get('base_weights_path', None)
    if base_weights_path is None:
        raise Exception('want to load for {}, weight path must provide in {}'.format(config['base_arch'],
                                                                                     config['base_weights_path']))

    # base is in the latest layer (-1) in the sequence
    out = learn.model[-1].load_state_dict(torch.load(base_weights_path), strict=False)
    print('load base weight {} status'.format(config['base_arch']), out)

    return config


def exchange_fc(**config):
    """
    load weights and update order of label (fc layer at the latest)
    Must provide:
        new_pos, if none then no exchange, e.g. [1, 0, 2, 3, 4, 5, 6, 7]
    LOAD BEFORE MODEL LOAD/RESUME/BUILD
    :return: update base_weights_path
    """
    print("Do the FC exchange for label order change")
    print('new order', config.get('new_pos', None))
    base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))

    base_weights_path = config.get('base_weights_path', None)
    base_model = create_cnn_model(base_arch=base_arch, nc=config['n_classes'])
    if base_weights_path is not None and Path(base_weights_path).is_file():
        state_dict = torch.load(base_weights_path)

        two_latest_keys = list(state_dict.keys())[-2:]
        stored_lbl_size = state_dict[two_latest_keys[-1]].size()[0]
        if stored_lbl_size != config['n_classes']:
            base_model = create_cnn_model(base_arch=base_arch, nc=stored_lbl_size)

        o = base_model.load_state_dict(state_dict=state_dict, strict=False)
        print('load weights to exchange from ', base_weights_path, o)

    fc_here = base_model[-1]
    if isinstance(fc_here, nn.Sequential):
        fc_here = fc_here[-1]
    fc_exchange_label(fc_here, config.get('new_pos', None), in_place=True)
    save_to = config.get('base_weights_path_transfer', '/tmp/{}.w'.format(config.get('base_arch', 'vgg16_bn')))
    torch.save(base_model.state_dict(), save_to)

    config['base_weights_path'] = save_to
    return config


# *************** OTHERS **********************************
@deprecation.deprecated(details='replace by device_setup function')
def cpu_ws(**config):
    """
    To set train/test on cpu working space.
    If use, must be before all data-loader, model build, etc. in the pipeline
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    defaults.device = torch.device('cpu')
    return config


def device_setup(**config):
    """
    To set train/test on cpu working space.
    If use, must be before all data-loader, model build, etc. in the pipeline
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    defaults.device = torch.device(config.get('device', 'cuda'))
    return config


def fold_after(**config):
    """
    Do something for k-fold after report: move all file to new, important is checkpoint, weight
    Move all files to fold folder
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    cp_fold = config['cp'] / f'{config.get("fold", 1)}'
    cp_fold.mkdir(parents=True, exist_ok=True)

    cp_files = [o for o in list(config['cp'].iterdir()) if o.is_file()]
    [f.rename(cp_fold / f'{f.name}') for f in cp_files]

    return config


def unfreeze(**config):
    config['learn'].unfreeze()
    return config


def make_one_hot_df_pipe(**config):
    """
    Follow Pipeline Process template in `prlab.fastai.pipeline.pipeline_control_multi`.
    Consider to REPLACE `prlab.medical.data_helper.make_one_hot_df`, two pipe have similar idea but different implement,
    this implement try to general case of tabular.
    Call after `prlab.medical.data_helper.data_load_df`
    Update some field in df and make config to work with one-hot
    :param config:
    :return:
    """
    ndf = encode_and_bind(config['df'], config['cat_names'], keep_old=False)
    config['df'] = ndf

    # update cat_names to [] and cont_names to all fields (except fold)
    cont_names = config['cont_names']

    # new columns from cat_names that made in encode_and_bind
    # the new names will be COLNAME_{VALUE}
    all_col_names = config['df'].select_dtypes(include=[np.number]).columns.tolist()
    new_cols_names = [name for name in all_col_names if
                      np.any(["{}_".format(cat_name) in name for cat_name in config['cat_names']])]
    cont_names.extend(new_cols_names)

    config['cat_names'], config['cont_names'] = [], cont_names

    return config


from prlab.common.utils import PipeClassCallWrap, PipeClassWrap

PipeClassCallWrap, PipeClassWrap  # move to prlab.common.utils, just support old call

# define some popular pipe that can be widely use
default_conf_pipeline = {
    'process_pipeline_2': [
        # leave 0 and 1 for some early configure
        device_setup,
        general_configure,
    ],

    'process_pipeline_12': [
        # 2-11 is for data load and model build
        learn_general_setup,
        resume_learner,
    ],
    'process_pipeline_102': [make_report_general],
}
