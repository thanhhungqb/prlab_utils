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

import nltk
import sklearn
from fastai.vision import *

from outside.scikit.plot_confusion_matrix import plot_confusion_matrix
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


# ************* model *************************************
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


# *************** DATA **********************************
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


# *************** DATA **********************************
def make_report_cls(learn, **config):
    """
    Newer, simpler version of `prlab.emotion.ferplus.sr_stn_vgg_8classes.run_report`,
    better support `Pipeline Process template`
    This is for CLASSIFICATION, report on accs and f1 score.
    y labels could be one-hot or prob mode.
    Run TTA 3 times and make report to screen, reports.txt and results.npy
    :param learn:
    :param config: contains data_test store test in valid mode, tta_times (if have)
    :return: as description of `Pipeline Process template` including learn and config (not update in this func)
    """
    print('starting report')
    cp = config['cp']

    data_current = learn.data  # make backup of current data of learn
    learn.data = config['data_test']
    print(config['data_test'])

    accs, to_save = [], {}
    for run_num in range(config.get('tta_times', 3)):
        ys, y = learn.TTA(ds_type=DatasetType.Valid, scale=config.get('test_scale', 1.10))

        ys_npy, y_npy = ys.numpy(), y.numpy()
        ys_labels = np.argmax(ys_npy, axis=-1)
        #  support both one-hot and prob mode
        y_labels = y_npy if isinstance(y_npy.flat[0], (np.int64, np.int)) else np.argmax(y_npy, axis=-1)

        accs.append(nltk.accuracy(ys_labels, y_labels))
        f1 = sklearn.metrics.f1_score(y_labels, ys_labels, average='macro')  # micro macro
        to_save['time_{}'.format(run_num)] = {'ys': ys_npy, 'y': y_npy, 'acc': accs[-1], 'f1': f1}
        print('run', run_num, accs[-1], 'f1', f1)

        _, fig = plot_confusion_matrix(y_labels, ys_labels,
                                       classes=config.get('label_names', None),
                                       normalize=config.get('normalize_cm', True),
                                       title='Confusion matrix')
        fig.savefig(cp / 'run-{}.png'.format(run_num))

    stats = [np.average(accs), np.std(accs), np.max(accs), np.median(accs)]
    (config['model_path'] / "reports.txt").open('a').write('{}\t{}\tstats: {}\tf1: {}\n'.format(cp, accs, stats, f1))
    print('3 results', accs, 'stats', stats)

    np.save(cp / "results", to_save)

    # roll back for data in learn
    learn.data = data_current
    return learn, config
