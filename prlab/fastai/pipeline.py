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
import deprecation
import nltk
import sklearn
from fastai.vision import *

from outside.scikit.plot_confusion_matrix import plot_confusion_matrix
from outside.stn import STN
from outside.super_resolution.srnet import SRNet3
from prlab.fastai.utils import general_configure, base_arch_str_to_obj
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

    return learn, config


# ************* model func *******************
def basic_model_build(**config):
    """
    Build basic model: vgg, resnet, ...
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    base_arch = base_arch_str_to_obj(config.get('base_arch', 'vgg16_bn'))
    learn = cnn_learner(data=config['data_train'], base_arch=base_arch, model_dir=config['cp'])
    return learn, learn.layer_groups


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


# *************** TRAINING PROCESS **********************************
def training_adam_sgd(learn, **config):
    """
    A training process that use both adam and sgd, adam for first epochs and later is sgd.
    This process seems quicker to sgd at the begin, but also meet the best result.
    Note, for this process, `best` model could not be load correctly, then resume should be careful and addition work to
    load weights only.
    Note: at the beginning, `learn` mostly configure with ADAM
    :param learn:
    :param config:
    :return:
    """
    best_name = config.get('best_name', 'best')
    data_train, model, layer_groups = config['data_train'], config['model'], config['layer_groups']

    # TODO see note in header
    # resume

    learn.save(best_name)  # TODO why need in the newer version of pytorch

    learn.data = data_train

    lr = config.get('lr', 5e-3)
    learn.fit_one_cycle(config.get('epochs', 30), max_lr=lr)

    learn.save('best-{}'.format(config.get('epochs', 30)))

    torch.save(learn.model.state_dict(), config['cp'] / 'e_{}.w'.format(config.get('epochs', 30)))
    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    # SGD optimize
    opt = partial(optim.SGD, momentum=0.9)
    learn = Learner(data_train, model=model, opt_func=opt, metrics=config['metrics'],
                    layer_groups=layer_groups,
                    model_dir=config['cp'])
    learn.model.load_state_dict(torch.load(config['cp'] / 'e_{}.w'.format(config.get('epochs', 30))))

    learn, config, *_ = learn_general_setup(learn, **config)
    learn.data = data_train

    lr = config.get('lr_2', 1e-4)
    learn.fit_one_cycle(config.get('epochs_2', 30), max_lr=lr)

    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    return learn, config


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

    learn.unfreeze()
    lr_2 = config.get('lr_2', lr)
    learn.fit_one_cycle(config.get('epochs_2', 30), max_lr=lr_2)

    torch.save(learn.model.state_dict(), config['cp'] / 'final.w')

    return learn, config


# *************** REPORT **********************************
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

    accs_str = ' '.join(['{0:.4f}'.format(o) for o in accs])
    stats_str = ' '.join(['{0:.4f}'.format(o) for o in stats])
    (config['model_path'] / "reports.txt").open('a').write(
        '{}\t{}\tstats: {}\tf1: {:.4f}\n'.format(cp, accs_str, stats_str, f1))
    print('3 results', accs_str, 'stats', stats_str)

    np.save(cp / "results", to_save)

    # roll back for data in learn
    learn.data = data_current
    return learn, config


# *************** WEIGHTS LOAD **********************************
def srnet3_weights_load(learn, **config):
    """
    Use together with `prlab.fastai.pipeline.sr_xn_stn` to load pre-trained weights for
    `outside.super_resolution.srnet.SRNet3`
    Note: should call after model build.
    Follow Pipeline Process template.
    :param learn:
    :param config:
    :return:
    """
    xn = config.get('xn', 2)
    xn_weights_path = '/ws/models/super_resolution/facial_x{}.pth'.format(xn)
    xn_weights_path = xn_weights_path if config.get('xn_weights_path', None) is None else config['xn_weights_path']
    [learn.model[i].load_state_dict(torch.load(xn_weights_path))
     for i in range(3) if isinstance(learn.model[i], SRNet3)]

    return learn, config


def load_weights(learn, **config):
    learn.model.load_state_dict(torch.load(config['cp'] / 'final.w'))
    return learn, config


def resume_learner(learn, **config):
    """
    Resume, load weight from final.w or best_name in this order.
    Note: best_name maybe newer than final.w, then will override if both found
    :param learn:
    :param config:
    :return:
    """
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

    config['learn'] = learn
    return learn, config


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
    return None, config


def device_setup(**config):
    """
    To set train/test on cpu working space.
    If use, must be before all data-loader, model build, etc. in the pipeline
    Follow Pipeline Process template.
    :param config:
    :return:
    """
    defaults.device = torch.device(config.get('device', 'cpu'))
    return None, config
