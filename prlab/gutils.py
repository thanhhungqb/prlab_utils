import copy
import datetime
import importlib
import json
import logging
import random
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

# add ls function to Path for easy to use
from outside.utils import find_modules

Path.ls = lambda x: list(x.iterdir())
prlab_modules = ['prlab.{}'.format(o) for o in find_modules(['prlab'])]


def clean_str(s): return '_'.join(s.split()).replace('/', '')


def set_if(d, k, v, check=None):
    """
    Set value of k to v if check in dict d, check mostly None
    :param d: dict
    :param k: key
    :param v: value
    :param check: value to check, default is None (mean fill if None)
    :return:
    """
    d[k] = v if d.get(k, check) == check else d[k]
    return d


def lazy_object_fn_call(val, **params):
    """val should be object or object after fn call, if not then convert"""
    if isinstance(val, (list, tuple, dict, str)):
        val = convert_to_obj_or_fn(val, lazy=True, **params)
    return val


def convert_to_obj_or_fn(val, lazy=False, **params):
    """
    Convert data structure to Class/Fn/Object/Fn call
    `object` is reversed word to make object or function call
    `object_lazy` is reversed word to make object/fn when needed (lazy)
    coding style: class name should be start with UPPER letter
    :param val: current data to convert
    :param lazy: true/false, true if run in the lazy mode, false if in fresh mode, e.g. at beginning
    :param params: similar global variable e.g. whole dict
    :return:
    """
    if isinstance(val, dict):
        return {k: convert_to_obj_or_fn(v, **params) for k, v in val.items()}
    if isinstance(val, list):
        # ["object", class_name, dict] reverse for object make (as tuple below)
        # because object is reverse word, and never use to function or class, safe to use here to mark
        if len(val) > 1 and val[0] == "object":
            return convert_to_obj_or_fn(tuple(val[1:]), **params)
        # some object/fn just call when needed (or at special time when all params available)
        # simply keep as it to use later
        if len(val) > 1 and val[0] == "object_lazy":
            return convert_to_obj_or_fn(tuple(val[1:]), **params) if lazy else val
        return [convert_to_obj_or_fn(o, **params) for o in val]

    if isinstance(val, tuple):
        # object make (class_name, dict), omit form (class_name, params, dict*), len=2
        call_class = load_func_by_name(val[0])[0]
        new_params = {}
        new_params.update(params)
        # TODO bug:
        #   in some cases, val[-1] contains some string like class/func (e.g. a.b.c) but want to keep this form
        #   instead convert to object/func
        new_params.update(convert_to_obj_or_fn(val[-1], **params))
        return call_class(**new_params)

    if isinstance(val, str):
        try:
            if val.rsplit('.', 1)[-1][0].isupper():  # this is class, then call to make object
                if val.rsplit('.', 1)[-1].isupper():
                    # all letter in name upper then constant, e.g. variable, lambda func
                    return load_func_by_name(val)[0]
                return load_func_by_name(val)[0](**params)
            else:
                return load_func_by_name(val)[0]
        except:
            return val  # just normal str, not class or function
    return val


def check_convert_to_obj_or_fn(val, level=0, **params):
    """
    Similar to `prlab.gutils.convert_to_obj_or_fn` but just check to make sure module and class/function available.
    'object', 'object_lazy' is reversed words
    :param val: current data to convert
    :param params: similar global variable e.g. whole dict
    :return:
    """
    if isinstance(val, dict):
        [check_convert_to_obj_or_fn(v, **params) for _, v in val.items()]

    if isinstance(val, list):
        # ["object", class_name, dict] reverse for object make (as tuple below)
        # because object is reverse word, and never use to function or class, safe to use here to mark
        if len(val) > 1 and val[0] in ["object", "object_lazy"]:
            check_convert_to_obj_or_fn(tuple(val[1:]), **params)
        else:
            [check_convert_to_obj_or_fn(o, **params) for o in val]

    if isinstance(val, tuple):
        # object make (class_name, dict), omit form (class_name, params, dict*), len=2
        assert len(val) == 2

        call_class = load_func_by_name(val[0])[0]
        assert not isinstance(call_class, str)  # must be object
        # TODO bug: similar as `convert_to_obj_or_fn`
        check_convert_to_obj_or_fn(val[-1], level=level + 1, **params)

    if isinstance(val, str):
        # TODO how about str
        if level == 0:
            assert not isinstance(load_func_by_name(val)[0], str)


def convert_to_obj(val, **params):
    """
    Convert to function call or object
    :param val: mostly str or list of str
    :param params: params to call to make obj or function call
    :return:
    """
    if isinstance(val, dict):
        return {k: convert_to_obj(v, **params) for k, v in val.items()}
    if isinstance(val, (tuple, list)):
        return [convert_to_obj(o, **params) for o in val]
    if isinstance(val, str):
        return load_func_by_name(val)[0](**params)
    return val


def convert_to_fn(val, **kwargs):
    """
    Convert to function (not call, just fn) while `convert_to_obj` do the call
    :param val: mostly str or list of str
    :return:
    """
    if isinstance(val, dict):
        return {k: convert_to_fn(v) for k, v in val.items()}
    if isinstance(val, (tuple, list)):
        return [convert_to_fn(o) for o in val]
    if isinstance(val, str):
        return load_func_by_name(val)[0]
    return val


def constant_map_dict(dic, cons=None, excluded=None):
    """
    To mapping contants values and reused in JSON file (SELF CONSTANTS MAP).
    map for dict, and array and keep all others types
    :param dic: with format {constants:{}, ...} else cons must be provided, or list
    :param cons: if None then dic must be dic with constants
    :param excluded: list or set of key OMIT
    :return: new dic with replace constants
    """
    if cons is None:
        assert isinstance(dic, dict)
        cons = dic['constants']
    excluded = [] if excluded is None else excluded
    excluded = [excluded] if isinstance(excluded, str) else excluded
    excluded = set(excluded) if not isinstance(excluded, set) else excluded

    if isinstance(dic, dict):
        ret = {k: constant_map_dict(v, cons=cons, excluded=excluded) for k, v in dic.items()}
    elif isinstance(dic, list):
        ret = [constant_map_dict(k, cons=cons, excluded=excluded) for k in dic]
    else:
        # do nothing
        ret = cons.get(dic, dic) if isinstance(dic, str) and dic not in excluded else dic

    return ret


def to_json_writeable(js):
    """
    Convert js in JSON to JSON writeable to file, means only str, number, list
    :param js:
    :return:
    """
    if isinstance(js, (int, float)):
        return js
    if isinstance(js, str):
        return js
    if isinstance(js, (tuple, list)):
        return [to_json_writeable(o) for o in js]
    if isinstance(js, dict):
        return {k: to_json_writeable(v) for k, v in js.items()}

    # other wise, try to convert to str
    return str(js)


def load_json_text_lines(file_name):
    """
    Load text file, each line contains json
    :param file_name:
    :return: {line_count: JS}
    """
    with open(file_name) as f:
        lines = f.readlines()
        out = {}
        for i, text in enumerate(lines):
            out[i] = json.loads(text)

    return out


def backup_file(file_path, n=0):
    file_path = file_path if isinstance(file_path, Path) else Path(file_path)
    if file_path.is_file():
        next_f = file_path.parent / f'{file_path.name}.{n}'
        if next_f.is_file():
            return backup_file(file_path, n + 1)
        else:
            file_path.rename(next_f)
            return next_f
    else:
        return None


def make_check_point_folder(config={}, cp_base=None, cp_name=None):
    """
    make a checkpoint folder and csv log
    UPDATE best_name only 'best' instead full path, use cp for full
    :param config:
    :param cp_base: None for path, but usually in path/models
    :param cp_name:
    :return: path of cp, best cp, csv file path
    """
    path = config['path'] if config.get('model_path', None) is None else config['model_path']
    path = path if isinstance(path, Path) else Path(path)
    if cp_name is None:
        cp_name = "{}".format(time.strftime("%Y.%m.%d-%H.%M"))

    cp_path = path / cp_base if cp_base is not None else path
    cp_path = cp_path / cp_name
    best_name = "best"
    csv_log = cp_path / "loger"
    config.update({'cp': cp_path, 'best_name': best_name, 'csv_log': csv_log})

    # write configure to easy track later, could not write JSON because some object inside config
    cp_path.mkdir(parents=True, exist_ok=True)
    txtwriter = cp_path / 'configure.txt'
    txtwriter.write_text(str(config)) if not txtwriter.is_file() else None
    try:
        backup_file(cp_path / 'configure.json')  # backup old file in this folder
        with open(cp_path / 'configure.json', 'w') as f:
            json.dump(to_json_writeable(config), f, indent=2)
    except Exception as e:
        print('warning: ', e)

    return cp_path, best_name, csv_log


def map_str_prob(str_prob):
    """
    Mostly load column prob from csv or text
    :param str_prob: '[5, 3, 0, 1, 1]'
    :return: [0.5 0.3 0 .1 .1]
    """
    s = [o.replace('[', '').replace(']', '').replace(',', '').split() for o in str_prob]
    f = [[float(o) for o in row] for row in s]
    f = np.array(f)
    fn = preprocessing.normalize(f, axis=1, norm='l1')
    return fn


def map_str_list_to_type(str_lst, dtype=float):
    """
    mostly use when read list present in str format of float
    :param str_lst: [o1 o2 o3 o4 ...]
    :param dtype: type to convert, default is float
    :return:
    """
    s = str_lst.replace('[', '').replace(']', '').replace(',', '').strip().split()
    f = [dtype(o) for o in s]
    return f


def get_file_rec(path):
    """
    Get all file recursive
    :param path:
    :return: files
    """
    path = path if isinstance(path, Path) else Path(path)
    if path.is_file():
        return [path]

    ls = path.ls()
    files = [o for o in ls if o.is_file()]
    for cfolder in ls:
        if cfolder.is_dir():
            files.extend(get_file_rec(cfolder))

    return files


def get_folder_level(root_path, depth=1):
    """
    return all folder, sub-folder with depth
    :param root_path:
    :param depth: 1 for current folder only, 2 for all folder in sub-folder of current
    :return: [Path]
    """
    p = Path(root_path)
    curr_level = [p]
    all_level = [p]
    for level in range(depth):
        next_level = []
        for xp in curr_level:
            next_level.extend([o for o in xp.iterdir() if o.is_dir()])

        all_level.extend(next_level)
        curr_level = next_level

    return all_level


def best_match_path(path_list, name):
    """
    Find the best match of name in path_list.
    Wide use when name is an ID and ID are a part of Path name, e.g. CT_$name, $name.npy
    TODO now simple implement check name is in part (full name)
    :param path_list:
    :param name:
    :return:
    """
    candidate = []
    for o in path_list:
        if name in str(o):
            candidate.append(o)

    if len(candidate) == 0:
        return None
    if len(candidate) == 1:
        return candidate[0]
    return candidate[0]


def load_df_img_data(path, index_keys='filename', bb_key='bb'):
    """
    Load DataFrame from csv that has filename as index and bb
    For easy to track, bb should be in format [center_x center_y width heigh]*n
    :param path:
    :param index_keys:
    :param bb_key:
    :return:
    """
    data = pd.read_csv(path, delimiter=',')
    data = data.set_index(index_keys)

    float_bb = [map_str_list_to_type(o) for o in data[bb_key]]
    data[bb_key] = float_bb

    return data


def resample_to_distribution(x, y, dis=[]):
    """
    Resampling to data x and label y to correct distribution
    :param x: data X
    :param y: label y, same size with x or None
    :param dis: list int number, number of elements get each labels, if y number then list, else dict
    :return: new_x, new_y
    """
    x = np.array(x) if isinstance(x, list) else x
    y = np.array(y) if isinstance(y, list) else y
    dis = {i: val for i, val in enumerate(dis)} if isinstance(dis, list) else dis

    assert len(x) == len(y)
    label_set = set(y)
    assert len(label_set) == (len(dis) if not isinstance(dis, dict) else len(dis.keys()))

    ss = {label: x[y == label] for label in label_set}
    x_new, y_new = np.array([]), []
    for label in label_set:
        x_new = np.append(x_new, np.random.choice(ss[label], size=dis[label]))
        y_new.extend([label] * dis[label])

    return sklearn.utils.shuffle(np.stack(x_new), np.stack(y_new))


def obj_func_str_split(name_str, **kwargs):
    """
    Split object class/function name from str represent in three form:
    - raw name only, DO NOTHING, e.g.
        pcam.models.simple_transfer_model_xavier
        prlab.model.srss.SRSSVGGModel
    - func form (str of function name), CUT, e.g.
        <function prob_acc at 0x7efb63ec5ea0>
        <function prlab.fastai.utils.prob_acc(target, y, **kwargs)>
    - object form (str of object), CUT, e.g.
        <prlab.emotion.ferplus.data_helper.FerplusDataHelper object at 0x7efb63ed50b8>
    :param name_str:
    :param kwargs:
    :return:
    """
    if name_str.startswith('<'):
        if name_str.endswith('>'):
            # this form is function/object in memory, then have "at ..." at the end of str
            name_str = name_str[1:-1]
            arr = name_str.split()
            if not arr[-1].endswith(')') and arr[2] != 'at':
                raise Exception('Wrong represent string of function, object or class ')
            if arr[0] == 'function':
                # form <function fname at ...>
                return arr[1].split('(')[0]
            elif arr[1] == 'object':
                # form <obj_name object at ...>
                return arr[0]
            else:
                raise Exception('Wrong represent string of function, object or class ')
        else:
            raise Exception('Wrong represent string of function, object or class ')
    else:
        # this form raw, function or class name
        return name_str


def load_func_by_name(func_str):
    """
    Load function/object/class by full name, e.g. pcam.models.simple_transfer_model_xavier
    :param func_str: package.name
    :return: fn, module
    """
    func_str = obj_func_str_split(func_str)
    mod_name, func_name = func_str.rsplit('.', 1) if '.' in func_str else (__name__, func_str)
    if mod_name == '':
        # for related form .fn
        mod_name = __name__

    try:
        mod = importlib.import_module(mod_name)
    except:
        # case of module.class.fn or module.class.var
        mod1, fn1 = mod_name.rsplit('.', 1)
        mod = importlib.import_module(mod1)
        mod = getattr(mod, fn1)

    func = getattr(mod, func_name)
    return func, mod


def save_config_info(config, save_path='.'):
    """
    Save configure (dict) to file after add more detail about date, time, git, ...
    :param config:
    :param save_path:
    :return:
    """
    current_dt = datetime.datetime.now()
    config_copy = copy.deepcopy(config)
    config_copy['save time'] = str(current_dt)

    try:
        # add git info
        from git import Repo
        repo = Repo('.')
        master = repo.heads.master
        config_copy['git_info'] = {}
        config_copy['git_info']['branch'] = repo.active_branch.name
        config_copy['git_info']['last-hash'] = repo.active_branch.log()[-1].newhexsha
        config_copy['git_info']['last-master-hash'] = master.log()[-1].newhexsha
        config_copy['git_info']['remote-url'] = repo.remotes[0].url
    except:
        pass

    with open("{}/train_config.json".format(save_path), "w") as fw:
        json.dump(config_copy, fw, indent=2, sort_keys=True)


def parse_extra_args_click(ctx, is_digit_convert=True):
    """
    Parse extra args from `Click` option.
    Do some default cast before return: number, ?, str (default)
    note key must be in long form (--key), DO NOT support sort key

    Usage:
    @click.command(name='command_run_1', context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ))
    @click.option('-r', '--run', default='run-00', help='run id')
    @click.pass_context
    def command_run(ctx, run):
        extra_dict = parse_extra_args_click(ctx)
        ...

    :param ctx: ctx from click command
    :param is_digit_convert: convert to number
    :return: dict of {name:value}
    """
    out = {ctx.args[i][2:]: ctx.args[i + 1] for i in range(0, len(ctx.args), 2)}
    is_true_fn = lambda bstr: bstr.lower() in ['true']

    for key in out.keys():
        val = out[key]

        if val.isdigit():
            out[key] = int(val)
        elif val.replace('.', '', 1).isdigit():
            out[key] = float(val)
        elif val.upper() in ['TRUE', 'FALSE']:
            out[key] = is_true_fn(val)

    return out


@click.command(name='command_run', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--run_id', default='run-00', help='run id')
@click.option('--json_conf', default=None, help='json configure file')
@click.pass_context
def command_run(ctx, run_id, json_conf):
    """
    config to run command with callable. All param will pass to callable when call
    :param ctx:
    :param run_id:
    :param json_conf: load base configure from json file
    :return:
    """
    print('run ID', run_id)

    config = {}
    if json_conf:
        with open(json_conf) as fp:
            config = json.load(fp=fp)
        config['json_conf'] = json_conf

    extra_args = parse_extra_args_click(ctx)
    config.update(**extra_args)

    # all other configure json_conf2, ... will be in config too
    for idx in range(20):
        additional_conf = 'json_conf{}'.format(idx)
        if config.get(additional_conf, None) is not None:
            with open(config[additional_conf]) as fp:
                config2 = json.load(fp=fp)
                config.update(**config2)

    #  one more time to override configure from command line
    config.update(**extra_args)

    # load function by str
    fn, mod_ = load_func_by_name(config['call'])
    out = fn(**config)

    print(out)


@click.command(name='run_k_fold', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option('--run_id', default='run-00', help='run id')
@click.option('--json_conf', default=None, help='json configure file')
@click.pass_context
def run_k_fold(ctx, run_id, json_conf):
    """
    config to run k-fold with a callable command. All param will pass to callable when call.
    For complex configure, it should be in JSON file for easy to load and reuse.
    :param ctx:
    :param run_id:
    :param json_conf: load base configure from json file
    :return:
    """
    print('run ID', run_id)

    config = {}
    if json_conf:
        with open(json_conf) as fp:
            config = json.load(fp=fp)

    extra_args = parse_extra_args_click(ctx)
    config.update(**extra_args)

    # all other configure json_conf2, ... will be in config too
    for idx in range(20):
        additional_conf = 'json_conf{}'.format(idx)
        if config.get(additional_conf, None) is not None:
            with open(config[additional_conf]) as fp:
                config2 = json.load(fp=fp)
                config.update(**config2)

    #  one more time to override configure from command line
    config.update(**extra_args)

    config['run_id'] = run_id
    set_if(config, 'k', 5)

    print('final configure', config)
    print('run {} folds'.format(config['k']))
    # load function by str
    fn, mod_ = load_func_by_name(config['call'])

    out = []
    k_start = config.get('k_start', 0)
    for fold in range(k_start, config['k'] + k_start):
        config.update({'fold': fold})
        out.append(fn(**config))
        # support for dict['output'], correctly follow the template
        if isinstance(out[-1], dict):
            out[-1] = out[-1]['output']

    print(out)
    try:
        print('simple statistical', np.mean(out), np.std(out))
    except:
        pass

    return out


def encode_and_bind(df, features, keep_old=False, drop_first=False, **kwargs):
    """
    One-hot vector for dataframe
    Credit: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    Modified version to work with list of features instead one
    :param df:
    :param features: str or list(str)
    :param keep_old:
    :param drop_first: if true then dummies, else one-hot
    :return:
    """
    # if only one feature then could pass as str instead [str]
    if not isinstance(features, list):
        features = [features]

    dummies = [pd.get_dummies(df[[feature]], drop_first=drop_first) for feature in features]
    new_df = pd.concat([df] + dummies, axis=1)

    if not keep_old:
        new_df = new_df.drop(features, axis=1)

    return new_df


def column_map(df, feature_names, new_val, keep_old=False, **kwargs):
    """
    Mapping column values
    :param df:
    :param feature_names: [field_name]
    :param new_val: {field_name; {val:new_val}, new_val is value or 1-D array, must same length
    :param keep_old:
    :param kwargs:
    :return:
    """
    if not isinstance(feature_names, list):
        feature_names = [feature_names]

    new_df = pd.DataFrame()
    for field_name in feature_names:
        print('do for field', field_name)
        for k, v in new_val[field_name].items():
            xlen = len(v) if isinstance(v, list) else 1
            # xlen = len(new_val[field_name])
        x_names = ['{}_{}'.format(field_name, o) for o in range(xlen)]

        na_value = new_val[field_name]["#na#"]
        for o in df[field_name]:
            if new_val[field_name].get(o, None) is None:
                new_val[field_name][o] = na_value

        df[field_name] = df[field_name].str.strip()
        col = df[field_name].map(new_val[field_name])
        col.fillna(pd.Series(na_value), inplace=True)

        new_df[x_names] = pd.DataFrame(col.tolist())

    new_df = pd.concat([df, new_df], axis=1)
    if not keep_old:
        new_df = new_df.drop(feature_names, axis=1)
    return new_df


def npy_arr_pretty_print(npy_arr, fm='{:.4f}'):
    """
    Pretty print for npy array
    :param npy_arr:
    :param fm: format string, default is for 4 digits float
    :return:
    """
    if isinstance(npy_arr, (np.float, np.float64, np.int, np.int64, np.int16)):
        return fm.format(npy_arr)
    if isinstance(npy_arr, (int, float)):
        return fm.format(npy_arr)

    # else, then array
    arr_out = [npy_arr_pretty_print(o, fm=fm) for o in npy_arr]
    if not isinstance(npy_arr, np.ndarray):
        npy_arr = np.array(npy_arr)
    to_print = "\t".join(arr_out) if len(npy_arr.shape) < 2 else "\n".join(arr_out)
    return to_print


def fix_name_func_class(**config):
    """
    Fix name of func or class/obj when write to a JSON that it's not full path name (only class/func name)
    Try to load all names in prlab module and match.
    Error form:
        "<function norm_weights_acc at 0x7fa18fad5440>"
    Follow Pipeline Process template, Could add to the pipeline (if pass dict)
    :param config:
    :return:
    """

    for k in config.keys():
        val = config[k]
        if isinstance(val, dict):
            config[k] = fix_name_func_class(**val)
        if isinstance(val, (tuple, list)):
            config[k] = [fix_name_func_class(x=o)['x'] for o in val]

        if isinstance(val, str):
            if val.startswith('<function') and val.endswith('>'):
                name_fn = obj_func_str_split(val)
                arr = name_fn.split('.')
                if len(arr) == 1:
                    name_fn = arr[0]
                    la = []
                    for mod_name in prlab_modules:
                        try:
                            if hasattr(importlib.import_module(mod_name), name_fn):
                                la.append(mod_name)
                        except:
                            """there are some module not completed then raise exception when load"""

                    if len(la) == 0:
                        raise Exception('there are no function {} in prlab submodule'.format(name_fn))
                    if len(la) > 1:
                        print('there are many version for {} in prlab.* including {}'.format(name_fn, ';'.join(la)))

                    config[k] = '{}.{}'.format(la[0], name_fn)
                    print('update', k, config[k])

    return config


def balanced_sampler(labels: list, n_each=1000, replacement=False):
    """
    Balanced Sampler from labels and get n_each for each label kind.
    :param labels: list of label (int or str)
    :param n_each:
    :param replacement:
    :return: list of idx that sampler
    """
    n = len(labels)
    pos = [o for o in range(n)]
    random.shuffle(pos)
    if replacement:
        # [NOTE: n*3 is a trick]
        # that seem enough, but in general case should think about infinite this list
        # but catch when all labels full below
        # this implement want to keep this simple
        pos = [random.randint(0, n) for _ in range(n * 3)]

    selected_pos = []
    count_l = {}
    for p in pos:
        label_p = labels[p]
        if count_l.get(label_p, 0) < n_each:
            count_l[label_p] = count_l.get(label_p, 0) + 1
            selected_pos.append(p)

    return selected_pos


def avg_std_3d(arr):
    """
    3D average and std, mostly use for list of images
    :param arr: [n x w x h]
    :return: [w x h], [w x h] avg and std
    """
    size = arr.shape
    avg, std = np.zeros((size[1], size[2])), np.zeros((size[1], size[2]))
    for row in range(size[1]):
        for col in range(size[2]):
            avg[row][col] = np.average(arr[:, row, col])
            std[row][col] = np.std(arr[:, row, col])
    return avg, std


def summary_2d(arr, func=np.average):
    """
    The summary to 2d from the 3D input
    :param arr: [n x w x h]
    :param func: numpy function, work with 1D (along to n direction)
    :return: [w x h]
    """
    size = arr.shape
    summary = np.zeros((size[1], size[2]))
    for row in range(size[1]):
        for col in range(size[2]):
            summary[row][col] = np.average(arr[:, row, col])
    return summary


def merge_xlsx(files, merged_file=None):
    """
    Merge multi xlsx file to one, all file must have a same column name.
    Just simple add row by row.
    TODO now just support one sheet (first) in the excel file
    :param files: file paths
    :param merged_file: if given then save to new file
    :return:
    """
    dfs = [pd.read_excel(file) for file in files]
    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.to_excel(merged_file) if merged_file is not None else None

    return merged_df


class NameSpaceDict(dict):
    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self.pkey = 'parent'

    def get(self, k, d=None):
        ret = super().get(k, d=None)
        if ret is not None:
            return ret
        # get from parent if have
        if isinstance(super.get(self.pkey), dict):
            return super.get(self.pkey).get(k, d)

        return d


# ============================ PIPE =================================
class PipeClassWrap:
    """
    Convert/Wrap pipe function to class style.
    Note that class style can be easy to use with object or object_lazy with the custom params
    Usage:

        def function(**kwargs):
            print("GeeksforGeeks")
            print(kwargs)

        obj = PipeClassWrap(fn=function,test='a',o='b')
        obj(test='override')
    """

    def __init__(self, fn, **config):
        self.fn = lazy_object_fn_call(fn, **config)
        self.config = config

    def __call__(self, *args, **kwargs):
        # update and override with stored config
        params = {}
        params.update(self.config)
        params.update(kwargs)

        return self.fn(*args, **params)


class PipeClassCallWrap:
    """
    Wrap a function call with return to a pipe call with update configure
    """

    def __init__(self, fn, ret_name='out', params=None, map_name=None, fixed_params=None, **config):
        self.fn = lazy_object_fn_call(fn, **config)
        self.fn = self.fn if callable(self.fn) else eval(self.fn)
        self.ret_name = ret_name
        self.params = params if params is not None else {}
        self.map_param_name = {} if map_name is None else map_name
        self.fixed_params = fixed_params  # support for class/func that has fixed number params (does not allow **kw)

    def __call__(self, *args, **config):
        new_params = {k: config.get(v) for k, v in self.map_param_name.items()}
        to_pass = {**config, **self.params, **new_params}
        if self.fixed_params is not None:
            to_pass = {k: v for k, v in to_pass.items() if k in self.fixed_params}
        config[self.ret_name] = self.fn(*args, **to_pass)
        return config

    def __repr__(self):
        return f"PipeClassCallWrap ( {str(self.fn)} )"


def get_pipes(base_name, n_max=1000, **config):
    """
    get all pipes and flatten to one list. Pipe in form {base_name}-{i} with i<n_max
    :param base_name: e.g. preprocessing-pipeline
    :param n_max:
    :param config:
    :return:
    """
    p_names = [f'{base_name}-{i}' for i in range(n_max) if f'{base_name}-{i}' in config.keys()]

    logger = logging.getLogger(__name__)

    pipes = []
    for p_n in p_names:
        pipes.append(
            PipeClassCallWrap(fn=lambda msg, **kw: logger.info(msg),
                              params={'msg': f"run pipeline {p_n}"}))
        pipes.extend(config[p_n])

    # support old version without {i} part
    if len(pipes) == 0 and base_name in config.keys():
        pipes = [base_name]

    return pipes


# ============================ END OF PIPE =================================

# some normalization function
normalize_norm = lambda slices, **kw: (slices - slices.mean()) / slices.std()
normalize_0_1 = lambda slices, **kw: (slices - slices.min()) / (slices.max() - slices.min())
normalize_n1_1 = lambda slices, **kw: (slices * 2 - slices.max() - slices.min()) / (slices.max() - slices.min())


def test_make_check_point_folder():
    print(make_check_point_folder("/tmp"))


def test_map_str_list_to_type():
    print(map_str_list_to_type("[1 3 4 6]"))
    print(map_str_list_to_type("[1 3 4 6]", dtype=int))


if __name__ == '__main__':
    test_map_str_list_to_type()
