import copy
import datetime
import importlib
import json
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

# add ls function to Path for easy to use
Path.ls = lambda x: list(x.iterdir())


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


def convert_to_obj_or_fn(val, **params):
    if isinstance(val, dict):
        return {k: convert_to_obj_or_fn(v, **params) for k, v in val.items()}
    if isinstance(val, (tuple, list)):
        return [convert_to_obj_or_fn(o, **params) for o in val]
    if isinstance(val, str):
        if val.rsplit('.', 1)[-1][0].isupper():  # this is class, then call to make object
            return load_func_by_name(val)[0](**params)
        else:
            return load_func_by_name(val)[0]
    return val


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

    mod = importlib.import_module(mod_name)
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
        out.append(fn(fold=fold, **config))

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
    if isinstance(npy_arr, (np.float, np.float64, np.int, np.int64)):
        return fm.format(npy_arr)

    # else, then array
    arr_out = [npy_arr_pretty_print(o, fm=fm) for o in npy_arr]
    to_print = "\t".join(arr_out) if len(npy_arr.shape) < 2 else "\n".join(arr_out)
    return to_print


def test_make_check_point_folder():
    print(make_check_point_folder("/tmp"))


def test_map_str_list_to_type():
    print(map_str_list_to_type("[1 3 4 6]"))
    print(map_str_list_to_type("[1 3 4 6]", dtype=int))


if __name__ == '__main__':
    test_map_str_list_to_type()
