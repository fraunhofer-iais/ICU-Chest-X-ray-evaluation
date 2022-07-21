import copy
import itertools
from datetime import datetime as dt
from functools import partial, reduce
from importlib import import_module
from logging import Logger
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import yaml


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


class GridSearchList(list):
    pass


def gs(loader, node):
    seq = loader.construct_sequence(node)
    return GridSearchList(seq)


def do_not_parse(loader, node):
    return None


yaml.add_constructor("!join", join)
yaml.add_constructor("!gs", gs)
yaml.add_constructor("!do_not_parse", do_not_parse)


def load_params(path: str) -> Dict:
    with open(path, "r") as f:
        params = yaml.full_load(f)
    return params


def write_params(params: Dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(params, f)


Parameter = Tuple[str, Any]
ParameterCombination = List[Parameter]
ParameterPool = List[ParameterCombination]


def unpack_gs_parameters(params: Dict, prefix: str = None) -> ParameterPool:
    """
    Collects all grid search parameters in the parameter dict.
    Example:
    params = {
        'a': 1,
        'b: GridSearchList([1, 2]),
        'c': {
            'ca': [1, 2],
            'cb': GridSearchList([3, 4])
        }
    unpack_gs_parameters(params) = [
        [('c.cb', 3), ('c.cb', 4)],
        [('b', 1), ('b', 2)]
    ]

    Parameters
    ----------
    params Dict of parameters
    prefix Str only used recursively by this function

    Returns
    -------
    ParameterPool, i.e. List of parameter configurations, i.e. List of List of Tuple[ParameterName, ParameterValue]

    """
    gs_params = []
    for key, value in params.items():
        if isinstance(value, GridSearchList):
            if prefix is not None:
                key = ".".join([prefix, key])
            gs_params.append([(key, v) for v in value])
        elif isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_gs_parameters(value, prefix)
            if "." in prefix:
                prefix = prefix.rsplit(".", 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                gs_params.extend(param_pool)
        elif isinstance(value, Sequence) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_gs_parameters(v, prefix)
                    if "." in prefix:
                        prefix = prefix.rsplit(".", 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        gs_params.extend(param_pool)
    return gs_params


def replace_list_by_value_in_params(params: Dict, keys: List[str], value: Any) -> None:
    """
    Replace the GridSearchLists in the parameter dict by the split values.
    Changes params dict in-place
    ----------
    params Dict of params
    keys List of str, nested dictionary keys
    value Value expanded from GridSearchList

    Returns
    -------
    None
    """
    node = params
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return params
        else:
            if "#" in key:
                key, _id = key.split("#")
                if key not in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if key not in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def expand_params(
    params: Dict, adjust_run_name: bool = False, run_name_key: str = "run"
) -> Tuple[List[Dict], List[Optional[ParameterCombination]]]:
    param_pool = unpack_gs_parameters(params)

    if not param_pool:
        return [params], [None]

    parameter_combinations: List[ParameterCombination] = []
    cv_params = []
    for parameter_combination in itertools.product(*param_pool):
        sub_params = copy.deepcopy(params)
        if adjust_run_name:
            name = sub_params[run_name_key]
        for nested_parameter_name, value in parameter_combination:
            replace_list_by_value_in_params(sub_params, nested_parameter_name.split("."), value)
            if adjust_run_name:
                name += "_" + nested_parameter_name + "_" + str(value)
        if adjust_run_name:
            sub_params[run_name_key] = name.replace(".args.", "_")
        cv_params.append(sub_params)
        parameter_combinations.append(parameter_combination)
    return cv_params, parameter_combinations
