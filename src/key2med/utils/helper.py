# coding: utf-8

import copy
import hashlib
import itertools
import json
import os
import shutil
from datetime import datetime as dt
from functools import reduce
from importlib import import_module
from logging import Logger
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import linalg


def timestamp(format: str = "%Y-%m-%d-%H-%M") -> str:
    return dt.now().strftime(format)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def bytes_to_gigabytes(x: int) -> float:
    return np.round(x / (1024**3), 2)


def get_file_size(file_path: str) -> float:
    assert os.path.isfile(file_path)
    return bytes_to_gigabytes(os.path.getsize(file_path))


def get_disk_usage(path: str) -> Tuple[str, float, float, float]:
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    total, used, free = shutil.disk_usage(path)
    return (
        path,
        bytes_to_gigabytes(total),
        bytes_to_gigabytes(used),
        bytes_to_gigabytes(free),
    )


def hash_string(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def hash_dict(x: Dict) -> str:
    return hash_string(json.dumps(x, sort_keys=True))


def create_class_instance(module_name, class_name, param_args: Dict = None, *args, **kwargs):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param param_args: arguments needed for the class constructor, as dict from param dict
    :param args: arguments needed for the class constructor, as additional positional arguments
    :param kwargs: arguments needed for the class constructor, as additional keyword arguments
    :returns: instance of 'class_name'

    """

    module = import_module(module_name)
    clazz = getattr(module, class_name)
    instance = clazz(*args, **(param_args or {}), **kwargs)

    return instance


def create_instance(name, param_args, *args, **kwargs):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instantiate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = param_args[name]
    if type(i_params) in [list, tuple]:
        instance = [create_class_instance(p["module"], p["name"], p.get("args", {}), *args, **kwargs) for p in i_params]
    else:
        instance = create_class_instance(
            i_params["module"],
            i_params["name"],
            i_params.get("args", {}),
            *args,
            **kwargs,
        )
    return instance


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    return clazz


def get_device(params: dict, rank: int = 0, logger: Logger = None, no_cuda: bool = False) -> torch.device:
    """

    :param params:
    :param logger:
    :return: returns the device
    """
    gpus = params.get("gpus", [])
    if not no_cuda and len(gpus) > 0:
        if not torch.cuda.is_available():
            if logger is not None:
                logger.warning("No GPU's available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(gpus[rank]))
    else:
        device = torch.device("cpu")
    return device
