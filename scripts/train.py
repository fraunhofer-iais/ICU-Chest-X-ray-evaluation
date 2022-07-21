# coding: utf-8
import argparse
import copy
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import coloredlogs
import pytorch_lightning as pl
import torch
import wandb
from IPython.core import ultratb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from key2med import project_path
from key2med.data.loader import ADataLoader
from key2med.metrics.callbacks import CalculateMetrics
from key2med.models.model import BaseModel
from key2med.utils.helper import create_class_instance, create_instance, timestamp
from key2med.utils.yamls import (
    ParameterCombination,
    expand_params,
    load_params,
    write_params,
)

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        nargs="+",
        help="Paths to config.yaml files. "
        'Add multiple files for a gridseach by "-c /configs/path1.yaml /configs/path2.yaml"',
    )
    parser.add_argument("-nc", "--no-cuda", action="store_true", help="Disable GPUs on this run. Use for debugging.")
    parser.add_argument(
        "-debug",
        "--debug",
        action="store_true",
        help="Set up a debugging environment in the console as fallback on exceptions.",
    )
    args = parser.parse_args()

    for path in args.config:
        assert os.path.exists(path), f"Config file '{path}' not found!"
    logger.info(f"Loading params from {args.config}")
    gs_params, parameter_combinations, file_names = [], [], []
    for config_path in args.config:
        _gs_params, _parameter_combinations = expand_params(load_params(config_path), adjust_run_name=False)
        gs_params.extend(_gs_params)
        parameter_combinations.extend(_parameter_combinations)
        file_names.extend([config_path] * len(_gs_params))

    logger.info(f"Found {len(gs_params)} params for grid-search.")
    shared_data_loader: Optional[ADataLoader] = None
    if check_data_loaders(gs_params):
        logger.info(f"Initializing shared dataloader for grid-search.")
        shared_data_loader = init_dataloader(gs_params[0]["data_loader"])

    for index, (params, parameter_combination, file_name) in enumerate(
        zip(gs_params, parameter_combinations, file_names)
    ):
        params = copy.deepcopy(params)
        logger.info(get_log_string(index, len(gs_params), file_name, parameter_combination))
        train_params(params, shared_data_loader, args.no_cuda, args.debug)


def get_log_string(
    index: int,
    total: int,
    file_name: str,
    parameter_combination: Optional[ParameterCombination] = None,
) -> str:
    log_string = (
        f"\n"
        f"\t = = = = = = = = = = = = = = = = = = \n"
        f"\t RUNNING CONFIG NO. {index + 1} of {total} from {file_name}\n"
    )
    if parameter_combination is not None:
        log_string += "\t PARAMETERS:\n"
        for name, value in parameter_combination:
            log_string += f"\t\t {name}: {value}\n"
    return log_string


def check_data_loaders(gs_params: List[Dict]) -> bool:
    if len(gs_params) == 1:
        return True
    data_loader_config = gs_params[0]["data_loader"]
    for params in gs_params[1:]:
        if params["data_loader"] != data_loader_config:
            return False
    return True


def train_params(params, shared_data_loader: Optional[ADataLoader] = None, no_cuda: bool = False, debug: bool = False):
    if "seed" in params:
        torch.manual_seed(params["seed"])
    if "run" in params:
        run_name = params["run"]
    elif "name" in params:
        run_name = params["name"]
    else:
        run_name = "no_run_name"
    params["run_time"] = timestamp()
    checkpoint, checkpoint_config = init_checkpoint(params.get("load_checkpoint", None))
    if shared_data_loader is not None:
        data_loader = shared_data_loader
    else:
        data_loader = init_dataloader(params["data_loader"])

    if params["model"]["args"]["learning_rate_scheduler"]["name"] == "OneCycleLR":
        params["model"]["args"]["learning_rate_scheduler"]["args"]["steps_per_epoch"] = int(
            len(data_loader.train)
        )  # dirty fix OneCyleLR
    model = init_model(params["model"], data_loader, checkpoint, checkpoint_config)

    callbacks = init_callbacks(params, run_name, data_loader)

    if params["logging"].get("use_wandb", True):
        train_logger = WandbLogger(
            project=params.get("project", "key2med"),
            group=run_name,
            save_dir=params["logging"]["logging_dir"],
        )
        train_logger.experiment.config.update(params)
    else:
        train_logger = TensorBoardLogger(
            name=run_name,
            version=params["run_time"],
            save_dir=params["logging"]["logging_dir"],
        )

    trainer = pl.Trainer(
        logger=train_logger,
        callbacks=callbacks,
        gpus=1 if (torch.cuda.is_available() and not no_cuda) else 0,
        fast_dev_run=2 if debug else False,
        **params["trainer"]["args"],
    )
    trainer.fit(model, train_dataloaders=data_loader.train, val_dataloaders=data_loader.validate)
    hparams_path = os.path.join(train_logger.log_dir, "hparams.json")
    with open(hparams_path, "w") as fp:
        json.dump(params, fp)
    cleanup(trainer)


def cleanup(trainer: pl.Trainer) -> None:
    wandb.finish()


def init_callbacks(params, run_name, data_loader) -> List[pl.Callback]:
    callbacks = []

    calculate_metrics = CalculateMetrics(metrics=params["metrics"], index_to_label=data_loader.index_to_label)
    callbacks.append(calculate_metrics)

    save_checkpoints = params["model_checkpoints"].get("save_checkpoints", False)
    checkpoints_dir = os.path.join(params["model_checkpoints"]["checkpoints_dir"], run_name, params["run_time"])
    params["model_checkpoints"]["checkpoints_dir"] = checkpoints_dir
    if save_checkpoints:
        os.makedirs(checkpoints_dir, exist_ok=True)
        write_params(params, os.path.join(checkpoints_dir, "config.yaml"))
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor=params["model_checkpoints"].get("best_model_metric", "valid/AUROCMacro"),
        save_last=save_checkpoints,  # only save last if any checkpoints are saved
        verbose=True,
        mode="max",
        save_top_k=1 if save_checkpoints else 0,  # top_k = 0 -> no checkpointing,
    )
    callbacks.append(model_checkpoint)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    model_summary = ModelSummary(max_depth=3)
    callbacks.append(model_summary)

    # early_stopping = EarlyStopping(monitor="valid/loss", patience=2, verbose=True, mode="min", check_on_train_epoch_end=True)
    # callbacks.append(early_stopping)

    return callbacks


def init_dataloader(dataloader_params: Dict, rank: int = 0, world_size: int = 1) -> ADataLoader:
    for key, value in dataloader_params["args"].items():
        if "_path" in key or "_dir" in key:
            dataloader_params["args"][key] = os.path.join(project_path, value)
    dataloader_params = init_transform(dataloader_params)  # initialise transform
    logger.info(f"Initializing dataloader")
    data_loader = create_class_instance(
        dataloader_params["module"],
        dataloader_params["name"],
        dataloader_params["args"],
        rank=rank,
        world_size=world_size,
    )
    return data_loader


def init_transform(dataloader_params: Dict):
    if "transform" in dataloader_params["args"].keys():
        dataloader_params["args"]["transform"] = create_instance("transform", dataloader_params["args"])
    return dataloader_params


def init_checkpoint(checkpoint_config: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    if checkpoint_config is not None and checkpoint_config["path"] is not None:
        checkpoint: Optional[Dict] = torch.load(checkpoint_config["path"])
    else:
        checkpoint = None
    return checkpoint, checkpoint_config


def init_model(model_params: Dict, data_loader: ADataLoader, checkpoint, checkpoint_config) -> BaseModel:
    logger.info(f"Initializing model")
    model = create_class_instance(
        model_params["module"],
        model_params["name"],
        model_params["args"],
        image_dim=data_loader.image_dim,
        label_dim=data_loader.label_dim,
        channels=data_loader.channels,
    )
    if checkpoint is not None and checkpoint_config.get("load_model", False):
        logger.info(f"Loading model weights from checkpoint.")
        # assert checkpoint['model_type'] == type(model).__name__
        # state_dict = checkpoint['model_state']
        try:
            state_dict = checkpoint["state_dict"]
        except:
            state_dict = checkpoint["model_state"]
        ignore_keys = checkpoint_config.get("ignore_weights", [])
        strict_load = len(ignore_keys) == 0
        for key in ignore_keys:
            logger.info(f"Removing weight {key} from state_dict.")
            state_dict.pop(key)
        logger.info(f"Loading state dict into model.")
        model.load_state_dict(state_dict, strict=strict_load)

    return model


if __name__ == "__main__":
    main()
