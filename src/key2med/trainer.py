import datetime
import json
import logging
import math
import os
import pickle
import shutil
import time
from abc import ABCMeta
from collections import ChainMap
from typing import Any, Dict, List, Tuple

import coloredlogs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from libauc.optimizers import PESG
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import wandb
from key2med.data.loader import ADataLoader
from key2med.metrics.evaluation import Evaluator
from key2med.utils.helper import (
    create_instance,
    get_device,
    is_array,
    is_primitive,
    optimizer_to,
)
from key2med.utils.logging import tqdm
from key2med.utils.plotting import image_to_tensor, plot_image_grid, tensor_to_image

coloredlogs.install(level=logging.INFO)
logging.basicConfig()


class MyDistributedDataParallel(DDP):
    def train_step(self, *args, **kwargs):
        return self.module.train_step(*args, **kwargs)

    def validate_step(self, *args, **kwargs):
        return self.module.validate_step(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.module.generate(*args, **kwargs)

    def reconstruct(self, *args, **kwargs):
        return self.module.reconstruct(*args, **kwargs)


class BaseTrainingProcedure(metaclass=ABCMeta):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: dict,
        resume: str,
        params: dict,
        data_loader: ADataLoader,
        evaluator: Evaluator,
        no_cuda: bool = False,
        delete_failed_runs: bool = False,
        **kwargs,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_loader: ADataLoader = data_loader
        self.distributed: bool = len(params.get("gpus", [])) > 1
        self.optimizer: dict = optimizer
        self.params: dict = params

        self.evaluator = evaluator
        if self.data_loader.index_to_label is not None:
            self.evaluator.index_to_label = self.data_loader.index_to_label

        self.rank = 0
        self.world_size = 1
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = get_device(params, self.rank, self.logger, no_cuda)
            self.model = model.to(self.device)
            self.model = MyDistributedDataParallel(self.model, device_ids=[self.rank])
            self.optimizer_to_device()
        else:
            self.device = get_device(params, self.rank, self.logger, no_cuda)
            self.model = model.to(self.device)
            self.optimizer_to_device()
        self.delete_failed_runs = delete_failed_runs

        self.run_group: str = self.params["name"]
        if len(self.run_group) > 200:
            self.run_group = "_".join([i if i.isdigit() else i[0:3] for i in self.run_group.split("_")])

        self.is_rank_0 = (not self.distributed) or (self.distributed and self.rank == 0)
        self.logger.info(str(self.is_rank_0))
        time.sleep(10)
        if self.is_rank_0:
            self.run_name: str = datetime.datetime.now().strftime("%m%d_%H%M%S")
            self._prepare_dirs()
            self._save_params()
            self._setup_tensorboard()
            self.train_logger = self._setup_train_logger()
        else:
            self.run_name = None

        self.wandb: bool = self.params["trainer"]["args"].get("wandb", False)
        if self.wandb and self.is_rank_0:
            self._setup_wandb()

        self.start_epoch: int = 0

        trainer_args = self.params["trainer"]["args"]
        self.n_epochs: int = trainer_args["epochs"]
        self.save_after_epoch: int = trainer_args["save_after_epoch"]
        self.bm_metric: str = trainer_args["bm_metric"]

        self.lr_schedulers = self.__init_lr_schedulers()

        if "schedulers" in trainer_args:
            self.schedulers = dict()
            schedulers_ = create_instance("schedulers", trainer_args)
            if type(schedulers_) is not list:
                schedulers_ = [schedulers_]
            for a, b in zip(trainer_args["schedulers"], schedulers_):
                self.schedulers[a["label"]] = b
        else:
            self.schedulers = None

        self.data_loader: ADataLoader = data_loader
        self.n_train_batches: int = data_loader.n_train_batches
        self.n_validate_batches: int = data_loader.n_validate_batches
        self.n_test_batches: int = data_loader.n_test_batches
        self.do_train = self.n_train_batches > 0
        self.do_validate = self.n_validate_batches > 0
        self.do_test = self.n_test_batches > 0

        self.n_generation: int = trainer_args.get("n_generation", 20)
        self.n_reconstruction: int = trainer_args.get("n_reconstruction", 20)
        self.do_generate = self.n_generation > 0
        self.do_reconstruct = self.n_reconstruction > 0

        self.global_step: int = 0
        self.epoch: int = 0
        self.best_model = {"train_loss": float("inf"), "val_loss": float("inf"), "train_metric": 0.5, "val_metric": 0.5}

        if resume:
            self._resume_check_point(resume)

    def train(self) -> Dict:
        if self.delete_failed_runs:
            return self._try_train_except()
        else:
            return self._train()

    def validate(self) -> Dict:
        return self._validate()

    def _try_train_except(self) -> Dict:
        try:
            return self._train()

        except KeyboardInterrupt:
            try:
                self.logger.warning(
                    "Run aborted by Ctrl+C. Press Ctrl+C again within 10 seconds " "to remove the run directories."
                )
                time.sleep(10)
            except KeyboardInterrupt:
                self._remove_dirs()
            raise

        except Exception as e:
            try:
                self.logger.warning(f"Raised exception {e}")
                self.logger.warning("Press Ctrl+C within 10 seconds to remove " "the run directories.")
                time.sleep(10)
            except KeyboardInterrupt:
                self._remove_dirs()
            raise

    def _train(self) -> Dict:
        e_bar = tqdm(
            desc=f"Rank {self.rank}, Epoch: ",
            total=self.n_epochs,
            unit="epoch",
            initial=self.start_epoch,
            position=self.rank * 2,
            ascii=True,
            leave=True,
        )
        for epoch in range(self.start_epoch, self.n_epochs):
            train_log, validate_log, test_log = None, None, None
            if self.do_train:
                if self.epoch > 1 and isinstance(self.optimizer["loss_optimizer"]["opt"], PESG):
                    self.optimizer["loss_optimizer"]["opt"].update_regularizer(decay_factor=10)
                train_log = self._train_epoch()
            if self.do_validate:
                validate_log = self._validate_epoch()
            if self.do_test:
                test_log = self._test_epoch()

            if self.do_generate:
                self.plot_generation(self.n_generation, epoch)
            if self.do_reconstruct:
                self.plot_test_reconstruction(self.n_reconstruction, epoch)

            self._anneal_lr(validate_log)

            self._update_p_bar(e_bar, train_log, validate_log, test_log)
            self._save_model_epoch(epoch, train_log, validate_log)
            self.epoch += 1
            if validate_log is not None and self._check_early_stopping(validate_log):
                break

        self._clear_logging_resources(e_bar)
        return self.best_model

    def _validate(self) -> Dict:
        e_bar = tqdm(
            desc=f"Rank {self.rank}, Epoch: ",
            total=self.n_epochs,
            unit="epoch",
            initial=self.start_epoch,
            position=self.rank * 2,
            ascii=True,
            leave=True,
        )

        for epoch in range(self.start_epoch, self.n_epochs):
            self.epoch += 1
            validate_log, test_log = None, None
            if self.do_validate:
                validate_log = self._validate_epoch()
            if self.do_test:
                test_log = self._test_epoch()

            if self.do_generate:
                self.plot_generation(self.n_generation, epoch)
            if self.do_reconstruct:
                self.plot_test_reconstruction(self.n_reconstruction, epoch)

            self._update_p_bar(e_bar, validate_log=validate_log, test_log=test_log)
            self._save_model_epoch(epoch, validate_log=validate_log)

        self._clear_logging_resources(e_bar)
        return self.best_model

    def optimizer_to_device(self):
        """
        Loading a state dict might move the optimizer back to cpu. Make sure all optimizers are on GPU!
        :return: None
        """
        for key in self.optimizer:
            optimizer_to(self.optimizer[key]["opt"], self.device)

    def _setup_wandb(self):
        try:
            import wandb

            wandb.login()
            wandb.init(
                project="key2med",
                group=self.run_group,
                config={
                    **self.params,
                    **{
                        "run_name": self.run_name,
                        "checkpoint_dir": self.checkpoint_dir,
                        "logging_dir": self.logging_dir,
                        "tensorboard_dir": self.tensorboard_dir,
                    },
                },
            )
        except Exception:
            self.logger.error(
                "Could not log into wandb or initialize run. To use wandb, set up once on this machine:\n"
                '"!pip install wandb"\n'
                '"import wandb\n'
                '"wandb.login()"'
            )
            raise

    def _setup_tensorboard(self):
        self.tensorboard = SummaryWriter(self.tensorboard_dir)
        self.logger.info(
            f"Set up tensorboard in {self.tensorboard.log_dir}\n"
            f"Event file name: {self.tensorboard.file_writer.event_writer._file_name}"
        )

    def __init_lr_schedulers(self):
        lr_schedulers = self.params["trainer"]["args"].get("lr_schedulers", None)
        trainer_args = self.params["trainer"]["args"]
        if lr_schedulers is None:
            return None
        schedulers = dict()
        for idx, scheduler in enumerate(lr_schedulers):
            opt_name = scheduler["args"]["optimizer"]
            trainer_args["lr_schedulers"][idx]["args"]["optimizer"] = self.optimizer[opt_name]["opt"]
        lr_schedulers_ = create_instance("lr_schedulers", trainer_args)
        for a, b in zip(trainer_args["lr_schedulers"], lr_schedulers_):
            schedulers[a["label"]] = b
        return schedulers

    def _clear_logging_resources(self, e_bar: tqdm) -> None:
        if self.is_rank_0:
            self.tensorboard.flush()
            self.tensorboard.close()
        if self.wandb and self.is_rank_0:
            wandb.finish()
        e_bar.close()

    def _save_model_epoch(self, epoch: int, train_log: dict = None, validate_log: dict = None) -> None:
        if not self.is_rank_0:
            return
        self._check_and_save_best_model(train_log, validate_log)
        if epoch == 0 or self.save_after_epoch is None or self.save_after_epoch == -1:
            return
        if epoch % self.save_after_epoch == 0:
            self._save_check_point(epoch)

    def _anneal_lr(self, validate_log: dict) -> None:
        if self.lr_schedulers is not None:
            for key in self.lr_schedulers.keys():
                self.lr_schedulers[key].step()

    def _check_early_stopping(self, log: dict) -> bool:
        cond = [x for x in self.optimizer.values() if x["opt"].param_groups[0]["lr"] < float(x.get("min_lr_rate", 0.0))]
        if len(cond) != 0:
            self.logger.warning("Stopping early: LR condition met!")
            return True
        loss = log["loss"]
        if np.isinf(loss):
            self.logger.warning("Stopping early: Loss is inf!")
            return True
        if np.isnan(loss):
            self.logger.warning("Stopping early: Loss is nan!")
            return True
        return False

    def _train_epoch(self) -> dict:
        self.model.train()
        with tqdm(
            desc=f"Rank {self.rank}, Training batch: ",
            total=self.n_train_batches,
            unit="batch",
            leave=False,
            ascii=True,
            position=self.rank * 2 + 1,
        ) as p_bar:
            self.evaluator.reset_y()
            for batch_idx, data in enumerate(self.data_loader.train):
                batch_stats = self._train_step(data, batch_idx, p_bar)
                self.evaluator.add_batch(batch_stats)

        epoch_stats = self.evaluator.evaluate(train=True)
        self._log_epoch("train/epoch/", epoch_stats)
        return epoch_stats

    def _train_step(self, minibatch: Any, batch_idx: int, p_bar: tqdm) -> dict:
        stats = self.model.train_step(minibatch, self.optimizer, self.global_step, scheduler=self.schedulers)
        self._update_step_p_bar(p_bar, stats)
        # stats = self._recv_stats_across_nodes(stats)

        self._log_step("train", batch_idx, self.data_loader.train_set_size, stats)
        self.global_step += 1

        return stats

    def _validate_epoch(self) -> dict:
        if not self.is_rank_0:
            return None
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                desc=f"Rank {self.rank}, Validation batch: ",
                total=self.n_validate_batches,
                unit="batch",
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1,
            ) as p_bar:
                self.evaluator.reset_y()
                for batch_idx, data in enumerate(self.data_loader.validate):
                    batch_stats = self._validate_step(data, batch_idx, p_bar)
                    self.evaluator.add_batch(batch_stats)

            epoch_stats = self.evaluator.evaluate(train=False)
            self._log_epoch("validate/epoch/", epoch_stats)

            return epoch_stats

    def _validate_step(self, minibatch: dict, batch_idx: int, p_bar: tqdm) -> dict:
        stats = self.model.validate_step(minibatch, scheduler=self.schedulers)
        self._update_step_p_bar(p_bar, stats)

        self._log_step("train", batch_idx, self.data_loader.valid_set_size, stats)
        return stats

    def _test_epoch(self) -> dict:
        if not self.is_rank_0:
            return None
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                desc=f"Rank {self.rank}, Test batch: ",
                total=self.n_test_batches,
                unit="batch",
                ascii=True,
                position=self.rank * 2 + 1,
                leave=False,
            ) as p_bar:
                self.evaluator.reset_y()
                for batch_idx, data in enumerate(self.data_loader.test):
                    batch_stats = self._test_step(data, batch_idx, p_bar)
                    self.evaluator.add_batch(batch_stats)

            epoch_stats = self.evaluator.evaluate(train=False)
            self._log_epoch("test/epoch/", epoch_stats)

        return epoch_stats

    def generate_data(self, n) -> torch.tensor:
        return self.model.generate(n)

    def reconstruct_batch(self, data_loader, n: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            data = next(iter(data_loader))
            if n is not None:
                data = data[:n]
            # Here we overwrite data in case we have labels.
            # The model.reconstruct implements returning only the original data, not the label
            data, data_recon = self.model.reconstruct(data)
        return data, data_recon

    def plot_test_reconstruction(self, n: int, epoch: int) -> plt.Figure:
        try:
            reconstruct_loader = self.data_loader.test or self.data_loader.validate or self.data_loader.train
            data, data_recon = self.reconstruct_batch(reconstruct_loader, n)
        except NotImplementedError:  # self.model.reconstruct is not implemented
            return None
        grid_tensor = plot_image_grid(data, data_recon, as_image=False)
        if self.tensorboard is not None:
            self.tensorboard.add_image("generation", grid_tensor, epoch)
        if self.wandb:
            wandb.log({"reconstruction": wandb.Image(grid_tensor)})

    def plot_generation(self, n: int, epoch: int) -> plt.Figure:
        try:
            reconstruct_loader = self.data_loader.test or self.data_loader.validate or self.data_loader.train
            data, data_recon = self.reconstruct_batch(reconstruct_loader, n)
        except NotImplementedError:  # self.model.reconstruct is not implemented
            return None
        grid_tensor = plot_image_grid(data, as_image=False)
        if self.tensorboard is not None:
            self.tensorboard.add_image("generation", grid_tensor, epoch)
        if self.wandb:
            wandb.log({"generation": wandb.Image(grid_tensor)})

    def _test_step(self, minibatch: dict, batch_idx: int, p_bar: tqdm) -> dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)

        self._log_step("test", batch_idx, self.data_loader.test_set_size, stats)
        return stats

    def _log_epoch(self, log_label: str, statistics: dict) -> None:
        if not self.is_rank_0:
            return None

        self.logger.info(f"\nMetrics {log_label}{self.epoch}: \n")
        for k, v in statistics.items():
            if v is None:
                continue
            try:
                v = float(v.item())
            except:
                pass
            if is_primitive(v):
                self.logger.info(f"{log_label}{self.epoch} {k}: {v}")
                self.tensorboard.add_scalar(log_label + k, v, self.global_step)
                if self.wandb:
                    wandb.log({log_label + k: v})
            elif isinstance(v, str):
                self.logger.info(f"{log_label}{self.epoch} {k}:\n{v}")
                self.tensorboard.add_text(log_label + k, v, self.global_step)
                if self.wandb:
                    wandb.log({log_label + k: v})
            elif isinstance(v, list) and isinstance(v[0], int):
                self.tensorboard.add_histogram(log_label + k, v, self.global_step)
            elif isinstance(v, matplotlib.figure.Figure):
                self.tensorboard.add_figure(log_label + k, figure=v, global_step=self.global_step)

    def _prepare_dirs(self) -> None:
        trainer_par = self.params["trainer"]
        self.checkpoint_dir = os.path.join(trainer_par["save_dir"], self.run_group, self.run_name)
        self.logging_dir = os.path.join(trainer_par["logging"]["logging_dir"], "raw", self.run_group, self.run_name)
        self.tensorboard_dir = os.path.join(
            trainer_par["logging"]["logging_dir"], "tensorboard", self.run_group, self.run_name
        )

        self.logger.info(
            f"Saving raw logs in {self.logging_dir}\n"
            f"Saving model checkpoints in {self.checkpoint_dir}\n"
            f"Saving tensorboard files in {self.tensorboard_dir}."
        )
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def _remove_dirs(self) -> None:
        """
        Function to call when a run is aborted or crashes during debugging.
        Deletes all logging and saving dirs for this run!
        """
        self.logger.warning(f"DELETING THE RUN DIRECTORIES!")
        for dir in [self.logging_dir, self.checkpoint_dir, self.tensorboard_dir]:
            self.logger.warning(f"DELETING DIRECTORY {dir} AND ALL CONTENTS!")
            shutil.rmtree(dir)
        if self.wandb and self.is_rank_0:
            self._remove_wandb_run()

    def _remove_wandb_run(self) -> None:
        """
        Function to call when a run is aborted or crashes during debugging.
        Deletes all logging and saving dirs for this run!
        """
        wandb.finish()
        api = wandb.Api()
        run = api.run(wandb.run.path)
        run.delete()

    def _save_params(self):
        params_path = os.path.join(self.logging_dir, "config.yaml")
        self.logger.info(f"Saving config into {params_path}")
        yaml.dump(self.params, open(params_path, "w"), default_flow_style=False)

    def _save_model(self, file_name: str) -> None:
        model_type = type(self.model).__name__
        state = {
            "model_type": model_type,
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "params": self.params,
        }
        for key in self.optimizer:
            state[key] = self.optimizer[key]["opt"].state_dict()

        torch.save(state, file_name)

    def _save_model_parameters(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.params, f, indent=4)

    def _save_check_point(self, epoch: int) -> None:
        file_name = os.path.join(self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch))
        self.logger.info("Saving checkpoint: {} ...".format(file_name))
        self._save_model(file_name)

    def _save_best_model(self) -> None:
        print("*******")
        file_name = os.path.join(self.checkpoint_dir, "best_model.pth")
        self._save_model(file_name)

    def _resume_check_point(self, path: str) -> None:
        self.logger.info("Loading checkpoint: {} ...".format(path))
        if torch.cuda.is_available() is False:
            state = torch.load(path, map_location="cpu")
        else:
            state = torch.load(path)
        self.params = state["params"]
        if state["epoch"] is None:
            self.start_epoch = 1
        else:
            self.start_epoch = state["epoch"] + 1
        self.model.load_state_dict(state["model_state"])
        for key in self.optimizer:
            self.optimizer[key].load_state_dict(state[key])
        self.logger.info("Finished loading checkpoint: {} ...".format(path))

    def _setup_train_logger(self) -> logging.Logger:
        logger = logging.getLogger("train_logger")
        logger.propagate = False  # Only write logging to file, not to console
        logger.setLevel(logging.INFO)
        file_name = os.path.join(self.logging_dir, "train.log")
        fh = logging.FileHandler(file_name)
        formatter = logging.Formatter(self.params["trainer"]["logging"]["formatters"]["simple"])
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def _log_step(self, step_type: str, batch_idx: int, data_len: int, statistics: dict) -> None:
        if not self.is_rank_0:
            return None
        log = self._build_raw_log_str(
            f"{step_type} epoch", batch_idx, statistics, float(data_len), self.data_loader.batch_size
        )
        self.train_logger.info(log)
        for k, v in statistics.items():
            if v is None:
                continue
            try:
                v = float(v.item())
            except:
                pass
            if is_primitive(v):
                self.tensorboard.add_scalar(f"{step_type}/batch/" + k, v, self.global_step)
                if self.wandb:
                    wandb.log({f"{step_type}/batch/" + k: v})

    def _build_raw_log_str(self, prefix: str, batch_idx: int, logs: dict, data_len: float, batch_size: int):
        sb = prefix + ": {} [{}/{} ({:.0%})]".format(
            self.epoch, batch_idx * batch_size, data_len, 100.0 * batch_idx / data_len
        )
        for k, v in logs.items():
            if is_primitive(v):
                sb += " {}: {:.6f}".format(k, v)
        return sb

    def _check_and_save_best_model(self, train_log: dict, validate_log: dict) -> None:
        if validate_log is not None and self.bm_metric in validate_log:
            if validate_log[self.bm_metric] is None:
                return
            if np.abs(validate_log[self.bm_metric] - 0.5) < np.abs(self.best_model["val_metric"] - 0.5):
                return
            self.logger.info(
                f"New best validation metric {self.bm_metric}: {validate_log[self.bm_metric]}. Saving model."
            )
            self._save_best_model()
            self._update_best_model_flag(train_log, validate_log)
        elif train_log is not None and self.bm_metric in train_log:
            if np.abs(train_log[self.bm_metric] - 0.5) < np.abs(self.best_model["val_metric"] - 0.5):
                return
            self.logger.info(f"New best training metric {self.bm_metric}: {train_log[self.bm_metric]}. Saving model.")
            self._save_best_model()
            self._update_best_model_flag(train_log, validate_log)
        else:
            return

    def _update_p_bar(
        self, e_bar: tqdm, train_log: dict = None, validate_log: dict = None, test_log: dict = None
    ) -> None:
        postfix_str = ""
        if train_log is not None:
            postfix_str += f"train loss: {train_log['loss']:4.4g} "
            if train_log.get(self.bm_metric, None) is not None:
                postfix_str += f"train {self.bm_metric}: {train_log[self.bm_metric]:4.4g}, "
            else:
                postfix_str += f"train {self.bm_metric}: None, "
        if validate_log is not None:
            postfix_str += f"validation loss: {validate_log['loss']:4.4g}, "
            if validate_log.get(self.bm_metric, None) is not None:
                postfix_str += f"validation {self.bm_metric}: {validate_log[self.bm_metric]:4.4g}, "
            else:
                postfix_str += f"validation {self.bm_metric}: None, "
        if test_log is not None:
            postfix_str += f"test loss: {test_log['loss']:4.4g}, "
            if test_log.get(self.bm_metric, None) is not None:
                postfix_str += f"test {self.bm_metric}: {test_log[self.bm_metric]:4.4g}, "
            else:
                postfix_str += f"test {self.bm_metric}: None, "
        e_bar.set_postfix_str(postfix_str)
        e_bar.update()

    @staticmethod
    def _update_step_p_bar(p_bar: tqdm, stats: dict):
        log_str = ""
        for key, value in stats.items():
            if not (isinstance(value, float) or (is_array(value) and value.shape == (1,))):
                continue
            log_str += f"{key}: {float(value):4.4g} "

        p_bar.update()
        p_bar.set_postfix_str(log_str)

    def _update_best_model_flag(self, train_log: dict, validate_log: dict) -> None:
        if train_log is not None:
            self.best_model["train_loss"] = train_log["loss"]
            self.best_model["train_metric"] = train_log[self.bm_metric]
        if validate_log is not None:
            self.best_model["val_loss"] = validate_log["loss"]
            self.best_model["val_metric"] = validate_log[self.bm_metric]
        self.best_model["name"] = self.params["name"]

    @staticmethod
    def tensor_2_item(stats):
        for key, value in stats.items():
            if type(value) is torch.Tensor and len(value.size()) == 0:
                stats[key] = value.item()
        return stats


class ContinuousTrainer(BaseTrainingProcedure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_every = kwargs.get("evaluate_every", 400)
        self.total_steps = 0
        self.evaluator_validation = self.evaluator.__copy__()

    def _train_epoch(self) -> Dict:
        self.model.train()
        with tqdm(
            desc=f"Rank {self.rank}, Training batch: ",
            total=self.n_train_batches,
            unit="batch",
            leave=False,
            ascii=True,
            position=self.rank * 2 + 1,
        ) as p_bar:
            self.evaluator.reset_y()
            for batch_idx, data in enumerate(self.data_loader.train):
                batch_stats = self._train_step(data, batch_idx, p_bar)
                self.evaluator.add_batch(batch_stats)
                self.total_steps += 1
                if self.total_steps % self.evaluate_every == 0 and self.do_validate:
                    validate_log = self._validate_epoch()
                    self._check_and_save_best_model(None, validate_log)
                    self.model.train()

        epoch_stats = self.evaluator.evaluate(train=True)
        self._log_epoch("train/epoch/", epoch_stats)
        return epoch_stats

    def _validate_epoch(self) -> dict:
        if not self.is_rank_0:
            return None
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                desc=f"Rank {self.rank}, Validation batch: ",
                total=self.n_validate_batches,
                unit="batch",
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1,
            ) as p_bar:
                self.evaluator_validation.reset_y()
                for batch_idx, data in enumerate(self.data_loader.validate):
                    batch_stats = self._validate_step(data, batch_idx, p_bar)
                    self.evaluator_validation.add_batch(batch_stats)

            epoch_stats = self.evaluator_validation.evaluate(train=True)
            self._log_epoch("validate/epoch/", epoch_stats)
            return epoch_stats
