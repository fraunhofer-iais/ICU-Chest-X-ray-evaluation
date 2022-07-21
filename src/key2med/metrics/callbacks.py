import logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import humanize
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from key2med.metrics.binary_metrics import BinaryMetric
from key2med.metrics.multilabel_metrics import MultiLabelMetric
from key2med.utils.helper import create_class_instance
from key2med.utils.logging import log_text

logger = logging.getLogger(__name__)

Metric = Union[BinaryMetric, MultiLabelMetric]


class CalculateMetrics(pl.Callback):
    def __init__(
        self,
        metrics: List[Dict],
        index_to_label: List[str] = None,
        only_evaluate_on: List[int] = None,
        exception_on_none_value: bool = False,
    ):
        self.metrics_dict = metrics
        _metrics = [create_class_instance(p["module"], p["name"], p.get("args", {})) for p in self.metrics_dict]
        self.metrics: List[Metric] = _metrics

        self.index_to_label = index_to_label
        self.only_evaluate_on = only_evaluate_on

        self.exception_on_none_value = exception_on_none_value

        self.y_target: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None

        self.train_targets: List[np.ndarray] = []
        self.train_preds: List[np.ndarray] = []
        self.valid_targets: List[np.ndarray] = []
        self.valid_preds: List[np.ndarray] = []
        self.test_targets: List[np.ndarray] = []
        self.test_preds: List[np.ndarray] = []

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self.train_targets.append(outputs["target"])
        self.train_preds.append(outputs["pred"])

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[pl.utilities.types.STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.valid_targets.append(outputs["target"])
        self.valid_preds.append(outputs["pred"])

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.test_targets.append(outputs["target"])
        self.test_preds.append(outputs["pred"])

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_metrics(
            targets=self.train_targets,
            preds=self.train_preds,
            logging_prefix="train",
            train=True,
            trainer=trainer,
            pl_module=pl_module,
        )
        self.train_targets = []
        self.train_preds = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_metrics(
            targets=self.valid_targets,
            preds=self.valid_preds,
            logging_prefix="valid",
            train=False,
            trainer=trainer,
            pl_module=pl_module,
        )
        self.valid_targets = []
        self.valid_preds = []

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_metrics(
            targets=self.test_targets,
            preds=self.test_preds,
            logging_prefix="test",
            train=False,
            trainer=trainer,
            pl_module=pl_module,
        )
        self.test_targets = []
        self.test_preds = []

    def log_metrics(
        self,
        targets: List[np.ndarray],
        preds: List[np.ndarray],
        train: bool,
        logging_prefix: str,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)

        batch_size, num_labels = preds.shape

        if self.index_to_label is None:
            index_to_label = [str(label).zfill(2) for label in range(num_labels)]
        else:
            assert len(self.index_to_label) == num_labels
            index_to_label = self.index_to_label

        if self.only_evaluate_on is not None:
            keep_labels = self.only_evaluate_on
            keep_label_names = [index_to_label[i] for i in keep_labels]
        else:
            keep_labels = slice(None)
            keep_label_names = index_to_label

        for metric in self.metrics:
            metric_name = type(metric).__name__
            _targets = targets.copy()
            _preds = preds.copy()
            if isinstance(metric, MultiLabelMetric):
                value = metric(
                    _preds[:, keep_labels],
                    _targets[:, keep_labels],
                    label_names=keep_label_names,
                    train=train,
                )
                if value is None and not self.exception_on_none_value:
                    continue
                if isinstance(value, str):
                    log_text(f"{logging_prefix}/{metric_name}", value, trainer)
                else:
                    pl_module.log(f"{logging_prefix}/{metric_name}", value)

            elif isinstance(metric, BinaryMetric):
                for label, label_name in enumerate(index_to_label):
                    if self.only_evaluate_on is not None and label not in self.only_evaluate_on:
                        continue
                    value = metric(_preds[:, label], _targets[:, label], train=train)
                    if value is None and not self.exception_on_none_value:
                        continue
                    if isinstance(value, str):
                        log_text(
                            f"{logging_prefix}/{metric_name}/{label_name}",
                            value,
                            trainer,
                        )
                    else:
                        pl_module.log(f"{logging_prefix}/{metric_name}/{label_name}", value)
