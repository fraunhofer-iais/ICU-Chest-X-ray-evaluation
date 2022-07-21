from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from torch.nn.modules.loss import _Loss as TorchLoss

from key2med.metrics.binary_metrics import BinaryMetric
from key2med.metrics.multilabel_metrics import MultiLabelMetric
from key2med.utils.helper import create_class_instance

Metric = Union[MultiLabelMetric, BinaryMetric, TorchLoss]


class Evaluator:
    def __init__(self, metrics: List[Dict], index_to_label: List[str] = None, only_evaluate_on: List[int] = None):
        self.metrics_dict = metrics
        _metrics = [create_class_instance(p["module"], p["name"], p.get("args", {})) for p in self.metrics_dict]
        self.metrics = _metrics

        self.y_target: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.losses: Dict = defaultdict(list)

        self.index_to_label = index_to_label
        self.only_evaluate_on = only_evaluate_on

    def add_batch(self, batch_stats: Dict):
        y_target = batch_stats["y_target"]
        y_pred = batch_stats["y_pred"]
        if self.y_target is None:
            self.y_target = y_target.copy()
        else:
            self.y_target = np.concatenate([self.y_target, y_target], axis=0)
        if self.y_pred is None:
            self.y_pred = y_pred.copy()
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0)

        for key, value in batch_stats.items():
            if key in ["y_target", "y_pred"]:
                continue
            self.losses[key].append(value)

    def __copy__(self):
        return Evaluator(self.metrics_dict, self.index_to_label, self.only_evaluate_on)

    def reset_y(self):
        self.y_target = None
        self.y_pred = None

    def reset_losses(self):
        self.losses: Dict = defaultdict(list)

    def new_stats(self) -> Dict[str, Union[float, str]]:
        return {}

    def evaluate(self, train: bool = True):
        stats = self.new_stats()

        num_labels = self.y_pred.shape[1]
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

        for m in self.metrics:
            if isinstance(m, MultiLabelMetric):
                y_pred_copy = self.y_pred.copy()
                y_target_copy = self.y_target.copy()
                stats[type(m).__name__] = m(
                    y_pred_copy[:, keep_labels],
                    y_target_copy[:, keep_labels],
                    label_names=keep_label_names,
                    train=train,
                )
            elif isinstance(m, BinaryMetric):
                for label, label_name in enumerate(index_to_label):
                    if self.only_evaluate_on is not None and label not in self.only_evaluate_on:
                        continue
                    y_pred_copy = self.y_pred.copy()
                    y_target_copy = self.y_target.copy()
                    stats[f"{type(m).__name__}_{label_name}"] = m(
                        y_pred_copy[:, label], y_target_copy[:, label], train=train
                    )

            else:
                pass
        self.reset_y()

        for key, value in self.losses.items():
            stats[key] = np.mean(value)
        self.reset_losses()
        return stats
