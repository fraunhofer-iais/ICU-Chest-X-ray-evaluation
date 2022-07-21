import warnings
from typing import Any, List, Optional

import numpy
import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.metrics._classification import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

"""
Functions/Classes for binary classification,
i.e. one output is probability in [0, 1] for a single class association.

Example:
y      = 1
y_pred = 0.9

Each metric must be a class,
which will be instantiated and then called with __call__

E.g.
 class Accuracy:

    def __call__(y_pred, y_target):
        return (y_pred == y_target).sum() / y_pred.nelement()

For batched inputs, y_pred.shape == [batch_size]
Metrics output a single value for a batch of inputs.
"""


class BinaryMetric:
    mode: str = "train"

    def __init__(self, evaluate_on: List[str] = ("train", "eval")):
        self.evaluate_on: List[str] = evaluate_on

    def __call__(self, y_pred, y_target, train: bool = True, **kwargs) -> Optional[float]:
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        if not isinstance(y_target, np.ndarray):
            y_pred = np.array(y_target)  # why set y_target as y_pred
        assert y_pred.shape == y_target.shape
        assert len(y_pred.shape) == 1  # [batch_size]

        self.mode = "train" if train else "eval"
        if self.mode not in self.evaluate_on:
            return None

        if self.mode == "train":
            y_target[0.3 >= abs(y_target - 0.55)] = 2
            if str(self.__class__.__name__) == "F1Score":
                y_pred[y_pred <= 0.5] = 0.0
                y_pred[y_pred > 0.5] = 1.0
                y_pred[y_target == 2] = 2
        else:
            if str(self.__class__.__name__) == "F1Score":
                y_pred[y_pred < 0.5] = 0.0
                y_pred[y_pred >= 0.5] = 1.0

        return self._metric(y_pred, y_target)

    def _metric(self, y_pred, y_target) -> Any:
        raise NotImplementedError


class BinaryCrossEntropy(BinaryMetric):
    def cross_entropy(self, predictions, targets):
        N = predictions.shape[0]
        # ce = -np.sum(targets * np.log(predictions)) / N
        ce = np.sum(targets * np.log(predictions) + (1 - targets) * np.log(predictions)) / N
        return ce

    def _metric(self, y_pred, y_target, **kwargs):
        return self.cross_entropy(y_pred, y_target)


class ZeroOneLoss(BinaryMetric):
    """
    y_target = [0, 1, 0, 1]
    y_pred = [1, 1, 0, 0]
    ZeroOneLoss = 0.5

    y_target = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    ZeroOneLoss = 0.0
    """

    def _metric(self, y_pred, y_target):
        return zero_one_loss(y_target, y_pred)


class PredictionPositiveRate(BinaryMetric):
    def _metric(self, y_pred, y_target):
        return y_pred.sum() / y_pred.size


class TargetPositiveRate(BinaryMetric):
    def _metric(self, y_pred, y_target):
        return y_target.sum() / y_target.size


class PositiveRateDiff(BinaryMetric):
    def _metric(self, y_pred, y_target):
        return y_target.sum() / y_target.size - y_pred.sum() / y_pred.size


class F1Score(BinaryMetric):
    def _metric(self, y_pred, y_target) -> float:
        score = f1_score(y_target, y_pred, average="binary" if self.mode == "eval" else "micro")
        return score


class AUROC(BinaryMetric):
    def _metric(self, y_pred, y_target, **kwargs):
        """
        Calculate AUROC score for each class separately, then average over all classes.
        If train mode, uncertain values are mapped onto a '2' class. Here we ignore these datapoints.
        If a batch for one class is only one label (e.g. all zeros) or the batch is
        empty (all uncertain during train mode) the AUROC is not defined and
        roc_auc_score raises a ValueError. Then the class is ignored for the average
        calculation.
        Return None if all classes are ignored for the average calcualtion.

        :param y_target: np.ndarray shape [batch_size, n_classes]
        :param y_pred:   np.ndarray shape [batch_size, n_classes]
        :return: value float or None
        :raises: None
        """
        try:
            keep_indices = slice(None)
            if self.mode == "train":
                keep_indices = np.where(y_target != 2.0)[0]
            y_target_keep = y_target[keep_indices]
            y_pred_keep = y_pred[keep_indices]
            score = roc_auc_score(y_target_keep, y_pred_keep)
        except ValueError:
            score = None
            pass
        return score


class Precision(BinaryMetric):
    def _metric(self, y_pred, y_target):
        score = precision_score(y_target, y_pred, average="binary" if self.mode == "eval" else "micro")
        return score


class Recall(BinaryMetric):
    def _metric(self, y_pred, y_target):
        score = recall_score(y_target, y_pred, average="binary" if self.mode == "eval" else "micro")
        return score
