import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    mean_squared_error,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)
from sklearn.metrics._classification import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import logging

import coloredlogs

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)

"""

Functions/Classes for multiclass multilabel classification,
i.e. one output is a vector over N classes, each entry a probability in [0, 1]
and class association is non-exclusive.

Example:
y      = [0,   1,   1,   0]
y_pred = [0.5, 0.9, 0.1, 0.2]

Each metric must be a class,
which will be instantiated and then called with __call__

E.g.
 class Accuracy:

    def __call__(y_pred, y_target):
        return (y_pred == y_target).sum() / y_pred.nelement()

"""


class MultiLabelMetric:
    mode: str = "train"

    def __init__(self, evaluate_on: List[str] = ("train", "eval")):
        self.evaluate_on: List[str] = evaluate_on

    def map_values(self, y_pred, y_target):
        raise NotImplementedError

    def __call__(self, y_pred, y_target, train: bool = True, **kwargs) -> Optional[float]:
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        if not isinstance(y_target, np.ndarray):
            y_pred = np.array(y_target)
        assert y_pred.shape == y_target.shape
        assert len(y_pred.shape) == 2  # [batch_size, label_dim]
        self.mode = "train" if train else "eval"
        if self.mode not in self.evaluate_on:
            return None

        y_pred, y_target = y_pred.copy(), y_target.copy()
        y_pred, y_target = self.map_values(y_pred, y_target)

        return self._metric(y_pred, y_target, **kwargs)

    def _metric(self, y_pred, y_target, **kwargs) -> Optional[Any]:
        raise NotImplementedError


class BinaryCrossEntropy(MultiLabelMetric):
    def map_values(self, y_pred, y_target):
        return y_pred, y_target

    def cross_entropy(self, predictions, targets):
        N = predictions.shape[0]
        ce = -np.sum(targets * np.log(predictions)) / N
        return ce

    def _metric(self, y_pred, y_target, **kwargs):
        return self.cross_entropy(y_pred, y_target)


class MSE(MultiLabelMetric):
    def map_values(self, y_pred, y_target):
        return y_pred, y_target

    def _metric(self, y_pred, y_target, **kwargs):
        return mean_squared_error(y_pred, y_target)


class MultiLabelMetricBinary(MultiLabelMetric):
    def map_values(self, y_pred, y_target):
        if self.mode == "train":
            # map the uncertainty onto a "2" class
            y_pred[y_pred < 0.55] = 0.0
            y_pred[y_pred > 0.85] = 1.0
            y_pred[0.3 >= abs(y_pred - 0.55)] = 2
            y_target[y_target < 0.55] = 0.0
            y_target[y_target > 0.85] = 1.0
            y_target[0.3 >= abs(y_target - 0.55)] = 2
        else:
            y_pred[y_pred < 0.5] = 0.0
            y_pred[y_pred >= 0.5] = 1.0
        return y_pred, y_target

    def _metric(self, y_pred, y_target, **kwargs) -> Optional[Any]:
        raise NotImplementedError


class MultiLabelMetricProbs(MultiLabelMetric):
    def map_values(self, y_pred, y_target):
        return y_pred, y_target

    def _metric(self, y_pred, y_target, **kwargs) -> Optional[Any]:
        raise NotImplementedError


class ClassificationReport(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs) -> str:
        index_to_label = kwargs.get("label_names")
        if index_to_label is None:
            num_labels = y_pred.shape[1]
            index_to_label = [str(label).zfill(2) for label in range(num_labels)]
        try:
            return classification_report(y_true=y_target, y_pred=y_pred, target_names=index_to_label)
        except ValueError as e:
            return f"ValueError: {str(e)}"


class MultilabelConfusionMatrix(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs) -> str:
        index_to_label = kwargs.get("label_names")
        if index_to_label is None:
            num_labels = y_pred.shape[1]
            index_to_label = [str(label).zfill(2) for label in range(num_labels)]
        output = ""
        for index, label in enumerate(index_to_label):
            output += f"{label}:\n"
            output += self._confusion_matrix(y_target[:, index], y_pred[:, index])
            output += "\n"
        return output

    def _confusion_matrix(self, y_target, y_pred) -> str:
        assert len(y_target.shape) == len(y_pred.shape) == 1
        df = pd.DataFrame({"y_target": y_target, "y_pred": y_pred})
        confusion_matrix = pd.crosstab(df["y_target"], df["y_pred"], rownames=["Actual"], colnames=["Predicted"])
        return str(confusion_matrix)


class HammingLoss(MultiLabelMetricBinary):
    """
    y_target = [0, 1, 0, 1]
    x_pred = [1, 1, 0, 0]
    HammingLoss = 50%
    """

    def _metric(self, y_pred, y_target, **kwargs):
        temp = 0
        for i in range(y_target.shape[0]):
            temp += np.size(y_target[i] == y_pred[i]) - np.count_nonzero(y_target[i] == y_pred[i])
        return temp / (y_target.shape[0] * y_target.shape[1])


class ConfusionMatrix(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        target_1d = y_target.reshape((-1, 1))
        pred_1d = y_pred.reshape((-1, 1))
        return confusion_matrix(target_1d, pred_1d)


class AUROC(MultiLabelMetricProbs):
    average = "micro"

    def _metric(self, y_pred, y_target, **kwargs):
        """
        Calculate AUROC score for each class separately, then average over all classes.
        If train mode, uncertain values are mapped onto a '2' class or inbetween 0. and 1.. Here we ignore these datapoints.
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

        keep_indices = slice(None)
        if self.mode == "train":
            keep_indices = np.all((y_target == 1.0) | (y_target == 0.0), axis=1)
        y_target, y_pred = y_target[keep_indices], y_pred[keep_indices]

        try:
            return roc_auc_score(y_target, y_pred, average=self.average)
        except ValueError:
            return None


class AUROCMacro(AUROC):
    average = "macro"


class AUROCWeighted(AUROC):
    average = "weighted"


class ConfidenceInterval(MultiLabelMetricProbs):
    def _metric(self, y_pred, y_target, **kwargs):

        keep_indices = slice(None)
        if self.mode == "train":
            keep_indices = np.all((y_target == 1.0) | (y_target == 0.0), axis=1)
        y_target, y_pred = y_target[keep_indices], y_pred[keep_indices]

        accuracy = roc_auc_score(y_target, y_pred, average="weighted")

        confidence = 0.95  # Change to your desired confidence level
        z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)

        interval = z_value * np.sqrt((accuracy * (1 - accuracy)) / y_target.shape[0])

        # ci_lower = accuracy - interval
        # ci_upper = accuracy + interval
        return interval


class ZeroOneLoss(MultiLabelMetricBinary):
    """
    Calculates loss on exact match
    ZeroOneLoss([0, 1, 0, 1], [1, 1, 0, 0]) = 1.0
    ZeroOneLoss([1, 1, 0, 0], [1, 1, 0, 0]) = 0.0

    For batches, returns the average over the batch
    ZeroOneLoss([[1, 0, 0, 0],
                 [0, 1, 0, 0]],
                [[0, 1, 0, 0],
                 [0, 1, 0, 0]]) = 0.5
    """

    def _metric(self, y_pred, y_target, **kwargs) -> float:
        return zero_one_loss(y_target, y_pred)


class PredictionPositiveRate(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        return y_pred.sum() / y_pred.size


class TargetPositiveRate(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        return y_target.sum() / y_target.size


class PositiveRateDiff(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        return y_target.sum() / y_target.size - y_pred.sum() / y_pred.size


class MicroF1Score(MultiLabelMetricBinary):
    """
    Computes the micro average f1-score over all classes, i.e. for each class
    the f1-score over the batch is calculated, then averaged.
    """

    def _metric(self, y_pred, y_target, **kwargs):
        y_target = y_target.reshape((-1, 1))
        y_pred = y_pred.reshape((-1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = f1_score(y_target, y_pred, average="micro")
        return score


class MacroF1Score(MultiLabelMetricBinary):
    """
    Computes the macro average f1-score over all classes, i.e. for each class
    the f1-score over the batch is calculated, then averaged.
    """

    def _metric(self, y_pred, y_target, **kwargs):
        y_target = y_target.reshape((-1, 1))
        y_pred = y_pred.reshape((-1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = f1_score(y_target, y_pred, average="macro")
        return score


class PrecisionFirstClass(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        y_pred[y_pred >= 0.5] = 1.0
        y_pred[y_pred < 0.5] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = precision_score(y_target, y_pred, average="macro", labels=[0])
        return score


class RecallFirstClass(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        y_pred[y_pred >= 0.5] = 1.0
        y_pred[y_pred < 0.5] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = recall_score(y_target, y_pred, average="macro", labels=[0])
        return score


class F1FirstClass(MultiLabelMetricBinary):
    def _metric(self, y_pred, y_target, **kwargs):
        y_pred[y_pred >= 0.5] = 1.0
        y_pred[y_pred < 0.5] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = f1_score(y_target, y_pred, average="macro", labels=[0])
        return score
