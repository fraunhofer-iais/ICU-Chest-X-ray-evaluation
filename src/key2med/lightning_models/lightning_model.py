from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, TypedDict, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

from key2med.utils.helper import create_class_instance


class Output(TypedDict):
    loss: Optional[torch.FloatTensor]
    pred: Optional[np.ndarray]
    target: Optional[np.ndarray]


class OptimizerDict(TypedDict):
    optimizer: Optimizer
    lr_scheduler: Optional[Dict]


Loss = torch.FloatTensor
Predictions = torch.FloatTensor


class LightningModel(pl.LightningModule, ABC):
    def __init__(self, learning_rate: float = 0.1, learning_rate_scheduler_config: Optional[Dict] = None, weight_decay: int = 1.0e-05):
        super().__init__()

        self.learning_rate = learning_rate
        self.learning_rate_scheduler_config = learning_rate_scheduler_config
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self, x: torch.FloatTensor, *args, **kwargs):
        """

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs) -> torch.FloatTensor:
        """

        :param y_pred:
        :param y_target:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def training_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        ...

    def configure_optimizers(self) -> Union[Optimizer, OptimizerDict]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.learning_rate_scheduler_config is None:
            return optimizer
        learning_rate_scheduler: Dict = self.init_learning_rate_scheduler(
            self.learning_rate_scheduler_config, optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": learning_rate_scheduler}

    @staticmethod
    def init_learning_rate_scheduler(learning_rate_scheduler_config: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        additional_kwargs = {}
        for key in ["interval", "frequency", "monitor", "strict"]:
            if key in learning_rate_scheduler_config["args"]:
                additional_kwargs[key] = learning_rate_scheduler_config["args"].pop(key)
        return {
            "scheduler": create_class_instance(
                learning_rate_scheduler_config["module"],
                learning_rate_scheduler_config["name"],
                learning_rate_scheduler_config["args"],
                optimizer=optimizer,
            ),
            **additional_kwargs,
        }
