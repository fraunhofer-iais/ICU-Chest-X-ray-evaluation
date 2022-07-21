import logging
from typing import Tuple

import coloredlogs
import torch
import torch.nn as nn

from key2med.lightning_models.lightning_model import LightningModel, Output

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


class MLP(LightningModel):
    """
    DenseNet
    """

    def __init__(self, image_dim: int, label_dim: int, channels: int, learning_rate: float = 0.01):
        """
        Constructor of DenseNet

        Parameters
        ----------
        kwargs
        """
        super().__init__(learning_rate=learning_rate)
        self.label_dim = label_dim
        self.input_dim = (image_dim**2) * channels
        self.output_nonlinearity = nn.Sigmoid()

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim // 100),
            torch.nn.Linear(self.input_dim // 100, self.label_dim),
        )

        self.loss_function = torch.nn.BCELoss()

    def forward(self, x: torch.FloatTensor):
        """

        :param x:
        :return:
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        logits = self.linear_layers(x)
        probs = self.output_nonlinearity(logits)
        return probs

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs):
        """

        :param y_pred:
        :param y_target:
        :return:
        """

        loss = self.loss_function(y_pred, y_target)
        return loss

    def training_step(self, batch: Tuple[torch.FloatTensor, torch.Tensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        output = self.process_batch(batch)
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        output = self.process_batch(batch)
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        output = self.process_batch(batch)
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        output = self.process_batch(batch)
        return output

    def process_batch(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """

        :param batch:
        :param args:
        :param kwargs:
        :return:
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.forward(x)
        loss = self.loss(y_pred=y_pred, y_target=y)

        output = {"loss": loss, "pred": y_pred.detach().cpu().numpy(), "target": y.detach().cpu().numpy()}
        return output
