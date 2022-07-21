import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import coloredlogs
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor

from key2med.lightning_models.lightning_model import LightningModel, Output
from key2med.utils.helper import create_class_instance

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


class RNNModel(LightningModel):
    """
    RNNModel
    """

    def __init__(
        self,
        image_dim: int,
        channels: int,
        latent_dim: int,
        label_dim: int,
        encoder: Dict,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        learning_rate: float = 0.1,
        learning_rate_scheduler: Optional[Dict] = None,
        use_previous_labels: bool = True,
        cheat: bool = False,
    ):
        """
        Constructor of RNNModel.

        Parameters
        ----------
        image_dim: int
            Dimension of the image
        channels: int
            number of channels for model
        latent_dim: int

        label_dim: int
            dimension of the output features
        use_previous_labels: bool
            whether or not previous labels are used
        cheat: bool
        """
        super().__init__(learning_rate=learning_rate, learning_rate_scheduler_config=learning_rate_scheduler)
        self.image_dim = image_dim
        self.channels = channels
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.use_previous_labels = use_previous_labels
        self.cheat = cheat
        if self.cheat:
            assert self.use_previous_labels

        self.encoder = create_class_instance(
            encoder["module"],
            encoder["name"],
            **encoder["args"],
            label_dim=self.label_dim,
            latent_dim=self.latent_dim,
            image_dim=self.image_dim,
            channels=self.channels,
        )
        self.classifier = torch.nn.Sequential(torch.nn.Linear(self.latent_dim, self.label_dim), torch.nn.Sigmoid())
        rnn_input_size = self.encoder.latent_dim
        if self.use_previous_labels:
            rnn_input_size += self.label_dim
        if rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=self.latent_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif rnn_type == "gru":
            self.rnn = torch.nn.GRU(
                input_size=rnn_input_size,
                hidden_size=self.latent_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise NotImplementedError

        self.loss_function = F.binary_cross_entropy

    def forward(self, x: List[Tensor], y: List[Tensor], lens: List[int]):
        """
        Implementation of Forward Pass.

        L1, ...,LN sequence lengths for each example, C channels, D image dimension

        Parameters
        ----------
        x : List[Tensor]
            Shape [L1 + L2 + ... + LN, C, D, D]
        y : List[Tensor]
            Shape [L1 + L2 + ... + LN, K]
        lens : List[int]
            [L1, L2, ..., LN]

        Returns
        -------
        prediction
        """
        z = [self.encoder(x_.to(self.device)) for x_ in x]
        if self.use_previous_labels:
            input_sequence = [torch.cat([z_, y_.to(self.device)], dim=1) for z_, y_ in zip(z, y)]
        else:
            input_sequence = z

        input_sequence = torch.nn.utils.rnn.pad_sequence(input_sequence, batch_first=True)
        input_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            input_sequence, batch_first=True, lengths=lens, enforce_sorted=False
        )

        if isinstance(self.rnn, torch.nn.LSTM):
            _, (hn, _) = self.rnn(input_sequence)
        else:
            _, hn = self.rnn(input_sequence)
        y_pred = self.classifier(hn)
        return y_pred

    def predict(self, x, y, lens):
        """
        predict

        Parameters
        ----------
        x
        y
        lens

        Returns
        -------

        """
        raise NotImplementedError("Predict is not implemented but should be.")

    def loss(self, y_pred: Tensor, y_target: Tensor, *args, **kwargs):
        """
        Returns the loss given in the yaml file used for optimization.

        Parameters
        y_pred: Tensor
            predicted output label by the model
        y_target: Tensor
            teacher output label

        Returns
        -------

        """

        loss = self.loss_function(y_pred, y_target)

        return loss

    def process_batch(self, batch: Tuple[FloatTensor, Tensor, Tensor], *args, **kwargs) -> Output:
        x, y, lens = batch

        # Last labels of sequence are target labels
        # Copy those into y_target and overwrite with 0. so the model
        # does not see them during prediction
        y = [y_.clone() for y_ in y]
        y_target = torch.stack([y_[-1].clone() for y_ in y])
        if not self.cheat:
            for y_ in y:
                y_[-1] = 0.0

        # Train loss
        y_pred = self.forward(x, y, lens)
        y_pred = y_pred.type(y[0].type()).view(y_target.shape)

        loss = self.loss(y_pred=y_pred, y_target=y_target)
        output = {"loss": loss, "pred": y_pred.detach().cpu().numpy(), "target": y_target.detach().cpu().numpy()}
        return output

    def training_step(self, batch: Tuple[FloatTensor, Tensor, Tensor], *args, **kwargs) -> Output:
        """
        Implementation of Train Step.

        Parameters
        ----------
        batch: Tuple[FloatTensor, Tensor]
            training batch
        args
        kwargs


        Returns
        -------

        """
        output = self.process_batch(batch)
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def validation_step(self, batch: Tuple[FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Validate Step.
        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------

        """
        output = self.process_batch(batch)
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Validate Step.
        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------

        """
        output = self.process_batch(batch)
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Validate Step.
        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------

        """
        output = self.process_batch(batch)
        return output


class NoRNNModel(RNNModel):
    """
    NoRNNModel
    """

    def predict(self, x, y, lens):
        pass

    def __init__(self, *args, **kwargs):
        """
        Constructor of NoRNNModel.
        This is a simple DenseNet in the RNN-Framework.
        We ignore all images but the last image, skip the RNN and only use the DenseNet-Encoding for classification.

        Parameters
        ----------
        See super()-class RNNModel for parameters
        """
        super(NoRNNModel, self).__init__(*args, **kwargs)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.latent_dim, self.label_dim), torch.nn.Sigmoid()
        )

    def forward(self, x: List[Tensor], y: List[Tensor], lens: List[int]):
        """
        Implementation of Forward Pass.

        L1, ...,LN sequence lengths for each example, C channels, D image dimension

        Parameters
        ----------
        x : List[Tensor]
            Shape [L1 + L2 + ... + LN, C, D, D]
        y : List[Tensor]
            Shape [L1 + L2 + ... + LN, K]
        lens : List[int]
            [L1, L2, ..., LN]

        Returns
        -------
        prediction
        """
        x = torch.stack([x_[-1] for x_ in x])
        z = self.encoder(x.to(self.device))
        y_pred = self.classifier(z)
        return y_pred
