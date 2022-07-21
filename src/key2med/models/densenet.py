import logging
from typing import Any, Dict, List, Tuple

import coloredlogs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.manual_seed(0)

from key2med.models.model import BaseModel, Output
from key2med.utils.helper import create_class_instance, get_class_nonlinearity

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


class DenseNet(BaseModel):
    """
    Initialization of DenseNet.

    Parameters
    ----------
    label_dim: int
        dimension of the output features
    pretrained: bool
        whether or not to use a pretrained model
    fine_tuning: str
        specifies which parts of the model should be finetuned, where rest is frozen: "model" / "classifier" / "classifier+last"
    model_path: str
        path to .pth file of the pretrained model one wants to finetune
    imratio: float
        when loss_function is AUCMLoss
    imratios: Dict
        when loss_function is AUCM_MultiLabel
    nonlinearity: Dict
        determines which nonlinear output function should be used in between sequential layers
    learning_rate: float
        learning rate for optimizer
    model_number: int
        model id for DenseNets: 121, 169, 201 etc.
    loss_function: Dict
        specifies which loss function to use in the training, usually set to BCELoss
    skip_classifier: bool
        bool variable to decide whether or not to have a classifier layer at the end
    **kwargs:
        all the other arguments necessary for model initialisation definition
    """

    def __init__(
        self,
        label_dim: int,
        pretrained: bool = True,
        fine_tuning: str = "model",
        model_path: str = None,
        imratio: float = 0.1,
        imratios=None,
        nonlinearity=None,
        learning_rate: float = 0.01,
        model_number: int = 121,
        loss_function: Dict = None,
        skip_classifier: bool = False,
        **kwargs,
    ):

        super().__init__(
            learning_rate=learning_rate,
            learning_rate_scheduler_config=kwargs.get("learning_rate_scheduler", None),
            weight_decay=kwargs.get("weight_decay", 0),
        )
        self.label_dim = label_dim
        if "force_label_dim" in kwargs:
            self.label_dim = kwargs["force_label_dim"]
        self.pretrained = pretrained
        self.fine_tuning = fine_tuning
        self.imratio = imratio
        self.imratios = imratios
        drop_out = kwargs.get("drop_out", 0)
        if nonlinearity is not None:
            self.output_nonlinearity = get_class_nonlinearity(nonlinearity)()
        else:
            self.output_nonlinearity = nn.Sigmoid()

        if model_number == 121:
            self.model = torchvision.models.densenet121(pretrained=self.pretrained, drop_rate=drop_out)
        elif model_number == 169:
            self.model = torchvision.models.densenet169(pretrained=self.pretrained)
        elif model_number == 201:
            self.model = torchvision.models.densenet201(pretrained=self.pretrained)
        elif model_number == 161:
            self.model = torchvision.models.densenet161(pretrained=self.pretrained)
        else:
            raise NotImplementedError(
                "DenseNet Model not included in torchvision.models, define model_number as (121, 169, 201)"
            )
        self.latent_dim = self.model.classifier.in_features

        self.loss_function = self.init_loss_function(loss_function)

        if skip_classifier:
            self.model.classifier = None
        else:
            self.model.classifier = nn.Sequential(nn.Linear(self.latent_dim, self.label_dim), self.output_nonlinearity)
        if model_path is not None:
            logger.info(f"Loading state dict for densenet model from path\n\t{model_path}")
            self.load_state_dict(torch.load(model_path, map_location="cuda:0")["model_state"])

    def init_loss_function(self, params: Dict = None):
        """
        Initializes the Loss Function.

        Parameters
        ----------
        params: Dict
            parameter to specify which loss_function to use

        Returns
        -------
        Instance of loss function
        """
        if params is None:
            return torch.nn.BCELoss()
        if "AUCM_MultiLabel" in params["name"]:
            return create_class_instance(params["module"], params["name"], params["args"], imratio=self.imratios)
        if "AUCMLoss" in params["name"]:
            return create_class_instance(params["module"], params["name"], params["args"], imratio=self.imratio)
        return create_class_instance(params["module"], params["name"], params["args"])

    def forward(self, x: torch.FloatTensor):
        """
        Does a forward pass.

        Parameters
        ----------
        x: torch.FloatTensor
            minibatch being forwarded into the model

        Returns
        -------
        model's prediction
        """
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if self.model.classifier is not None:
            out = self.model.classifier(out)
        return out

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, *args, **kwargs):
        """
        Calculates the loss.

        Parameters
        ----------
        y_pred: torch.Tensor
            predicted output label by the model
        y_target: torch.Tensor
            teacher output label

        Returns
        -------
        loss value between predicted and true labels in a batch
        """

        if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
            y_target = y_target.type(torch.LongTensor).to(self.device)

        y_target = y_target.to(torch.float32)
        loss = self.loss_function(y_pred, y_target).to(torch.float32)

        return loss

    def training_step(self, batch: Tuple[torch.FloatTensor, torch.Tensor], *args, **kwargs) -> Output:
        """
        Implementation of Train Step.

        Parameters
        ----------
        batch: Tuple[torch.FloatTensor, torch.Tensor]
            training batch
        args
        kwargs

        Returns
        -------
        metrics of training step
        """

        if self.fine_tuning != "model":
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            if self.fine_tuning == "classifier":
                for parameter in self.model.classifier.parameters():
                    parameter.requires_grad = True
            if self.fine_tuning == "classifier+last":
                for parameter in self.model.features.denseblock4.parameters():  #
                    parameter.requires_grad = True

        output = self.process_batch(batch)
        self.log(f"train/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Validate Step.

        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------
        metrics of validation step
        """
        output = self.process_batch(batch)
        self.log(f"valid/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def test_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Test Step.

        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------
        metrics of a test step
        """
        output = self.process_batch(batch)
        self.log(f"test/loss", output["loss"], on_epoch=True, on_step=True)
        return output

    def predict_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], *args, **kwargs) -> Output:
        """
        Implementation of Predict Step.

        Parameters
        ----------
        batch
        *args
        **kwargs

        Returns
        -------
        metrics of a predict step
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
