import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
from typing import Any, Optional
from utils.resnet_flex import ResNet, myresnet18
import utils.resnet_flex as resnet

from pytorch_lightning.utilities.types import STEP_OUTPUT


class WristResNetwork(pytorch_lightning.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # init implementation of resnet-18
        self.model = myresnet18(in_channels=1, num_classes=4, pretrained=False)

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model.forward(x)

    def training_step(self, batch, *args, **kwargs):
        x = batch["image"]
        y = (batch["points"].squeeze(axis=0)).squeeze(axis=0)
        train_loss = F.mse_loss(self.forward(x), y)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, *args, **kwargs):
        x = batch["image"]
        y = (batch["points"].squeeze(axis=0)).squeeze(axis=0)
        val_loss = F.mse_loss(self.forward(x), y)
        self.log("val_loss", val_loss)
        return val_loss

# TODO: WHAT is the difference between test_step and predict_step?
    def test_step(self, batch, *args, **kwargs):
        x = batch["image"]
        y = (batch["points"].squeeze(axis=0)).squeeze(axis=0)
        y_hat = self.forward(x)
        test_loss = F.mse_loss(y_hat, y)
        self.log("test_loss", test_loss)

        return {'test_loss': test_loss, 'preds': y_hat, 'target': y}

        # trainer = Trainer()
        # trainer.fit(model)

        # automatically loads the best weights for you
        # trainer.test(model)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # enable Monte Carlo Dropout
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0001)

