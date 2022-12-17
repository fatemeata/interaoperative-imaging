import torch
import torch.nn.functional as F
import pytorch_lightning
from typing import Any
from utils.unet import UNet


class RadiusNetwork(pytorch_lightning.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # init implementation of resnet-18
        self.model = UNet(in_channels=1, num_classes=2)

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model.forward(x)

    def training_step(self, batch, *args, **kwargs):
        x = batch["image"]
        print(x.shape)
        y = batch["radius"]
        train_loss = F.binary_cross_entropy_with_logits(self.forward(x), y)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, *args, **kwargs):
        x = batch["image"]
        y = batch["radius"]
        val_loss = F.binary_cross_entropy_with_logits(self.forward(x), y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, *args, **kwargs):
        x = batch["image"]
        y = batch["radius"]
        y_hat = self.forward(x)
        test_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("test_loss", test_loss, on_step=True, prog_bar=True, logger=True)

        return {'test_loss': test_loss, 'preds': y_hat, 'target': y}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x = batch["image"]
        y = batch["radius"]
        pred = self.forward(x)
        return pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

