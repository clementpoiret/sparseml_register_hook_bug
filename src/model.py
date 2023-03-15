import pytorch_lightning as pl
import torch
from sparseml.pytorch.models import ModelRegistry
from torch import nn

registry = ModelRegistry()


class Regressor(nn.Module):

    def __init__(self, n_predictions: int, dropout_rate: float = 0.2):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(256, 1280, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(1280, n_predictions),
        )

    def forward(self, x):
        x = self.regressor(x)

        # Normally, the backbone returns a tuple of (x, x).
        # As returning only x causes an error when calling the backbone,
        # we return a tuple of (x, dummy) instead.
        return x, torch.tensor(0)


class LightningModel(pl.LightningModule):

    def __init__(
        self,
        n_predictions: int,
        dropout_rate: float = 0.2,
        model_name: str = "efficientnet_v2_s",
        pretrained: bool = True,
        lr: float = 5e-4,
    ):
        super().__init__()

        self.lr = lr

        self.backbone = registry.create(model_name, pretrained=pretrained)

        # Classifier is: conv -> bn -> SiLU -> pool -> dropout -> fc
        self.backbone.classifier = Regressor(n_predictions, dropout_rate)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.backbone(x)[0]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
