from typing import Optional
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    AUROC,
)


class MulticlassClassificationTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.05,
        frozen_feature_extractor: Optional[nn.Module] = None,
    ):
        super(MulticlassClassificationTask, self).__init__()
        self.save_hyperparameters(logger=False)

        self.model = model
        self.lr = lr
        self.frozen_feature_extractor = frozen_feature_extractor

        self.train_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.train_cm = ConfusionMatrix(task="multiclass", num_classes=3)
        self.train_auroc = AUROC(task="multiclass", num_classes=3, average="macro")

        self.val_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.val_cm = ConfusionMatrix(task="multiclass", num_classes=3)
        self.val_auroc = AUROC(task="multiclass", num_classes=3, average="macro")

    def forward(self, x):
        if self.frozen_feature_extractor is None:
            y_pred = self.model(x)
        else:
            self.frozen_feature_extractor.eval()
            with torch.no_grad():
                x = self.frozen_feature_extractor(x)
            y_pred = self.model(x)
        if not self.training:
            y_pred = torch.softmax(y_pred, dim=-1)
        return y_pred

    def training_step(self, batch):
        x, y_true = batch

        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y_true)

        self.log("train_loss", loss)
        self.update_logs(y_pred, y_true)

        return loss

    def on_train_epoch_end(self):
        self.make_logs()

    def validation_step(self, batch):
        x, y_true = batch

        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y_true)

        self.log("val_loss", loss)
        self.update_logs(y_pred, y_true)

    def on_validation_epoch_end(self):
        self.make_logs()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=2e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=10, T_mult=1, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def update_logs(self, y_pred, y_true):
        if self.training:
            acc, cm, auroc = (
                self.train_accuracy,
                self.train_cm,
                self.train_auroc,
            )
        else:
            acc, cm, auroc = (
                self.val_accuracy,
                self.val_cm,
                self.val_auroc,
            )

        y_pred = torch.softmax(y_pred, dim=-1)

        acc.update(y_pred, y_true)
        cm.update(y_pred.argmax(dim=-1), y_true)
        auroc.update(y_pred, y_true)

    def make_logs(self):
        if self.training:
            acc, cm, auroc = (
                self.train_accuracy,
                self.train_cm,
                self.train_auroc,
            )
            metric_prefix = "train"
        else:
            acc, cm, auroc = (
                self.val_accuracy,
                self.val_cm,
                self.val_auroc,
            )
            metric_prefix = "val"

        self.log(f"{metric_prefix}_accuracy", acc.compute())
        self.log_confusion_matrix(f"{metric_prefix}_cm", cm.compute())
        self.log(f"{metric_prefix}_auroc", auroc.compute())

        acc.reset()
        cm.reset()
        auroc.reset()

    def log_confusion_matrix(self, name, cm):
        df_cm = pd.DataFrame(cm.cpu().numpy(), index=range(3), columns=range(3))
        df_cm = df_cm.rename_axis(index="True", columns="Predicted")
        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(df_cm, annot=True)
        fig = ax.get_figure()
        plt.close(fig)
        self.logger.experiment.add_figure(name, fig, self.global_step)
