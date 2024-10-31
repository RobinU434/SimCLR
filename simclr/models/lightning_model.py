from typing import Any, Dict, Tuple
from lightning import LightningModule
import torch
import torch.optim as optim
from simclr.metrics.nt_xent import NTCrossEntropy
from simclr.models.encoder import ResNetEncoder
from simclr.models.projection_head import ProjectionHead, Passthrough
from simclr.models.resnet_simclr import SimCLRNet
from simclr.utils.filesystem import load_yaml
from simclr.utils.config import Config
from utils import accuracy


class SimCLRLightning(LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        input_shape: torch.Size,
        T_max: int, # max number of steps
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.config = Config(**config)
        self._device = device
        self.input_shape = input_shape
        self.T_max = T_max
        self.model: SimCLRNet

        self.criterion = NTCrossEntropy(device=device)

    def setup(self, stage):
        encoder = ResNetEncoder(**self.config.ResNetEncoder.__dict__, input_shape=self.input_shape)
        
        if self.config.projection_head_enabled:
            projection_head = ProjectionHead(**self.config.ProjectionHead.__dict__, input_dim=encoder.out_dim)
        else:
            projection_head = Passthrough()

        self.model: SimCLRNet = SimCLRNet(encoder, projection_head).to(self._device)


    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x_0, x_1 = batch
        z_0 = self.model.forward(x_0)
        z_1 = self.model.forward(x_1)
        loss, logits, labels = self.criterion(z_0, z_1)

        labels = torch.argwhere(labels)[:, 1]
        top_1, top_5 = accuracy(logits, labels, topk=(1, 5))

        return loss, {
            "train_loss": loss,
            "train_top_1_acc": top_1,
            "train_top_5_acc": top_5,
        }

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x_0, x_1 = batch
        z_0 = self.model.forward(x_0)
        z_1 = self.model.forward(x_1)
        loss, logits, labels = self.criterion(z_0, z_1)

        labels = torch.argwhere(labels)[:, 1]
        top_1, top_5 = accuracy(logits, labels, topk=(1, 5))

        return loss, {"val_loss": loss, "val_top_1_acc": top_1, "val_top_5_acc": top_5}

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer_name)(
            self.model.parameters(), lr=float(self.config.optimizer_params.lr)
        )

        warmup_epochs = self.config.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=0, last_epoch=-1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
