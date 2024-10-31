from torch import Tensor
import torch.nn as nn


class SimCLRNet(nn.Module):
    def __init__(self, encoder: nn.Module, projection_head: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.projection_head = projection_head

        self.latent: Tensor

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encoder.forward(x)
        self.latent = latent.clone()
        out = self.projection_head.forward(latent)
        return out
