from typing import Callable
import torch.nn as nn
from torch import Tensor


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,  # hidden dimension in the original paper?
        activation_func: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.net = None
        if activation_func is None:
            # single layer
            self.net = nn.Sequential(nn.Linear(input_dim, output_dim))
        else:
            activation_func = getattr(nn, activation_func)()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_func,
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)


class Passthrough(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x
