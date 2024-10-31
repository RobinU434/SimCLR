import torch
from exceptions.exceptions import InvalidBackboneError
import torchvision.models as models
import torch.nn as nn


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        input_shape: torch.Size,
        pretrained: bool = False,
    ):
        super(ResNetEncoder, self).__init__()
        self.out_dim: int
        self.input_shape = input_shape
        self.backbone = self._get_basemodel(base_model_name, pretrained=pretrained)

    def _get_basemodel(self, model_name: str, pretrained: bool = False) -> nn.Module:
        try:
            model_cls = getattr(models, model_name)
        except AttributeError as exc:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34, resnet50, resnet101, resnet152"
            ) from exc

        weights = None
        if pretrained:
            num = int(model_name[6:])
            weights = getattr(models, f"ResNet{num}_Weights").IMAGENET1K_V1
        model = model_cls(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1], nn.Flatten())

        mockup_tensor = torch.zeros(self.input_shape)
        self.out_dim = model.forward(mockup_tensor).shape[1]

        return model

    def forward(self, x):
        return self.backbone(x)
