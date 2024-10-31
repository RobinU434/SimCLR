import torch

from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class NTCrossEntropy:
    def __init__(
        self,
        tau: float = 1,
        normalize: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self._tau = tau
        self._normalize = normalize
        self._device = device

        self._cross_entropy = torch.nn.CrossEntropyLoss().to(self._device)

    def _get_label(self, batch_size: int, n_views: int = 2):
        diag = torch.diag(torch.ones(batch_size), diagonal=batch_size)
        labels = torch.eye(n_views * batch_size) + diag + diag.T
        labels = labels.to(self._device)
        return labels

    def __call__(self, z_0: Tensor, z_1: Tensor) -> Tensor:
        n_views = 2
        batch_size = len(z_0)

        features = torch.cat([z_0, z_1], dim=0)
        label = self._get_label(batch_size, n_views)

        if self._normalize:
            features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(label.shape[0], dtype=torch.bool).to(self._device)
        label = label[~mask].view(label.shape[0], -1)
        logits = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # NOTE: logsoftmax not needed because CrossEntropy gets the unnormalized logits
        logits = logits / self._tau

        loss = self._cross_entropy(logits, label)
        return loss, logits, label
