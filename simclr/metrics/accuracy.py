from typing import List, Tuple
import torch


def accuracy(
    predictions: torch.Tensor, target: torch.Tensor, top_k: Tuple[int] = (1,)
) -> List[float]:
    """Multi-Class top-k accuracy

    Args:
        predictions (torch.Tensor): 2D array with logits or prediction distribution (n_samples, classes)
        target (torch.Tensor): 1D array true class index (n_samples, )
        top_k (Tuple[int], optional): which top k accuracy would you like to have. Defaults to (1,).

    Returns:
        List[float]: accuracies is the same order as k (n_k, )
    """
    with torch.no_grad():
        _, pred = predictions.topk(max(top_k), 1, True, True)
        hits = target[:, None] == pred

        res = []
        for k in top_k:
            hit = hits[:, :k].sum(dim=1).bool()
            res.append(hit.float().mean())
        return res
