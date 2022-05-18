import torch
from torch import Tensor


def cluster_loss_fn(X: Tensor, coords: Tensor) -> float:
    assignments = torch.cdist(X, coords).argmin(1)
    loss = 0.
    for clust in range(assignments.max().item()):
        data = X[assignments == clust]
        mu = data.mean(0)
        loss += (data - mu).square().sum(1).sqrt().sum()

    return loss