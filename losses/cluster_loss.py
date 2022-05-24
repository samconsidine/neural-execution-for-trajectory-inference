import torch
from torch import Tensor


def cluster_loss_fn(X: Tensor, coords: Tensor) -> float:
    """Cluster loss function based on distance to nearest cluster

    Args:
        X (Tensor): Onput tensor
        coords (Tensor): coordinates

    Returns:
        float: sum of distances to clostest centroid.
    """
    assignments = torch.cdist(X, coords).argmin(1)
    loss = 0.
    for clust in range(assignments.max().item()):
        data = X[assignments == clust]
        mu = data.mean(0)
        loss += (data - mu).abs().sum(1).sum()

    return loss


def cluster_training_loss_fn(latent, target, pool):
    cluster_distances = torch.cdist(latent, pool.coords)
    return cluster_distances.gather(1, target.unsqueeze(-1)).mean()