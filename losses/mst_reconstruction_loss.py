from torch import Tensor
from torch.nn import Module
from utils.graphs import Graph

from linalg.projections import project_onto_mst


def mst_reconstruction_loss_fn(
    latent: Tensor,  # Output of encoder
    mst: Graph,   # Tensor of edge_index
    X: Tensor,
    autoencoder: Module
) -> float:

    projection_distances, projected_coords = project_onto_mst(latent, mst)
    projection_probabilities = projection_distances.softmax(1)

    reconstructions = autoencoder.decoder(projected_coords)
    reconstruction_loss = ((X.unsqueeze(1) - reconstructions).pow(2)).sum(-1).sqrt()
    reg = (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities).sum(1)
    loss = reconstruction_loss * (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities) / reg.unsqueeze(-1)

    return loss.mean()


def mst_reconstruction_loss_with_backbone(
    latent: Tensor,
    mst: Graph,
    X: Tensor,
    autoencoder: Module,
    mst_reconstruction_coef: float,
    backbone_distance_coef: float
) -> float:

    projection_distances, projected_coords = project_onto_mst(latent, mst)
    projection_probabilities = projection_distances.softmax(1)

    reconstructions = autoencoder.decoder(projected_coords)
    reconstruction_loss = ((X.unsqueeze(1) - reconstructions).pow(2)).sum(-1).sqrt()
    reg = (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities).sum(1)
    mst_loss = reconstruction_loss * (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities) / reg.unsqueeze(-1)
    ## The dims here are really suspicious
    distance_loss = projection_distances.min(1).values / projection_distances.mean(1)

    return mst_reconstruction_coef * mst_loss.mean() + backbone_distance_coef * distance_loss.mean()
