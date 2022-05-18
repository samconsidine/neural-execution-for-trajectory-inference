from torch import Tensor
from torch.nn import Module
from utils.graphs import Graph

from linalg.projections import project_onto_mst


def mst_reconstruction_loss_fn(
    latent: Tensor,  # Output of encoder
    mst: Graph,   # Tensor of edge_index
    X: Tensor,
    decoder: Module
) -> float:

    projection_probabilities, projected_coords = project_onto_mst(latent, mst)

    reconstructions = decoder(projected_coords)
    distances = ((X.unsqueeze(1) - reconstructions).pow(2)).sum(-1).sqrt()
    loss = distances * (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities)

    return loss.mean()
