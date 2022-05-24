from torch import Tensor
from torch.nn import Module
from utils.graphs import Graph
from utils.debugging import test_gradient

from linalg.projections import project_onto_mst


def mst_reconstruction_loss_fn(
    latent: Tensor,  # Output of encoder
    mst: Graph,   # Tensor of edge_index
    X: Tensor,
    autoencoder: Module
) -> float:

    projection_probabilities, projected_coords = project_onto_mst(latent, mst)

    reconstructions = autoencoder.decoder(projected_coords)
    reconstruction_loss = ((X.unsqueeze(1) - reconstructions).pow(2)).sum(-1).sqrt()
    reg = (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities).sum(1)
    loss = reconstruction_loss * (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities) / reg

    return loss.mean()
