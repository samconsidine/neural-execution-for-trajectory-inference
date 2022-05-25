import torch
from torch import Tensor
from torch.linalg import vector_norm
from utils.graphs import Graph

from typing import Tuple


def lineseg_projection(p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Project n points onto the closest point of s line segments in d dimensions.

    Args:
        p1 (Tensor): Shape s x d - The starting points of the line segments.
        p2 (Tensor): Shape s x d - The finishing points of the line segments.
        p3 (Tensor): Shape n x d - The points to be projected onto the segments.

    Returns:
        Tensor: Shape n x s x d - A 3d tensor of the projection points of each
        of the n points onto the s line segments.
    """
    l2 = torch.sum((p1 - p2)**2, dim=-1) # s
    p1p2 = p2 - p1  # s x d
    p1p3 = p3.unsqueeze(1) - p1.unsqueeze(0)  # n x 1 x d - 1 x s x d -> n x s x d
    
    t = torch.sum(p1p2 * p1p3, dim=-1)
    t = (t * (l2 > 0)) / (l2 + 1e-8)
    t = torch.clamp(t, min=0, max=1)
    projection  =  p1 + t.unsqueeze(-1) * p1p2
    # n x s x d =  s x d     n x s x d
    return projection


def project_onto_mst(
    latent: Tensor,  # n x d
    tree: Graph
) -> Tuple[Tensor, Tensor]:
    """Project each point in `latent` onto the MST.

    Args:
        latent (Tensor): The embedded cells, output of the encoder.

    Returns:
        Tuple[Tensor, Tensor]: The projection points of each cell onto the MST as
        well as the probability of each projection.
    """

    edges = tree.edge_index

    from_segments = tree.nodes[edges[0]]
    to_segments = tree.nodes[edges[1]]
    projection_coords = lineseg_projection(from_segments, to_segments, latent)  # n x s x d

    distances = vector_norm(latent.unsqueeze(1).broadcast_to(projection_coords.shape) - 
                            projection_coords, dim=-1)

    return distances, projection_coords
