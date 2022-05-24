from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data
from dataclasses import dataclass

from typing import Optional


@dataclass
class Graph:
    """Graph container class with some functionality

    Args:
        nodes (Tensor): coordinates of the graph nodes
        edge_index (Tensor): The 2 x e edge index of the graph
        edge_attr (Tensor): The edge weights
        probabilities (Optional[Tensor]): The probability of an edge existing
    """

    nodes: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    probabilities: Optional[Tensor] = None

    @classmethod
    def fc_from_random_geometry(cls, n_nodes: int, n_dims: int) -> Graph:
        nodes = torch.rand(n_nodes, n_dims)
        edge_index = fc_edge_index(n_nodes)
        edge_attr = (nodes[edge_index[0]] - nodes[edge_index[1]]).pow(2).sum(1).sqrt()
        return cls(nodes=nodes, edge_index=edge_index, edge_attr=edge_attr)

    @property
    def num_nodes(self):
        return self.nodes.shape[0]

    @property
    def dim(self):
        return self.nodes.shape[1]


def fc_edge_index(size: int) -> Tensor:
    """Generate the edge index (2 x size^2) of a fully connected graph

    Args:
        size (int): Number of nodes

    Returns:
        Tensor: The edge index
    """
    return torch.ones(size, size).nonzero().T


def pairwise_edge_distance(X: Tensor, edge_index: Tensor) -> Tensor:
    """Calculate the length of all edges for points X.

    Args:
        X (Tensor): Points between which edge distances are calculated
        edge_index (Tensor): Edges

    Returns:
        Tensor: Pairwise edge lenths.
    """
    # return (X[edge_index[0]] - X[edge_index[1]]).pow(2).sum(-1).sqrt()
    return torch.cdist(X, X).view(-1)


def geom_to_fc_graph(X: Tensor) -> Data:
    """Join all points in input into a fully connected graph with
    edge weights equal to the euclidian distance between them.

    Args:
        X (Tensor): Points in space.

    Returns:
        Data: Graph data object, fully connected with distance weights.
    """
    edge_index = fc_edge_index(X.shape[0])
    edge_attr = pairwise_edge_distance(X, edge_index)

    return Data(
        x=X,
        edge_attr=edge_attr,
        edge_index=edge_index
    )