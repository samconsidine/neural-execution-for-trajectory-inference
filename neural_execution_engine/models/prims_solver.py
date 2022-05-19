from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, LeakyReLU, Sigmoid, Sequential, GRUCell
from torch_geometric.nn import MessagePassing

from config import NeuralExecutionConfig
from utils import fc_edge_index
from utils.graphs import pairwise_edge_distance
from utils.debugging import test_gradient


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrimsSolver(Module):
    def __init__(
        self, 
        num_nodes: int,
        latent_dim: int,
        encoder: Encoder,
        processor: ProcessorNetwork,
        mst_decoder: MSTDecoder,
        predecessor_decoder: PredecessorDecoder,
        **kwargs
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.processor = processor
        self.mst_decoder = mst_decoder
        self.predecessor_decoder = predecessor_decoder

    def forward(self, X: Tensor) -> Tensor:
        h = torch.zeros((self.num_nodes, self.latent_dim))
        prev_tree = torch.zeros(self.num_nodes, 1).long()
        edge_index = fc_edge_index(self.num_nodes)
        edge_weights = pairwise_edge_distance(X, edge_index)

        for step in range(self.num_nodes - 1):
            encoded = self.encoder(prev_tree, h)
            h = self.processor(x=encoded, edge_attr=edge_weights,
                               edge_index=edge_index, hidden=h)

            mst_logits = self.mst_decoder(encoded, h)
            pred_logits = self.predecessor_decoder(encoded, h, edge_index)
            prev_tree[mst_logits.argmax()] = 1

        # We're only interested in the final prediction
        return pred_logits

    @classmethod
    def from_config(cls, config: NeuralExecutionConfig) -> PrimsSolver:
        n_nodes = config.n_nodes
        latent_dim = config.latent_dim
        output_dim = config.batch_size * config.n_nodes
        node_features = config.node_features
        return cls(
            num_nodes=n_nodes,
            latent_dim=latent_dim,
            encoder=Encoder(node_feature_dim=node_features, latent_dim=latent_dim),
            processor=ProcessorNetwork(in_channels=latent_dim, out_channels=latent_dim),
            mst_decoder=MSTDecoder(latent_dim=latent_dim),
            predecessor_decoder=PredecessorDecoder(latent_dim=latent_dim, n_outputs=output_dim)
        )


class ProcessorNetwork(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='max', bias=False,  # Channels?
            flow='source_to_target', use_gru=False):

        super(ProcessorNetwork, self).__init__(aggr=aggr, flow=flow)

        self.M = Sequential(
            Linear(2*in_channels+1, out_channels, bias=bias),
            LeakyReLU(),
            Linear(out_channels, out_channels, bias=bias),
            LeakyReLU()
        )

        self.U = Sequential(
            Linear(2*in_channels, out_channels, bias=bias),
            LeakyReLU()
        )

        self.use_gru = use_gru

        if use_gru:
            self.gru = GRUCell(out_channels, out_channels, bias=bias)

        self.out_channels = out_channels

        self.to(device)

    def forward(self, x, edge_attr, edge_index, hidden):
        out = self.propagate(edge_index, x=x, hidden=hidden, edge_attr=edge_attr)

        if not self.training:
            out = torch.clamp(out, -1e9, 1e9)

        return out

    def message(self, x_i, x_j, edge_attr):
        edge_weights_col_vec = edge_attr.unsqueeze(0).T
        return self.M(torch.cat((x_i, x_j, edge_weights_col_vec), dim=1))

    def update(self, aggr_out, x, hidden):

        if self.use_gru:
            out = self.gru(self.U(torch.cat((x, aggr_out), dim=1)), hidden)
        else:
            out = self.U(torch.cat((x, aggr_out), dim=1))

        return out


class Encoder(Module):
    def __init__(self, node_feature_dim: int, latent_dim: int):
        super().__init__()

        self.layers = Sequential(
            Linear(node_feature_dim + latent_dim, latent_dim),
            ReLU()
        )

        self.to(device)

    def forward(self, prev_tree: Tensor, latent: Tensor) -> Tensor:
        model_in = torch.cat([prev_tree, latent], axis=1)
        return self.layers(model_in)


class MSTDecoder(Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.layers = Sequential(
            Linear(latent_dim * 2, 1),
            Sigmoid()
        )

        self.to(device)

    def forward(self, encoded: Tensor, h: Tensor) -> Tensor:
        model_in = torch.cat([encoded, h], axis=1)
        return self.layers(model_in)


class PredecessorDecoder(Module):
    def __init__(self, latent_dim: int, n_outputs: int):
        super().__init__()
        self.layers = Sequential(
            Linear(latent_dim * 4, latent_dim),
            ReLU(),
            Linear(latent_dim, 1)
        )
        self.n_outputs = n_outputs
        self.to(device)

    def forward(self, encoded: Tensor, h: Tensor, edge_index: Tensor) -> Tensor:
        left_encoded = encoded[edge_index[0]]
        right_encoded = encoded[edge_index[1]]
        left_h = h[edge_index[0]]
        right_h = h[edge_index[1]]

        out = self.layers(torch.cat((left_encoded, left_h, right_encoded, right_h), axis=1))

        out = out.reshape((-1, self.n_outputs))
        return out
