import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from typing import Tuple, List
from utils.graphs import Graph
import random


INF = torch.tensor(torch.inf, dtype=torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_prims_training_instance(graph: Graph) -> Tuple[Tensor, Tensor, Tensor]:
    visited = torch.zeros(graph.num_nodes, dtype=torch.bool)
    random_idx = random.randint(0, graph.num_nodes - 1)
    visited[random_idx] = True
    predecessor = torch.zeros(graph.num_nodes)
    predecessor.fill_(torch.nan)
    predecessor[random_idx] = random_idx

    all_visited = []
    all_prev_visited = []
    all_predecessors = []

    for _ in range(graph.num_nodes-1):
        prev_visited = visited.clone()

        next_node_flat_idx = mask_visited(graph.edge_index, graph.edge_attr, visited).argmin().item()
        from_node_idx, to_node_idx = divmod(next_node_flat_idx, graph.num_nodes) 
        visited[to_node_idx] = True
        predecessor[to_node_idx] = from_node_idx

        all_visited.append(visited.float())
        all_prev_visited.append(prev_visited.float())
        all_predecessors.append(predecessor.clone())

    return (torch.stack(all_visited, axis=1), 
            torch.stack(all_prev_visited, axis=1), 
            torch.stack(all_predecessors, axis=1))


def rand_geometric_graphs_iterator(num_graphs: int, n_nodes: int, n_dims: int) -> Graph:
    for _ in range(num_graphs):
        yield Graph.fc_from_random_geometry(n_nodes, n_dims)


def mask_visited(edge_index: Tensor, edge_attr: Tensor, visited: Tensor) -> Tensor:
    not_visited = torch.logical_not(visited)
    dense_weights = to_dense_adj(edge_index, edge_attr=edge_attr)
    masked_weights = dense_weights * visited.unsqueeze(1) * not_visited
    return torch.where(masked_weights != 0, masked_weights, INF)

class PrimData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'predecessor':
            return self.num_nodes
        return super().__inc__(key, value, *args, *kwargs)

def gen_prims_data_instance(n_data: int, n_nodes: int, n_dims: int) -> Data:
    for graph in rand_geometric_graphs_iterator(n_data, n_nodes, n_dims):
        x_next, x_prev, predecessor  = generate_prims_training_instance(graph)
        # breakpoint()
        # predecessor = predecessor.T  # For batching

        yield PrimData(
            x=x_prev.to(device), 
            y=x_next.to(device), 
            stackx=x_prev.unsqueeze(0).to(device),
            stacky=x_next.unsqueeze(0).to(device),
            edge_attr=graph.edge_attr.to(device),
            edge_index=graph.edge_index.to(device), 
            predecessor=predecessor.to(device),
            graph_size=graph.num_nodes,
            geom=graph.nodes
        )


def generate_prims_dataset(n_data, n_nodes, n_dims) -> List[Data]:
    dataset = []
    for data in gen_prims_data_instance(n_data, n_nodes, n_dims):
        dataset.append(data)
    return dataset
