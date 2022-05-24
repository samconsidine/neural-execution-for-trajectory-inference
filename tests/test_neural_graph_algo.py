import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from config import NeuralExecutionConfig
from neural_execution_engine.models import PrimsSolver
from neural_execution_engine.datagen import gen_prims_data_instance
from neural_execution_engine.datagen.prims import generate_prims_dataset 
from utils.debugging import print_gradients

import torch
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

device = 'cpu'


def test_prims_solver(
    loader: Data, 
    config: NeuralExecutionConfig
) -> PrimsSolver:

    solver = PrimsSolver.from_config(config)

    mst_loss_fn = torch.nn.BCELoss()
    pred_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(solver.parameters())
    losses = []

    graph_size = config.n_nodes

    for epoch in tqdm(range(config.n_epochs), disable=True):
        pred_acc = 0.

        for data in loader:
            # Train
            h = torch.zeros((config.n_nodes, config.latent_dim), device=device)

            mst_loss = 0.
            pred_loss = 0.

            for step in range(graph_size - 1):
                prev_tree = data.x[:, step:(step+1)]
                current_tree = data.y[:, step:(step+1)]
                predecessors = data.predecessor[step].long().to(device) 
                mask = data.y[:, step].bool()
                breakpoint()

                encoded = solver.encoder(prev_tree, h) 
                h = solver.processor(x=encoded, edge_attr=data.edge_attr,
                                     edge_index=data.edge_index, hidden=h)

                mst_logits = solver.mst_decoder(encoded, h)
                pred_logits = solver.predecessor_decoder(encoded, h, data.edge_index)

                mst_loss += mst_loss_fn(mst_logits, current_tree)
                pred_loss += pred_loss_fn(pred_logits[mask], predecessors[mask])

            loss = pred_loss + mst_loss
            loss /= data.graph_size - 1
            print('='*80)
            print(pred_logits.argmax(1))
            print(predecessors)
            print('total loss:', loss.item(), ' | mst loss:', mst_loss.item(), ' | predecessor loss', pred_loss.item())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    breakpoint()
    return solver


def seed_everything(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(0)


def test_solver_on_synthetic_data(config):
    solver = PrimsSolver.from_config(config)
    if config.load_from:
        solver.load_state_dict(torch.load(config.load_from))
        print('loaded solver from file')

    seed_everything(2)
    # print(config.n_data, 
    #                                        config.n_nodes,
    #                                        config.latent_dim)
    # input()
    prims_dataset = generate_prims_dataset(config.n_data, 
                                           config.n_nodes,
                                           config.latent_dim)
    # print(prims_dataset[0])
    # print(prims_dataset[0].edge_index[:5])
    # print(prims_dataset[0].edge_weights[:5])
    # ta = list(gen_prims_data_instance(2, n_nodes=config.n_nodes, n_dims=config.latent_dim))
    for item in prims_dataset:
        print(item.predecessor[-1])
        print(solver(None, item).argmax(1))


if __name__=="__main__":
    seed_everything(2)
    config = NeuralExecutionConfig(n_nodes=10, latent_dim=32, n_epochs=10000, learning_rate=1e-3, batch_size=2)
    # data = list(gen_prims_data_instance(2, n_nodes=config.n_nodes, n_dims=config.latent_dim))
    # loader = DataLoader(data, batch_size=config.batch_size)
    # model = PrimsSolver.from_config(config)

    # test_prims_solver(loader, config)

    test_solver_on_synthetic_data(config)