import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import NeuralExecutionConfig

from neural_execution_engine.models import PrimsSolver

device = 'cpu'


def instantiate_prims_solver(
    loader: DataLoader, 
    config: NeuralExecutionConfig
) -> PrimsSolver:

    solver = PrimsSolver.from_config(config)

    mst_loss_fn = torch.nn.BCELoss()
    pred_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(solver.parameters())
    losses = []

    graph_size = config.n_nodes

    for epoch in tqdm(range(config.n_epochs)):
        # batch_ctr = 0.
        # acc_avg = 0.
        # pred_acc = 0.

        # Train
        for data in loader:
            h = torch.zeros((config.n_nodes, config.latent_dim), device=device)

            mst_loss = 0.
            pred_loss = 0.

            for step in range(graph_size - 1):
                prev_tree = data.x[:, step:(step+1)]
                current_tree = data.y[:, step:(step+1)]
                predecessors = data.predecessor[-1].long().to(device) 

                encoded = solver.encoder(prev_tree, h) 
                h = solver.processor(x=encoded, edge_attr=data.edge_weights,
                                     edge_index=data.edge_index, hidden=h)
                mst_logits = solver.mst_decoder(encoded, h)
                pred_logits = solver.predecessor_decoder(encoded, h, data.edge_index)

                mst_loss += mst_loss_fn(mst_logits, current_tree)
                pred_loss += pred_loss_fn(pred_logits, predecessors)

                in_mst = (mst_logits > 0.5).float()  # Why > 0?
                # acc_avg += batch_mst_acc(in_mst, current_tree)
                # batch_ctr += 1

            # pred_acc += batch_mst_acc(pred_logits.argmax(1), predecessors)
            loss = pred_loss + mst_loss
            loss /= data.graph_size - 1
            losses.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return solver

