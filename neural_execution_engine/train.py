import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import warnings
from config import NeuralExecutionConfig
import os

from neural_execution_engine.models import PrimsSolver


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_mst_acc(preds, real):
    with torch.no_grad():
        return (preds == real).float().mean()


def eval_model(model: PrimsSolver, loader: DataLoader) -> float:
    with torch.no_grad():
        predecessor_accuracy = 0.
        for data in loader:
            preds = model(data)
            predecessor_accuracy += batch_mst_acc(preds.argmax(1), data.predecessor[-1])

    predecessor_accuracy /= len(loader)
    return predecessor_accuracy


def instantiate_prims_solver(
    train_loader: DataLoader, 
    val_loader: DataLoader,
    config: NeuralExecutionConfig
) -> PrimsSolver:

    solver = PrimsSolver.from_config(config)

    if (not config.load_model) and (not config.train_model):
        warnings.warn('NOT LOADING OR TRAINING NEURALISED CLUSTERING MODEL, CHECK PARAMETERS ARE CORRECT')

    # if config.load_model:
    #     solver.load_state_dict(torch.load(config.load_from, map_location=device))

    if os.path.exists(config.save_to):
        solver.load_state_dict(torch.load(config.load_from, map_location=device))
        return solver
        
    if not config.train_model:
        return solver

    mst_loss_fn = torch.nn.BCEWithLogitsLoss()
    pred_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(solver.parameters())
    losses = []

    graph_size = config.n_nodes
    len_loader = len(list(train_loader))

    best_accuracy = 0
    for epoch in tqdm(range(config.n_epochs)):
        batch_ctr = 0.
        acc_avg = 0.
        pred_acc = 0.

        # Train
        for data in train_loader:
            h = torch.zeros((config.n_nodes, config.emb_dim), device=device)

            mst_loss = 0.
            pred_loss = 0.

            acc_avg_data = 0.
            for step in range(graph_size - 1):
                prev_tree = data.x[:, step:(step+1)]
                current_tree = data.y[:, step:(step+1)]
                predecessors = data.predecessor[-1].long()
                mask = data.y[:, step].bool()

                encoded = solver.encoder(prev_tree, h) 
                h = solver.processor(x=encoded, edge_attr=data.edge_attr,
                                     edge_index=data.edge_index, hidden=h)
                mst_logits = solver.mst_decoder(encoded, h)
                pred_logits = solver.predecessor_decoder(encoded, h, data.edge_index)

                mst_loss += mst_loss_fn(mst_logits, current_tree)
                pred_loss += pred_loss_fn(pred_logits[mask], predecessors[mask])

                in_mst = (mst_logits > 0).float()
                acc_avg_data += batch_mst_acc(in_mst, current_tree)
                batch_ctr += 1

            pred_acc += batch_mst_acc(pred_logits.argmax(1), predecessors)
            acc_avg += acc_avg_data/(graph_size-1)
            loss = pred_loss + mst_loss
            loss /= data.graph_size - 1
            losses.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = eval_model(solver, val_loader)
        print('final pred acc', pred_acc/len_loader)
        print('average inmst acc', acc_avg/len_loader)
        print(f'test predecessor accuracy {accuracy}')

        if config.save_to:
            if accuracy > best_accuracy:
                torch.save(solver.state_dict(), config.save_to)
                print(f'{accuracy} > {best_accuracy}. Saving model.')
                best_accuracy = accuracy


    return solver

