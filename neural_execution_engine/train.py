import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.loader.dataloader import DataLoader
from tqdm import tqdm
from yaml import warnings
from config import NeuralExecutionConfig
import os

from neural_execution_engine.models import PrimsSolver


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_mst_acc(preds, real):
    with torch.no_grad():
        return (preds == real).float().sum(), preds.shape[0]

def batch_mst_acc_next_mst_node(preds, real):
    with torch.no_grad():
        assert real.sum() != 0, breakpoint()
        return (preds&real).sum()/real.sum()

def pad_batch(thing_to_pad, batch_ids, padding_value=1e9):
    padded_ttp = []
    for i in torch.unique(batch_ids):
        mskcurrbatchel = batch_ids == i
        padded_ttp.append(thing_to_pad[mskcurrbatchel])
    return torch.nn.utils.rnn.pad_sequence(padded_ttp, batch_first=True, padding_value=1e9)

def eval_model(model: PrimsSolver, loader: DataLoader) -> float:
    with torch.no_grad():
        predecessor_correct, predecessor_all = 0., 0
        for data in loader:
            preds = model(data)
            correct, cnt = batch_mst_acc(preds.argmax(1), data.predecessor[:, -1])
            predecessor_correct += correct
            predecessor_all += cnt

    predecessor_accuracy = predecessor_correct / predecessor_all
    return predecessor_accuracy


def instantiate_prims_solver(
    train_loader: DataLoader,  # NOTE Wrong annotation
    val_loader: DataLoader, # NOTE Wrong annotation
    config: NeuralExecutionConfig
) -> PrimsSolver:

    solver = PrimsSolver.from_config(config)

    if (not config.load_model) and (not config.train_model):
        warnings.warn('NOT LOADING OR TRAINING NEURALISED CLUSTERING MODEL, CHECK PARAMETERS ARE CORRECT')

    # if config.load_model:
    #     solver.load_state_dict(torch.load(config.load_from, map_location=device))

    if os.path.exists(config.save_to) and config.load_model:
        solver.load_state_dict(torch.load(config.load_from, map_location=device))
        return solver
        
    if not config.train_model:
        return solver

    mst_loss_fn = torch.nn.BCEWithLogitsLoss()
    pred_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(solver.parameters())
    losses_mst = []
    losses_pred = []

    graph_size = config.n_nodes
    firsteverseen = -1
    for tl in train_loader:
        if firsteverseen == -1:
            firsteverseen = tl.predecessor.shape[0]
        assert tl.predecessor.shape[0] == firsteverseen
    train_loader = DataLoader(train_loader, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=config.batch_size, shuffle=False)
    len_loader = len(list(train_loader))

    best_accuracy = 0
    for epoch in tqdm(range(config.n_epochs)):
        acc_avg = 0.
        pred_acc = 0.
        pred_correct = 0.
        pred_all = 0.

        # Train
        for data in train_loader:
            h = torch.zeros((data.num_nodes, config.emb_dim), device=device)

            mst_loss = 0.
            pred_loss = 0.

            acc_avg_data = 0.
            in_mst = data.x[:, 0:1]
            for step in range(graph_size - 1):
                # breakpoint()
                # in_mst_b4 = in_mst.clone()
                prev_tree = data.x[:, step:(step+1)]
                prev_tree_padded = data.stackx[:, :, step]#pad_batch(prev_tree.bool().squeeze(-1), data.batch)
                # current_tree_in = data.x[:, step:(step+1)]
                current_tree = data.y[:, step:(step+1)]
                current_tree_padded = data.stacky[:, :, step]
                predecessors = data.predecessor[:, -1].long()
                mask = data.y[:, step].bool()

                encoded = solver.encoder(prev_tree, h) 
                h = solver.processor(x=encoded, edge_attr=data.edge_attr,
                                     edge_index=data.edge_index, hidden=h)
                mst_logits = solver.mst_decoder(encoded, h)
                mst_logits_padded = mst_logits.squeeze(-1).view(data.num_graphs, -1)#pad_batch(mst_logits.squeeze(-1), data.batch)
                chosen_node_padded = ~prev_tree_padded.bool()&current_tree_padded.bool()
                # prev_tree_padded = pad_batch(prev_tree.bool().squeeze(-1), data.batch)
                # chosen_node_padded = pad_batch((~prev_tree.bool()&current_tree.bool()).squeeze(-1), data.batch)
                # breakpoint()
                mst_logits_padded[prev_tree_padded.bool()] = -1e9

                pred_logits = solver.predecessor_decoder(encoded, h, data.edge_index, data.edge_attr)

                next_node_in_mst_real = chosen_node_padded.nonzero()[:, 1]
                mst_loss += F.cross_entropy(mst_logits_padded, next_node_in_mst_real)
                # mst_loss += F.binary_cross_entropy_with_logits(mst_logits, current_tree, pos_weight=current_tree.shape[0]/current_tree.sum())
                # mst_loss_fn(mst_logits, current_tree)
                pred_loss += pred_loss_fn(pred_logits[mask], predecessors[mask])

                next_node_in_mst = mst_logits_padded.argmax(-1)
                mst_correct, mst_all = batch_mst_acc(next_node_in_mst, next_node_in_mst_real)
                acc_avg_data += mst_correct/mst_all
                # breakpoint()
                # acc_avg_data += batch_mst_acc_next_mst_node((~in_mst_b4.bool())&(in_mst.bool()), (~data.x[:, step:(step+1)].bool()&current_tree.bool()))

            c_pred_correct, c_pred_all = batch_mst_acc(pred_logits.argmax(1), predecessors)
            pred_correct += c_pred_correct
            pred_all += c_pred_all
            acc_avg += acc_avg_data/(graph_size-1)
            loss = pred_loss + mst_loss
            loss /= graph_size - 1
            losses_pred.append(pred_loss.cpu().detach()/(graph_size-1))
            losses_mst.append(mst_loss.cpu().detach()/(graph_size-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('losses pred mean', torch.tensor(losses_pred).mean())
        print('losses mst mean', torch.tensor(losses_mst).mean())
        # print('final pred acc', pred_correct/pred_all)
        print('average inmst acc', acc_avg/len_loader)
        accuracy1 = eval_model(solver, train_loader)
        print(f'train predecessor accuracy {accuracy1}')
        # # accuracy2 = eval_model(solver, train_loader)
        # # print(f'train predecessor accuracy 2nd eval {accuracy2}')
        accuracy = eval_model(solver, val_loader)
        print(f'val predecessor accuracy {accuracy}')
        # # accuracy = eval_model(solver, val_loader)
        # # print(f'val predecessor accuracy 2nd eval {accuracy}')

        if config.save_to:
            if accuracy > best_accuracy:
                torch.save(solver.state_dict(), config.save_to)
                print(f'{accuracy} > {best_accuracy}. Saving model.')
                best_accuracy = accuracy


    return solver

