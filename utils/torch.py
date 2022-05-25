import torch

from typing import List


def combine_params(*modules: List[torch.nn.Module]) -> torch.nn.Parameter:
    params = []
    for module in modules:
        params += module.parameters()
    return params


def join_params(*parameters: List[torch.nn.Parameter]) -> torch.nn.Parameter:
    params = []
    for param in parameters:
        params += param
    return params

def seed_everything(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(0)

# Fuctions taking in a module and applying the specified weight initialization
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except: pass

def cluster_acc(Y_pred, Y):
    import numpy as np
    from scipy.optimize import linear_sum_assignment as linear_assignment

    assert Y_pred.size == Y.size , "Sizes do not match"
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(*ind)])*1.0/Y_pred.size, w


def freeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad = False

