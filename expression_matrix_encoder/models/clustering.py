import torch
from torch.nn import Module, Parameter

device = 'cpu'


class CentroidPool(Module):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = Parameter(torch.rand(n_clusts, n_dims, requires_grad=True))
        self.to(device)
