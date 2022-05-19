import torch
from torch.nn import Module, Parameter

device = 'cpu'


class NeuralisedClustering(Module):
    def __init__(self):
        super().__init__()


class CentroidPool(NeuralisedClustering):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = Parameter(torch.rand(n_clusts, n_dims, requires_grad=True))
        self.to(device)


class KMadness(NeuralisedClustering):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = Parameter(torch.rand(size=(n_clusts, n_dims)))
        self.to(device)

    @property
    def w(self):
        return f1(self.coords).T

    @property
    def b(self):
        return f2(sqnorm(self.coords))

    def fc(self, h):
        return (-(-h).topk(2, 1)[0])[:, 1, :]

    def forward(self, x):
        h = torch.matmul(self.w, x.T).T + self.b
        cluster_assignments = self.fc(h) > 0
        return cluster_assignments

    def closest_centroids(self, assignments):
        return self.coords[assignments.max(-1)[1]]


def f1(tensor):
    tensor = tensor.permute(1, 0)
    return torch.sub(tensor.unsqueeze(dim=2), tensor.unsqueeze(dim=1))


def f2(tensor):
    return tensor.unsqueeze(0).T - tensor


def sqnorm(tensor):
    return torch.linalg.norm(tensor, dim=1)**2


