from expression_matrix_encoder.training import train_autoencoder_clusterer
from expression_matrix_encoder.models import AutoEncoder, CentroidPool

from torch.nn import Module


class NARTI(Module):
    def __init__(self, data, num_clusters):
        ...

    def forward(self, X):
        ...