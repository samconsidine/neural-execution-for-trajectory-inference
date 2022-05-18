from torch.nn import Sequential, Linear, ReLU, Sigmoid, Module

from typing import List


device = 'cpu'


class AutoEncoder(Module):
    """A standard pytorch autoencoder model.

    Args:
        emb_dim (int): The dimension of the latent space into which to find an encoding.
        input_dimension (int): The dimension of input data.
    """
    def __init__(self, in_dim, emb_dim: int):
        super().__init__()

        self.encoder = Sequential(
            Linear(in_dim, 128),
            ReLU(),
            Linear(128, emb_dim),
            Sigmoid()
        )
        self.decoder = Sequential(
            Linear(emb_dim, 128),
            ReLU(),
            Linear(128, in_dim)
        )

        self.to(device)

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)

        return encode, decode
