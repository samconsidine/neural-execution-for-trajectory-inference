from tqdm import tqdm
from config import EncoderClusterConfig
from losses.cluster_loss import cluster_training_loss_fn
from utils import combine_params
from expression_matrix_encoder.models import AutoEncoder, CentroidPool

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Tuple


def train_autoencoder_clusterer(
    data: Tensor, 
    target: Tensor,
    autoencoder: AutoEncoder, 
    pool: CentroidPool, 
    config: EncoderClusterConfig
) -> Tuple[AutoEncoder, CentroidPool]:
    """Function to train the autoencoder and clustering models.

    Args:
        dataset (DataLoader): An scRNA-seq dataset on which to perform the training
        autoencoder (AutoEncoder): An autoencoder model whose forward method returns the latent and reconstructed data.
        pool (CentroidPool): A centroid pool model for the clustering.
        config (EncoderClusterConfig): The encoder/clusterer config object.

    Returns:
        Tuple[AutoEncoder, CentroidPool]: The trained autoencoder and centroid pool.
    """
    if config.load_model:
        autoencoder.load_state_dict(torch.load(config.load_autoencoder_from))
        pool.load_state_dict(torch.load(config.load_clustering_from))
        print('print loaded both models from file, skipping autoencoder training')
        return autoencoder, pool

    recon_loss_fn = torch.nn.MSELoss()
    cluster_loss_fn = cluster_training_loss_fn

    params = combine_params(autoencoder, pool)
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    dataset = DataLoader(list(zip(data, target)), batch_size=128)

    for epoch in range(config.n_epochs):
        mean_loss, total_cluster_loss, total_recon_loss = 0, 0, 0

        for X, y in dataset:
            latent, reconstruction = autoencoder(X)
            recon_loss = recon_loss_fn(reconstruction, X)

            clust_loss = cluster_loss_fn(latent, y, pool)

            loss = (config.clust_loss_coef * clust_loss 
                  + config.recon_loss_coef * recon_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mean_loss += loss.item()
                total_cluster_loss += clust_loss.item()
                total_recon_loss += recon_loss.item()

        print(f"Epoch finished. Mean loss = {mean_loss}, cluster loss = {total_cluster_loss}, reconstruction loss = {total_recon_loss}")

    if config.save_autoencoder_to:
        torch.save(autoencoder.state_dict(), config.save_autoencoder_to)
    if config.save_clustering_to:
        torch.save(pool.state_dict(), config.save_clustering_to)

    return autoencoder, pool