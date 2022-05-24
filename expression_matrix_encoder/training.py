from tqdm import tqdm
from config import EncoderClusterConfig
from utils import combine_params
from expression_matrix_encoder.models import AutoEncoder, CentroidPool

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Tuple


def train_autoencoder_clusterer(
    data: Tensor, 
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
    cluster_loss_fn = torch.nn.MSELoss()

    params = combine_params(autoencoder, pool)
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    dataset = DataLoader(data)

    for epoch in range(config.n_epochs):
        mean_loss, total_cluster_loss, total_recon_loss = 0, 0, 0

        for X in dataset:
            latent, reconstruction = autoencoder(X)
            recon_loss = recon_loss_fn(reconstruction, X)

            min_dist, min_idx = torch.cdist(latent, pool.coords).min(1)
            clust_loss = cluster_loss_fn(min_dist, torch.zeros_like(min_dist))

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