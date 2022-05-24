import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import scanpy as sc

from config import EncoderClusterConfig, ExperimentConfig, default_config
from expression_matrix_encoder.models import AutoEncoder, CentroidPool, KMadness
from expression_matrix_encoder.models.clustering import IDEC
from expression_matrix_encoder.training import train_autoencoder_clusterer
from neural_execution_engine.datagen.prims import generate_prims_dataset
from neural_execution_engine.train import instantiate_prims_solver
from losses import cluster_loss_fn, mst_reconstruction_loss_fn

from typing import Tuple

from utils.graphs import Graph, fc_edge_index, pairwise_edge_distance, geom_to_fc_graph
from utils.torch import combine_params, seed_everything
from utils.debugging import ensure_gradients, test_gradient
from utils.plotting import plot_mst, plot_output, test_results

config = default_config


def train_clusterer(
    X: Tensor, 
    latent_dim: int, 
    n_centroids: int, 
    config: EncoderClusterConfig
) -> Tuple[AutoEncoder, CentroidPool]:

    autoencoder = AutoEncoder(X.shape[1], latent_dim)
    centroid_pool = CentroidPool(n_centroids, latent_dim)
    autoencoder, centroid_pool = train_autoencoder_clusterer(X, autoencoder, centroid_pool, config)

    return autoencoder, centroid_pool


def train_narti(config: ExperimentConfig):
    recon_loss_fn = torch.nn.MSELoss()

    paul15 = sc.datasets.paul15()
    X = torch.tensor(paul15.X).float()

    # autoencoder, centroid_pool = train_clusterer(
    #     X=X,
    #     latent_dim=config.latent_dimension, 
    #     n_centroids=config.number_of_centroids,
    #     config=config.encoder_cluster_config
    # )

    autoencoder_clust = IDEC(X.shape[1], [256, 128, 128], 32, [128, 128, 256], 10)
    from itertools import repeat
    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    train_data_etc = DataLoader([(x, y, blank) for x, y, blank in zip(X, torch.tensor(label_enc.fit_transform(paul15.obs['paul15_clusters'].values)), repeat(1))], batch_size=32)
    if config.encoder_cluster_config.load_model:
        autoencoder_clust.load_state_dict(torch.load(config.encoder_cluster_config.load_autoencoder_from))
    else:
        autoencoder_clust.fit(train_data_etc, train_data_etc, config.encoder_cluster_config.load_autoencoder_from)
    autoencoder = autoencoder_clust.ae
    centroid_pool = autoencoder_clust
    reconstruction, assignments, latent = autoencoder_clust(X)
    plot_output(latent, assignments, paul15.obs['paul15_clusters'])

    seed_everything(2)

    prims_dataset = generate_prims_dataset(config.neural_exec_config.n_data, 
                                           config.number_of_centroids,
                                           config.neural_exec_config.latent_dim)
                                        
    val_dataset = generate_prims_dataset(config.neural_exec_config.n_data, 
                                         config.number_of_centroids,
                                         config.neural_exec_config.latent_dim)

    prims_solver = instantiate_prims_solver(prims_dataset, val_dataset, config.neural_exec_config)
    optimizer = torch.optim.Adam(
        combine_params(autoencoder, centroid_pool, prims_solver),
        lr=config.learning_rate
    )


    for epoch in range(config.n_epochs):
        for batch_x in DataLoader(X, batch_size=config.batch_size):
            reconstruction, assignments, latent = autoencoder_clust(batch_x)

            edges = fc_edge_index(config.number_of_centroids)
            weights = pairwise_edge_distance(centroid_pool.coords, edges)

            data = Data(x=centroid_pool.coords, edge_index=edges, edge_attr=weights)
            predecessor_logits = prims_solver(data)
            mst = Graph(nodes=centroid_pool.coords, edge_index=edges, edge_attr=weights,
                        probabilities=predecessor_logits.softmax(1))

            mst_recon_loss = mst_reconstruction_loss_fn(latent, mst, batch_x, autoencoder)
            recon_loss = recon_loss_fn(reconstruction, batch_x)
            cluster_loss = cluster_loss_fn(latent, centroid_pool.coords)

            print(f"{mst_recon_loss=}, {recon_loss=}, {cluster_loss=}")

            loss = (config.recon_loss_coef * recon_loss
                  + config.mst_loss_coef * mst_recon_loss
                  + config.cluster_loss_coef * cluster_loss)

            optimizer.zero_grad()
            loss.backward()
            ensure_gradients(autoencoder.encoder, autoencoder.decoder, centroid_pool,
                             prims_solver.encoder, prims_solver.processor, prims_solver.mst_decoder,
                             prims_solver.predecessor_decoder)
            optimizer.step()

            print(loss.item())

            with torch.no_grad():
                latent, _ = autoencoder(X)
                pred_logits = prims_solver(data)
                test_results(X, centroid_pool, paul15.obs['paul15_clusters'], pred_logits, autoencoder)


if __name__ == "__main__":
    train_narti(default_config)