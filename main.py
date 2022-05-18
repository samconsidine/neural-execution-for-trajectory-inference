import torch
from torch import Tensor
from torch.utils.data import DataLoader
import scanpy as sc

from config import EncoderClusterConfig, default_config
from expression_matrix_encoder.models import AutoEncoder, CentroidPool
from expression_matrix_encoder.training import train_autoencoder_clusterer
from neural_execution_engine.datagen.prims import generate_prims_dataset
from neural_execution_engine.train import instantiate_prims_solver
from utils.graphs import Graph, fc_edge_index
from losses import cluster_loss_fn, mst_reconstruction_loss_fn

from typing import Tuple

from utils.torch import combine_module_params

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


def train_narti(config):
    recon_loss_fn = torch.nn.MSELoss()

    data = sc.datasets.paul15()
    X = torch.tensor(data.X).float()

    autoencoder, centroid_pool = train_clusterer(
        X=X,
        latent_dim=config.latent_dimension, 
        n_centroids=config.number_of_centroids,
        config=config.encoder_cluster_config
    )

    prims_dataset = generate_prims_dataset(config.neural_exec_config.n_data, 
                                           config.number_of_centroids,
                                           config.latent_dimension)

    prims_solver = instantiate_prims_solver(prims_dataset, config.neural_exec_config)
    optimizer = torch.optim.Adam(combine_module_params(autoencoder, centroid_pool, prims_solver))

    for epoch in range(config.n_epochs):
        for batch_x in DataLoader(X, batch_size=config.batch_size):
            latent, reconstruction = autoencoder(batch_x)
            
            predecessor_logits = prims_solver(latent)

            edges = fc_edge_index(config.number_of_centroids)
            weights = ((centroid_pool.coords[edges[0]] - centroid_pool.coords[edges[1]])
                        .pow(2).sum(1).sqrt())

            mst = Graph(nodes=centroid_pool.coords, edge_index=edges, edge_weights=weights,
                        probabilities=predecessor_logits.softmax(1))

            mst_recon_loss = mst_reconstruction_loss_fn(latent, mst, batch_x, autoencoder.decoder)
            recon_loss = recon_loss_fn(reconstruction, batch_x)
            cluster_loss = cluster_loss_fn(latent, centroid_pool.coords)

            loss = (config.recon_loss_coef * recon_loss 
                + config.mst_loss_coef * mst_recon_loss
                + config.cluster_loss_coef * cluster_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())


def test_results(inputs, ae):
    with torch.no_grad():
        import seaborn as sns
        import matplotlib.pyplot as plt

        pca_latent = PCA(2)
        outs = ae.encoder(inputs).detach().numpy()
        if N_DIMS > 2:
            outs = pca_latent.fit_transform(outs)
        xs = outs[:, 0]
        ys = outs[:, 1]

        pca_centroids = PCA(2)
        coords = pool.coords
        if N_DIMS > 2:
            coords = pca_centroids.fit_transform(coords)
        cx = coords[:, 0]
        cy = coords[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=labels.values, legend=False)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
        print(tree_logits)
        plot_mst(tree_logits, ae.encoder(X).detach().numpy(), pool.coords, epoch)
        plt.savefig(f'plots/mst/{sys.argv[1]}/{epoch}.png')
        plt.clf()

        from_nodes = mst.edges[0]
        to_nodes = mst.edges[1]



if __name__ == "__main__":
    train_narti(default_config)