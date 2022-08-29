from argparse import ArgumentError
import sys
from sklearn.preprocessing import LabelEncoder
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import scanpy as sc

from config import EncoderClusterConfig, ExperimentConfig, NeuralExecutionConfig
from dataprocessing.dataset import RNASeqDataset
from expression_matrix_encoder.models import AutoEncoder, CentroidPool, KMadness
from expression_matrix_encoder.training import train_autoencoder_clusterer
from linalg.projections import project_onto_mst
from losses.cluster_loss import cluster_training_loss_fn
from losses.mst_reconstruction_loss import mst_reconstruction_loss_with_backbone
from neural_execution_engine.datagen.prims import generate_prims_dataset
from neural_execution_engine.train import instantiate_prims_solver
from losses import cluster_loss_fn, mst_reconstruction_loss_fn

from typing import Tuple

from utils.graphs import Graph, fc_edge_index, pairwise_edge_distance, geom_to_fc_graph, sanity_check_neural_exec
from utils.torch import combine_params, freeze_model_weights, seed_everything
from utils.debugging import ensure_gradients, test_gradient
from utils.plotting import plot_centers, plot_clusters, plot_edge_probabilities, plot_latent, plot_latent_with_fc, plot_most_probable_mst, plot_mst, plot_single_cell_projection, test_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_clusterer(
    X: Tensor, 
    y: Tensor,
    latent_dim: int, 
    n_centroids: int, 
    config: EncoderClusterConfig
) -> Tuple[AutoEncoder, CentroidPool]:

    autoencoder = AutoEncoder(X.shape[1], latent_dim)
    centroid_pool = CentroidPool(n_centroids, latent_dim)
    autoencoder, centroid_pool = train_autoencoder_clusterer(X, y, autoencoder, centroid_pool, config)

    return autoencoder, centroid_pool


def train_narti(config: ExperimentConfig, X: Tensor, y: Tensor):
    recon_loss_fn = torch.nn.MSELoss()

    if X is None:
        paul15 = sc.datasets.paul15()
        X = torch.tensor(paul15.X).float()
        y = paul15.obs['paul15_clusters'].values
        label_encoder = LabelEncoder()
        target = torch.tensor(label_encoder.fit_transform(y)).to(device)
        config.number_of_centroids = target.max().item() + 1
    else:
        target = y

    # data = config.load_data_fn()
    # X = data.X
    # target = data.y

    # config.number_of_centroids = target.unique().shape[0]

    autoencoder, centroid_pool = train_clusterer(
        X=X,
        y=target,
        latent_dim=config.latent_dimension, 
        n_centroids=config.number_of_centroids,
        config=config.encoder_cluster_config
    )
    latent, recon = autoencoder(X.to(device))
    clusters = centroid_pool(latent)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # v = latent.detach().numpy()
    # a = v[:, 0]
    # b = v[:, 1]
    # plot_clusters(latent, centroid_pool.coords, clusters.argmax(1), y)

    seed_everything(2)

    prims_dataset = generate_prims_dataset(config.neural_exec_config.n_data, 
                                           config.number_of_centroids,
                                           config.latent_dimension)
                                        
    val_dataset = generate_prims_dataset(config.neural_exec_config.n_data, 
                                         config.number_of_centroids,
                                         config.latent_dimension)

    prims_solver = instantiate_prims_solver(prims_dataset, val_dataset, config.neural_exec_config)
    optimizer = torch.optim.Adam(
        combine_params(autoencoder, centroid_pool, prims_solver),
        lr=config.learning_rate
    )

    train_dataset = DataLoader(list(zip(X.to(device), torch.tensor(target).to(device))), batch_size=config.batch_size)

    lowest_loss = 1000
    freeze_model_weights(prims_solver)

    if config.plotting:
        plot_latent(latent, y)
        plot_centers(latent, centroid_pool.coords, y)
        plot_latent_with_fc(latent, centroid_pool.coords, y)

        edges = fc_edge_index(config.number_of_centroids).to(device)
        weights = pairwise_edge_distance(centroid_pool.coords, edges)
        x = torch.zeros(centroid_pool.coords.shape[0], 1, requires_grad=False).to(centroid_pool.coords)
        x[0, 0] = 1
        data = Data(x=x, edge_index=edges, edge_attr=weights)
        data.num_graphs = 1
        mst_logits = prims_solver(data)

        plot_edge_probabilities(latent, centroid_pool.coords, y, mst_logits)
        mst = Graph(nodes=centroid_pool.coords, edge_index=edges, edge_attr=weights,
                                probabilities=mst_logits.softmax(1))
        projection_distances, projected_coords = project_onto_mst(latent, mst)
        plot_single_cell_projection(latent, centroid_pool.coords, y, mst_logits, (-8 * projection_distances).softmax(1), projected_coords)
        plot_most_probable_mst(latent, centroid_pool.coords, y, mst_logits)
        exit()

    edges = fc_edge_index(config.number_of_centroids).to(device)
    for epoch in range(config.n_epochs):
        recon_loss_total = 0.
        mst_loss_total = 0.
        cluster_loss_total = 0.
        loss_total = 0
        for batch_x, batch_y in train_dataset:
            latent, reconstruction = autoencoder(batch_x)

            weights = pairwise_edge_distance(centroid_pool.coords, edges)

            x = torch.zeros(centroid_pool.coords.shape[0], 1, requires_grad=False).to(centroid_pool.coords)
            x[0, 0] = 1
            data = Data(x=x, edge_index=edges, edge_attr=weights)
            data.num_graphs = 1
            predecessor_logits = prims_solver(data)
            # if not sanity_check_neural_exec(prims_solver, prims_dataset, centroid_pool):
            #     breakpoint()
            mst = Graph(nodes=centroid_pool.coords, edge_index=edges, edge_attr=weights,
                        probabilities=predecessor_logits.softmax(1))

            mst_recon_loss = mst_reconstruction_loss_with_backbone(latent, mst, batch_x, autoencoder, 1, config.backbone_distance_coef)
            recon_loss = recon_loss_fn(reconstruction, batch_x)

            cluster_loss = cluster_training_loss_fn(latent, batch_y, centroid_pool)

            # print(f"{mst_recon_loss=}, {recon_loss=}, {cluster_loss=}")

            loss = (config.recon_loss_coef * recon_loss
                  + config.mst_loss_coef * mst_recon_loss
                  + config.cluster_loss_coef * cluster_loss)

            recon_loss_total += (config.recon_loss_coef * recon_loss).item()
            mst_loss_total += (config.mst_loss_coef * mst_recon_loss).item()
            cluster_loss_total += (config.cluster_loss_coef * cluster_loss).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            if config.plotting:
                if epoch % 5 == 0:
                    with torch.no_grad():
                        print('testing?')
                        latent, _ = autoencoder(X.to(device))
                        pred_logits = prims_solver(data)
                        test_results(x.to(device), centroid_pool, paul15.obs['paul15_clusters'], pred_logits, autoencoder)

        with torch.no_grad():
            epoch_loss = loss_total / len(train_dataset)
            recon_loss_total /= len(train_dataset)
            mst_loss_total /= len(train_dataset)
            cluster_loss_total /= len(train_dataset)
            print(f'{epoch_loss=}, {recon_loss_total=}, {mst_loss_total=}, {cluster_loss_total=}')

        lowest_loss = min(epoch_loss, lowest_loss)
        if epoch_loss == lowest_loss:
            if config.save_models:
                torch.save(autoencoder.state_dict(), config.encoder_cluster_config.save_autoencoder_to)
                torch.save(centroid_pool.state_dict(), config.encoder_cluster_config.save_clustering_to)
                torch.save(prims_solver.state_dict(), config.neural_exec_config.save_to)

    return autoencoder, centroid_pool, prims_solver


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        print("training mode")
        config = ExperimentConfig()

    elif sys.argv[1] == 'test':
        print('testing mode')
        neural_exec = NeuralExecutionConfig(load_model=True, train_model=False)
        encoder_clust = EncoderClusterConfig(load_model=True)
        config = ExperimentConfig(save_models=False, plotting=True, neural_exec_config=neural_exec, encoder_cluster_config=encoder_clust)

    else:
        raise ArgumentError("Please define an arg")

    train_narti(config, None, None)
