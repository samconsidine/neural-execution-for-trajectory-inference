from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass

from dataprocessing.dataset import RNASeqDataset
from dataprocessing.synthetic import load_data_from_file


# SHARED PARAMETERS
EXPERIMENT_NAME = "Model1"
LATENT_DIM = 32
NUM_NODES = 10


@dataclass
class DataConfig:
    name:                   str                    = EXPERIMENT_NAME
    load_data_fn:  Callable[[None], RNASeqDataset] = lambda: load_data_from_file('bifurcating_3')


@dataclass
class EncoderClusterConfig:
    name:                   str                    = EXPERIMENT_NAME
    n_epochs:               int                    = 100
    learning_rate:          float                  = 1e-4
    clust_loss_coef:        float                  = 10.
    recon_loss_coef:        float                  = 1.

    load_model:             bool                   = True
    load_autoencoder_from:  Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_autoencoder_{LATENT_DIM}d.pt'
    load_clustering_from:   Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_clustering_{LATENT_DIM}d.pt'
    save_autoencoder_to:    Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_autoencoder_{LATENT_DIM}d.pt'
    save_clustering_to:     Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_clustering_{LATENT_DIM}d.pt'


@dataclass
class NeuralExecutionConfig:
    name:                   str                    = EXPERIMENT_NAME
    latent_dim:             int                    = LATENT_DIM
    n_nodes:                int                    = NUM_NODES
    n_epochs:               int                    = 300
    n_data:                 int                    = 1000
    processor_in_channels:  int                    = 16
    node_features:          int                    = 1
    batch_size:             int                    = 1
    learning_rate:          float                  = 3e-4
    
    load_model:             bool                   = True
    train_model:            bool                   = False
    load_from:              Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_neural_exec_{LATENT_DIM}d.pt'
    save_to:                Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_neural_exec_{LATENT_DIM}d.pt'


@dataclass
class ExperimentConfig:
    name:                   str                    = EXPERIMENT_NAME
    latent_dimension:       int                    = LATENT_DIM
    n_centroids:            int                    = NUM_NODES
    n_epochs:               int                    = 1000
    batch_size:             int                    = 128
    recon_loss_coef:        float                  = 1.
    mst_loss_coef:          float                  = 1.
    cluster_loss_coef:      float                  = 1.
    learning_rate:          float                  = 3e-5
    save_models:            bool                   = False

    data_config:            DataConfig             = DataConfig()
    encoder_cluster_config: EncoderClusterConfig   = EncoderClusterConfig()
    neural_exec_config:     NeuralExecutionConfig  = NeuralExecutionConfig()

    @property
    def number_of_centroids(self):
        return self.n_centroids

    @number_of_centroids.setter
    def number_of_centroids(self, val):
        self.n_centroids = val
        self.neural_exec_config.n_nodes = val


default_config = ExperimentConfig()
