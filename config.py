from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass

from dataprocessing.dataset import RNASeqDataset
from dataprocessing.synthetic import load_data_from_file


# SHARED PARAMETERS
EXPERIMENT_NAME = "FullModel"
LATENT_DIM = 32
NUM_NODES = 19


@dataclass
class DataConfig:
    name:                   str                    = EXPERIMENT_NAME
    load_data_fn:  Callable[[None], RNASeqDataset] = lambda: load_data_from_file('bifurcating_3')


@dataclass
class EncoderClusterConfig:
    name:                   str                    = EXPERIMENT_NAME
    n_epochs:               int                    = 200
    learning_rate:          float                  = 1e-4
    clust_loss_coef:        float                  = 3.5
    recon_loss_coef:        float                  = 1.

    load_model:             bool                   = False
    load_autoencoder_from:  Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_autoencoder_{LATENT_DIM}d.pt'
    load_clustering_from:   Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_clustering_{LATENT_DIM}d.pt'
    save_autoencoder_to:    Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_autoencoder_{LATENT_DIM}d-throwaway.pt'
    save_clustering_to:     Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_clustering_{LATENT_DIM}d-throwaway.pt'


@dataclass
class NeuralExecutionConfig:
    name:                   str                    = EXPERIMENT_NAME
    n_nodes:                int                    = NUM_NODES
    emb_dim:                int                    = 32
    n_epochs:               int                    = 200
    n_data:                 int                    = 1000
    processor_in_channels:  int                    = 16
    node_features:          int                    = 1
    batch_size:             int                    = 1
    learning_rate:          float                  = 3e-4
    
    load_model:             bool                   = False
    train_model:            bool                   = True
    load_from:              Optional[str]          = f'./saved_models/{EXPERIMENT_NAME}_{NUM_NODES}_neural_exec_{LATENT_DIM}d-1k_epochs.pt'

    @property
    def save_to(self):
        return f'./saved_models/{self.name}_{self.n_nodes}_neural_exec_{self.emb_dim}_{self.n_epochs}.pt'


@dataclass
class ExperimentConfig:
    name:                   str                    = EXPERIMENT_NAME
    latent_dimension:       int                    = LATENT_DIM
    n_centroids:            int                    = NUM_NODES
    n_epochs:               int                    = 0
    batch_size:             int                    = 128
    recon_loss_coef:        float                  = 2.
    mst_loss_coef:          float                  = 5.
    cluster_loss_coef:      float                  = 5.
    learning_rate:          float                  = 3e-4
    save_models:            bool                   = True
    plotting:               bool                   = False

    backbone_distance_coef: float                  = 1

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

    def test_mode(self):
        self.save_models = False
        self.encoder_cluster_config.load_model = True
        self.neural_exec_config.load_model = True
        self.neural_exec_config.train_model = False
        self.plotting = True

    def test_mode(self):
        ...


default_config = ExperimentConfig()

