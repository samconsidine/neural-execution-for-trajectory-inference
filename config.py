from __future__ import annotations
from re import S
from matplotlib.ft2font import LOAD_IGNORE_TRANSFORM

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass


# SHARED PARAMETERS
EXPERIMENT_NAME = "TEST"
LATENT_DIM = 32
NUM_NODES = 10


@dataclass
class DataConfig:
    name:                   str                    = EXPERIMENT_NAME
    datagen_method:         Callable               = lambda: 0
    datagen_parameters:     Dict[str, Any]         = dict


@dataclass
class EncoderClusterConfig:
    name:                   str                    = EXPERIMENT_NAME
    n_epochs:               int                    = 10
    learning_rate:          float                  = 1e-4
    clust_loss_coef:        float                  = 0.
    recon_loss_coef:        float                  = 1.

    load_autoencoder_from:  Optional[str]          = None
    load_clustering_from:   Optional[str]          = None
    save_autoencoder_to:    Optional[str]          = './saved_models/test_autoencoder.pt'
    save_clustering_to:     Optional[str]          = './saved_models/test_clustering.pt'


@dataclass
class NeuralExecutionConfig:
    name:                   str                    = EXPERIMENT_NAME
    latent_dim:             int                    = LATENT_DIM
    n_nodes:                int                    = NUM_NODES
    n_epochs:               int                    = 10
    n_data:                 int                    = 1000
    processor_in_channels:  int                    = 16
    node_features:          int                    = 1
    batch_size:             int                    = 1
    
    load_from:              Optional[str]          = None
    save_to:                Optional[str]          = './saved_models/test_neural_exec.pt'


@dataclass
class ExperimentConfig:
    name:                   str                    = EXPERIMENT_NAME
    latent_dimension:       int                    = LATENT_DIM
    number_of_centroids:    int                    = NUM_NODES
    n_epochs:               int                    = 100
    batch_size:             int                    = 1024
    recon_loss_coef:        float                  = 1.
    mst_loss_coef:          float                  = 100.
    cluster_loss_coef:      float                  = 0.
    learning_rate:          float                  = 3e-4

    data_config:            DataConfig             = DataConfig()
    encoder_cluster_config: EncoderClusterConfig   = EncoderClusterConfig()
    neural_exec_config:     NeuralExecutionConfig  = NeuralExecutionConfig()


default_config = ExperimentConfig()
