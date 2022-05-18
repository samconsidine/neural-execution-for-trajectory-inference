from __future__ import annotations

from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Callable, Dict, Any
from dataclasses import dataclass


EXPERIMENT_NAME = "TEST"
LATENT_DIM = 2
NUM_NODES = 10


@dataclass
class DataConfig:
    datagen_method:         Callable                = lambda: 0
    datagen_parameters:     Dict[str, Any]          =  dict


@dataclass
class EncoderClusterConfig:
    n_epochs:               int                     = 1
    layer_sizes:            List[int]               = list
    recon_loss_coef:        float                   = 1000
    cluster_loss_coef:      float                   = 1
    learning_rate:          float                   = 1e-4
    clust_loss_coef:        float                   = 1e-4
    recon_loss_coef:        float                   = 1.


@dataclass
class NeuralExecutionConfig:
    latent_dim:             int                     = LATENT_DIM
    n_nodes:                int                     = NUM_NODES
    n_epochs:               int                     = 1
    n_data:                 int                     = 10
    processor_in_channels:  int                     = 16
    node_features:          int                     = 1
    batch_size:             int                     = 1
    

@dataclass
class ProjectionConfig:
    ...


@dataclass
class ExperimentConfig:
    name:                   str                     = EXPERIMENT_NAME
    latent_dimension:       int                     = LATENT_DIM
    number_of_centroids:    int                     = NUM_NODES
    n_epochs:               int                     = 100
    batch_size:             int                     = 1024
    recon_loss_coef:        float                   = 1.
    mst_loss_coef:          float                   = 10.
    cluster_loss_coef:      float                   = .1

    data_config:            DataConfig              = DataConfig()
    encoder_cluster_config: EncoderClusterConfig    = EncoderClusterConfig()
    neural_exec_config:     NeuralExecutionConfig   = NeuralExecutionConfig()
    projection_config:      ProjectionConfig        = ProjectionConfig()


default_config = ExperimentConfig()
