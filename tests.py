
from config import EncoderClusterConfig, ExperimentConfig, default_config
from expression_matrix_encoder.models import AutoEncoder, CentroidPool
from expression_matrix_encoder.training import train_autoencoder_clusterer
from neural_execution_engine.datagen.prims import generate_prims_dataset, generate_prims_training_instance
from neural_execution_engine.train import instantiate_prims_solver
from utils.graphs import Graph, fc_edge_index, pairwise_edge_distance
from losses import cluster_loss_fn, mst_reconstruction_loss_fn

from typing import Tuple


def test_prims_datagen():

    def test_graph(graph):
        assert (graph.edge_weights < 1.415).test_graphall(), "All distances should be less than sqrt 2"

    def test_training_instance(training_instance):
        ...

    graph = Graph.fc_from_random_geometry(10, 2)
    test_graph(graph)
    training_instance = generate_prims_training_instance(graph)

    test_training_instance(training_instance)

def test_neural_clustering():
    ...


def test_cluster_loss_fn():
    ...


def test_mst_recon_loss():
    ...


def test_mst_projection():
    ...


def run_tests():
    test_prims_datagen()


if __name__ == "__main__":
    run_tests()