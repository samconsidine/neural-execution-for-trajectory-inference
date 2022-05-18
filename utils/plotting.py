import matplotlib.pyplot as plt

import numpy as np


def plot_mst(logits, X, centroids, epoch):
    to_nodes = logits.argmax(1).detach().numpy()
    from_nodes = np.arange(len(to_nodes))
    centroids = centroids.detach().numpy()
    for i in range(len(centroids)):
        xs = [centroids[from_nodes[i]][0], centroids[to_nodes[i]][0]]
        ys = [centroids[from_nodes[i]][1], centroids[to_nodes[i]][1]]
        plt.plot(xs, ys, 'ro-')
