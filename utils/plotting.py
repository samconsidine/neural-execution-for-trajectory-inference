import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import numpy as np


def plot_mst(logits, X, centroids, epoch):
    to_nodes = logits.argmax(1).detach().numpy()
    from_nodes = np.arange(len(to_nodes))
    for i in range(len(centroids)):
        xs = [centroids[from_nodes[i]][0], centroids[to_nodes[i]][0]]
        ys = [centroids[from_nodes[i]][1], centroids[to_nodes[i]][1]]
        plt.plot(xs, ys, 'ro-')


def test_results(inputs, pool, labels, tree_logits, ae):
    with torch.no_grad():

        outs = ae.encoder(inputs).detach().numpy()
        coords = pool.coords
        if outs.shape[1] > 2:
            pca = PCA(2)
            outs = pca.fit_transform(outs)
            coords = pca.transform(coords)
        xs = outs[:, 0]
        ys = outs[:, 1]
        cx = coords[:, 0]
        cy = coords[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=labels.values, legend=False)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
        print(xs)
        plot_mst(tree_logits, outs, coords, 1)
        # plt.savefig(f'plots/mst/{sys.argv[1]}/{epoch}.png')
        # plt.clf()
        plt.show()


def plot_clusters(latent, centers, assignments, labels):
    with torch.no_grad():
        if latent.shape[1] > 2:
            pca = PCA(2)
            latent = pca.fit_transform(latent)
            centers = pca.transform(centers)
        xs = latent[:, 0]
        ys = latent[:, 1]
        cx = centers[:, 0]
        cy = centers[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=assignments, legend=False)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
        plt.show()

        sns.scatterplot(x=xs, y=ys, hue=labels, legend=False)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
        plt.show()