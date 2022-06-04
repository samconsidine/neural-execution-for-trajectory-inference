from math import comb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import inspect
import matplotlib

import numpy as np

from utils.graphs import fc_edge_index

# sns.set_theme(style="whitegrid")
sns.set_context('paper')
plt.rcParams['figure.figsize'] = 8, 6
plt.rc("axes.spines", top=False, right=False)


def plot_mst(logits, centroids):
    to_nodes = logits.argmax(1).detach().numpy()
    from_nodes = np.arange(len(to_nodes))
    for i in range(len(centroids)):
        xs = [centroids[from_nodes[i]][0], centroids[to_nodes[i]][0]]
        ys = [centroids[from_nodes[i]][1], centroids[to_nodes[i]][1]]
        plt.plot(xs, ys, 'ro-')


def plot_fc(centroids):
    edges = fc_edge_index(centroids.shape[0])
    to_nodes = edges[0]
    from_nodes = edges[1]
    for i in range(len(to_nodes)):
        xs = [centroids[from_nodes[i]][0], centroids[to_nodes[i]][0]]
        ys = [centroids[from_nodes[i]][1], centroids[to_nodes[i]][1]]
        plt.plot(xs, ys, 'ro:', alpha=0.6)


def plot_probab_edges(centroids, logits):
    edges = fc_edge_index(centroids.shape[0])
    probabs = logits.softmax(1).numpy().flatten()
    to_nodes = edges[0]
    from_nodes = edges[1]
    for i in range(len(to_nodes)):
        xs = [centroids[from_nodes[i]][0], centroids[to_nodes[i]][0]]
        ys = [centroids[from_nodes[i]][1], centroids[to_nodes[i]][1]]
        plt.plot(xs, ys, 'ro-', alpha=probabs[i])



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
        plot_mst(tree_logits, coords)
        plt.tight_layout()
        # plt.savefig(f'plots/mst/{sys.argv[1]}/{epoch}.png')
        plt.clf()
        # plt.clf()
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()

def plot_latent_with_fc(latent, coords, labels):
    outs = latent.clone()
    with torch.no_grad():
        if outs.shape[1] > 2:
            pca = PCA(2)
            outs = pca.fit_transform(outs)
            coords = pca.transform(coords)
        xs = outs[:, 0]
        ys = outs[:, 1]
        cx = coords[:, 0]
        cy = coords[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=labels, legend=False, alpha=0.7)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black', s=110)
        plot_fc(coords)
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()

def plot_edge_probabilities(latent, coords, labels, mst_logits):
    outs = latent.clone()
    with torch.no_grad():
        if outs.shape[1] > 2:
            pca = pca(2)
            outs = pca.fit_transform(outs)
            coords = pca.transform(coords)
        xs = outs[:, 0]
        ys = outs[:, 1]
        cx = coords[:, 0]
        cy = coords[:, 1]

        sns.scatterplot(x=xs, y=ys, color='lightgray', legend=False, alpha=0.7)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black', s=110)
        plot_probab_edges(coords, mst_logits)
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()

    
def plot_most_probable_mst(latent, coords, labels, mst_logits):
    outs = latent.clone()
    with torch.no_grad():
        if outs.shape[1] > 2:
            pca = pca(2)
            outs = pca.fit_transform(outs)
            coords = pca.transform(coords)
        xs = outs[:, 0]
        ys = outs[:, 1]
        cx = coords[:, 0]
        cy = coords[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=labels, legend=False, alpha=0.7)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black', s=110)
        plot_mst(mst_logits, coords)
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()
    
    
def plot_single_cell_projection(latent, coords, labels, mst_logits, projection_probabilities, projection_coords, cellidx=5):
    outs = latent.clone()
    palette = ["lightgrey", "red"]
    with torch.no_grad():
        if outs.shape[1] > 2:
            pca = PCA(2)
            outs = pca.fit_transform(outs)
            coords = pca.transform(coords)
        xs = outs[:, 0]
        ys = outs[:, 1]
        cx = coords[:, 0]
        cy = coords[:, 1]

        colors, alphas = one_color_palette(len(outs), cellidx, 'indigo')
        sns.scatterplot(x=xs, y=ys, hue=colors, palette=palette, legend=False, alpha=alphas)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black', s=110)
        plot_probab_edges(coords, mst_logits)

        cell_coords = outs[cellidx]
        combined_probabs = projection_probabilities[cellidx] * mst_logits.softmax(1).flatten()
        combined_probabs_norm = (combined_probabs - combined_probabs.min()) / (combined_probabs.max() - combined_probabs.min())
        for i, proj_coords in enumerate(projection_coords[cellidx]):
            x = [cell_coords[0], proj_coords[0]]
            y = [cell_coords[1], proj_coords[1]]
            plt.plot(x, y, 'bo-', alpha=combined_probabs_norm[i].item())

        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()

def plot_centers(latent, centers, labels):
    with torch.no_grad():
        if latent.shape[1] > 2:
            pca = PCA(2)
            latent = pca.fit_transform(latent)
            centers = pca.transform(centers)
        xs = latent[:, 0]
        ys = latent[:, 1]
        cx = centers[:, 0]
        cy = centers[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=labels, legend=False, alpha=0.7)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black', s=110)
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()



def plot_latent(latent, labels):
    with torch.no_grad():
        if latent.shape[1] > 2:
            pca = PCA(2)
            latent = pca.fit_transform(latent)
        xs = latent[:, 0]
        ys = latent[:, 1]

        sns.scatterplot(x=xs, y=ys, hue=labels, legend=False, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()


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
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()

        sns.scatterplot(x=xs, y=ys, hue=labels, legend=False)
        sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
        plt.tight_layout()
        plt.savefig(f'/home/sam/thesis/report/figures/{inspect.stack()[0][3]}.png')
        plt.clf()


def one_color_palette(length, number, color):
    colors = ['lightgrey' for _ in range(length)]
    # palette=sns.color_palette(colors, length)
    colors[number] = color
    # print(palette[0], palette[1])
    alphas = [0.7 for _ in range(length)]
    alphas[number] = 1.0

    return colors, alphas