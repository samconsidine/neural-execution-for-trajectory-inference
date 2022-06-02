import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import umap
import seaborn as sns

matplotlib.use('TkAgg')


def plot_sparse(data):
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.spy(data.X)
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('Gene Counts')
    plt.ylabel('Samples')
    plt.show()


def plot_umap(data):
    plt.rcParams["figure.figsize"] = (8, 8)
    reducer = umap.UMAP()

    reduced = reducer.fit_transform(data.X)
    xs = reduced[:, 0]
    ys = reduced[:, 1]
    sns.scatterplot(x=xs, y=ys, hue=data.obs['paul15_clusters'], legend=False)

    plt.show()


if __name__ == "__main__":
    import scanpy as sc

    data = sc.datasets.paul15()

    reducer = umap.UMAP()

    reduced = reducer.fit_transform(data.X)

    umap1 = reduced[:, 0]

    idxs = umap1.argsort()

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.spy(data.X[idxs])
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('Gene Counts')
    plt.ylabel('Samples')
    plt.show()

