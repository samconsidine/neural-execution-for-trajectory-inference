import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

matplotlib.use('TkAgg')


if __name__ == "__main__":
    import scanpy as sc

    data = sc.datasets.paul15()
    plt.spy(data.X)
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('Gene Counts')
    plt.ylabel('Samples')
    plt.show()