from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sklearn import preprocessing
from skmisc import loess
import scipy as sp
import torch
from torch.nn import Module
from torch import Tensor
from expression_matrix_encoder.models.autoencoder import AutoEncoder
from expression_matrix_encoder.models.clustering import CentroidPool
from expression_matrix_encoder.training import train_autoencoder_clusterer
from neural_execution_engine.models.prims_solver import PrimsSolver
from config import ExperimentConfig
from typing import Optional
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
from train import train_narti

import warnings
import numpy as np
import networkx as nx
from utils.graphs import fc_edge_index, pairwise_edge_distance

from utils.metrics import topology, get_GRI
from linalg.projections import project_onto_mst


class NARTI(Module):
    def __init__(self, config: ExperimentConfig, data_dim: int):
        super().__init__()
        self.config = config
        self.autoencoder = AutoEncoder(data_dim, config.latent_dimension)
        self.cluster_network = CentroidPool(config.number_of_centroids, config.latent_dimension)
        self.processor_network = PrimsSolver.from_config(config.neural_exec_config)
        self.mu = self.cluster_network.coords.T  # Does this return a pointer? I hope so

        if config.neural_exec_config.load_model:
            self.processor_network.load_state_dict(torch.load(config.neural_exec_config.load_from))

    def get_data(self, X = None, adata = None, labels = None,
                 covariate = None, cell_names = None, gene_names = None):
        '''Get data for model. 
        
        (1) The user can provide a 2-dim numpy array as the count matrix `X`, either preprocessed or raw. 
        Some extra information `labels`, `cell_names` and `gene_names` (as 1-dim numpy arrays) are optional.

        (2) If the package `scanpy` is installed, then the function can also accept an `annData` input as `adata`. 
        Some extra information `labels`, `cell_names` and `gene_names` are extracted from 
        `adata.obs.cell_types`, `adata.obs_names.values` and `adata.var_names.values`, and
        a 1-dim numpy array `labels` can also be provided if `adata.obs.cell_types` does not exist.

        Covariates can be provided as a 2-dim numpy array.

        Parameters
        ----------
        X : np.array, optional
            \([N, G]\) The counts or expressions data.
        adata : AnnData, optional
            The scanpy object.      
        covariate : np.array, optional
            \([N, s]\) The covariate data.
        labels : np.array, optional
            \([N,]\) The list of labelss for cells.
        cell_names : np.array, optional
            \([N,]\) The list of cell names.
        gene_names : np.array, optional
            \([N,]\) The list of gene names.
        '''
        if adata is None and X is None:
            raise ValueError("Either X or adata should be given!")
        if adata is not None and X is not None:
            warnings.warn("Both X and adata are given, will use adata!")

        self.adata = adata        
        self.raw_X = None if X is None else X.astype(np.float32)
        self.c_score = None if covariate is None else np.array(covariate, np.float32)
        if sp.sparse.issparse(self.raw_X):
            self.raw_X = self.raw_X.toarray()
        self.raw_label_names = None if labels is None else np.array(labels, dtype = str)
        if X is None:
            self.raw_cell_names = None
            self.raw_gene_names = None
        else:            
            self.raw_cell_names = np.array(['c_%d'%i for i in range(self.raw_X.shape[0])]) if \
                cell_names is None else np.array(cell_names, dtype = str)
            self.raw_gene_names = np.array(['g_%d'%i for i in range(self.raw_X.shape[1])]) if \
                gene_names is None else np.array(gene_names, dtype = str)

    def preprocess_data(self, processed: bool = False, dimred: bool = False,
                        K: float = 1e4, gene_num: int = 2000, data_type: str = 'UMI', npc: int = 64):
        ''' Data preprocessing - log-normalization, feature selection, and scaling.                    

        If the inputs are preprocessed by users, then `Gaussian` model will be used and PCA will be performed to reduce the input dimension.
        Otherwise, preprocessing will be performed on `X` following Seurat's routine. 
        If `adata` is provided, the preprocession will be done via `scanpy`.

        Parameters
        ----------
        processed : boolean, optional
            Whether adata has been processed. If `processed=True`, then `Gaussian` model will be used.
        dimred : boolean, optional
            Whether the processed adata is after dimension reduction.
        K : float, optional              
            The constant summing gene expression in each cell up to.
        gene_num : int, optional
            The number of feature to select.
        data_type : str, optional
            'UMI', 'non-UMI' and 'Gaussian', default is 'UMI'. If the input is a processed scanpy object, data type is set to Gaussian.
        npc : int, optional
            The number of PCs to retain.
        '''
        if data_type not in set(['UMI', 'non-UMI', 'Gaussian']):
            raise ValueError("Invalid data type, must be one of 'UMI', 'non-UMI', and 'Gaussian'.")

        if (self.adata is not None) and processed:            
            self.data_type = 'Gaussian'
        else:
            self.data_type = data_type
        print('Using Gaussian likelihood.')

        raw_X = self.raw_X.copy() if self.raw_X is not None else None
        self.X_normalized, self.expression, self.X, self.c_score, \
        self.cell_names, self.gene_names, self.selected_gene_names, \
        self.scale_factor, self.labels, self.label_names, \
        self.le, self.gene_scalar = preprocess(
            self.adata,
            processed,
            dimred,
            raw_X,
            self.c_score,
            self.raw_label_names,
            self.raw_cell_names,
            self.raw_gene_names,
            K, gene_num, self.data_type, npc)
        self.dim_origin = self.X.shape[1]
        self.selected_cell_subset = self.cell_names
        self.selected_cell_subset_id = np.arange(len(self.cell_names))
        self.adata = None

    def evaluate(self, milestone_net, begin_node_true, grouping = None,
                thres: float = 0.5, no_loop: bool = True, cutoff: Optional[float] = None,
                method: str = 'mean', path: Optional[str] = None):
        ''' Evaluate the model.
        TODO:
        - [ ] Figure out meaning of `begin_node`
        - [ ] Find out meaning of all `self`'s
        - [ ] Make sure we are training on every eval dataset
        - [ ] Construct nx graph from trajectory backbone

        Parameters
        ----------
        milestone_net : pd.DataFrame
            The true milestone network. For real data, milestone_net will be a DataFrame of the graph of nodes.
            Eg.

            from|to
            ---|---
            cluster 1 | cluster 1
            cluster 1 | cluster 2

            For synthetic data, milestone_net will be a DataFrame of the (projected)
            positions of cells. The indexes are the orders of cells in the dataset.
            Eg.

            from|to|w
            ---|---|---
            cluster 1 | cluster 1 | 1
            cluster 1 | cluster 2 | 0.1
        begin_node_true : str or int
            The true begin node of the milestone.
        grouping : np.array, optional
            \([N,]\) The labels. For real data, grouping must be provided.

        Returns
        ----------
        res : pd.DataFrame
            The evaluation result.
        '''
        if not hasattr(self, 'le'):
            raise ValueError("No given labels for training.")

        # If the begin_node_true, need to encode it by self.le.
        if isinstance(begin_node_true, str):
            begin_node_true = self.le.transform([begin_node_true])[0]

        # For generated data, grouping information is already in milestone_net
        if 'w' in milestone_net.columns:
            grouping = None

        # If milestone_net is provided, transform them to be numeric.
        if milestone_net is not None:
            milestone_net['from'] = self.le.transform(milestone_net['from'])
            milestone_net['to'] = self.le.transform(milestone_net['to'])

        begin_node_pred = torch.argmin(torch.mean((
            self.z[self.labels==begin_node_true,:,None] -
            self.mu[None,:,:])**2, axis=(0,1))).item()

        # G, edges = self.inference(self.w_tilde, self.pc_x, thres, method, no_loop)
        # G, w, pseudotime = self.infer_trajectory(begin_node_pred, self.label_names, cutoff=cutoff, path=path, is_plot=False)
        G, w, pseudotime = self.infer_trajectory(begin_node_pred)#, self.label_names)

        # 1. Topology
        G_pred = nx.Graph()
        G_pred.add_nodes_from(G.nodes)
        G_pred.add_edges_from(G.edges)
        nx.set_node_attributes(G_pred, False, 'is_init')
        G_pred.nodes[begin_node_pred]['is_init'] = True

        G_true = nx.Graph()
        G_true.add_nodes_from(G.nodes)
        # if 'grouping' is not provided, assume 'milestone_net' contains proportions
        if grouping is None:
            G_true.add_edges_from(list(
                milestone_net[~pd.isna(milestone_net['w'])].groupby(['from', 'to']).count().index))
        # otherwise, 'milestone_net' indicates edges
        else:
            if milestone_net is not None:             
                G_true.add_edges_from(list(
                    milestone_net.groupby(['from', 'to']).count().index))
            grouping = self.le.transform(grouping)
        G_true.remove_edges_from(nx.selfloop_edges(G_true))
        nx.set_node_attributes(G_true, False, 'is_init')
        G_true.nodes[begin_node_true]['is_init'] = True
        res = topology(G_true, G_pred)
            
        # 2. Milestones assignment
        if grouping is None:
            milestones_true = milestone_net['from'].values.copy()
            milestones_true[(milestone_net['from']!=milestone_net['to'])
                           &(milestone_net['w']<0.5)] = milestone_net[(milestone_net['from']!=milestone_net['to'])
                                                                      &(milestone_net['w']<0.5)]['to'].values
        else:
            milestones_true = grouping
        milestones_true = milestones_true[pseudotime!=-1]
        milestones_pred = np.argmax(w[pseudotime!=-1,:], axis=1)
        res['ARI'] = (adjusted_rand_score(milestones_true, milestones_pred) + 1)/2
        
        if grouping is None:
            n_samples = len(milestone_net)
            prop = np.zeros((n_samples,n_samples))
            prop[np.arange(n_samples), milestone_net['to']] = 1-milestone_net['w']
            prop[np.arange(n_samples), milestone_net['from']] = np.where(np.isnan(milestone_net['w']), 1, milestone_net['w'])
            res['GRI'] = get_GRI(prop, w)
        else:
            res['GRI'] = get_GRI(grouping, w)
        
        # 3. Correlation between geodesic distances / Pseudotime
        if no_loop:
            if grouping is None:
                pseudotime_true = milestone_net['from'].values + 1 - milestone_net['w'].values
                pseudotime_true[np.isnan(pseudotime_true)] = milestone_net[pd.isna(milestone_net['w'])]['from'].values            
            else:
                pseudotime_true = - np.ones(len(grouping))
                nx.set_edge_attributes(G_true, values = 1, name = 'weight')
                connected_comps = nx.node_connected_component(G_true, begin_node_true)
                subG = G_true.subgraph(connected_comps)
                milestone_net_true = self.build_milestone_net(subG, begin_node_true)
                if len(milestone_net_true)>0:
                    pseudotime_true[grouping==int(milestone_net_true[0,0])] = 0
                    for i in range(len(milestone_net_true)):
                        pseudotime_true[grouping==int(milestone_net_true[i,1])] = milestone_net_true[i,-1]
            pseudotime_true = pseudotime_true[pseudotime>-1]
            pseudotime_pred = pseudotime[pseudotime>-1]
            res['PDT score'] = (np.corrcoef(pseudotime_true,pseudotime_pred)[0,1]+1)/2
        else:
            res['PDT score'] = np.nan

        # 4. Shape
        # score_cos_theta = 0
        # for (_from,_to) in G.edges:
        #     _z = self.z[(w[:,_from]>0) & (w[:,_to]>0),:]
        #     v_1 = _z - self.mu[:,_from]
        #     v_2 = _z - self.mu[:,_to]
        #     cos_theta = np.sum(v_1*v_2, -1)/(np.linalg.norm(v_1,axis=-1)*np.linalg.norm(v_2,axis=-1)+1e-12)

        #     score_cos_theta += np.sum((1-cos_theta)/2)

        # res['score_cos_theta'] = score_cos_theta/(np.sum(np.sum(w>0, axis=-1)==2)+1e-12)
        return res


    def train(self):
        self.autoencoder, self.centroid_pool, self.processor_network = train_narti(
            self.config, torch.tensor(self.X_normalized), torch.tensor(self.labels))
        self.z = self.autoencoder.encoder(torch.tensor(self.X_normalized))

    def infer_trajectory(self, root):
        x = self.cluster_network.coords.detach().clone()
        fc_edges = fc_edge_index(self.config.number_of_centroids)
        modelin = torch.zeros((self.config.number_of_centroids, self.config.number_of_centroids-1))
        modelin[root, 0] = 1
        data = Data(x=modelin, edge_attr=pairwise_edge_distance(x, fc_edges), edge_index=fc_edges, num_nodes=self.config.number_of_centroids)
        edge_probabilities = self.processor_network(data)
        edges = torch.stack([edge_probabilities.argmax(1), torch.arange(edge_probabilities.shape[0])])

        mst = Graph(edge_index=edges, nodes=x, edge_attr=pairwise_edge_distance(x, edges))
        distances, self.w_tilde, t = project_onto_mst(self.z, mst)
        assignments = distances.argmin(1)  # Assigned to what??
        from_nodes = mst.edge_index[0, assignments]
        to_nodes = mst.edge_index[1, assignments]
        # assignments = torch.stack([from_nodes, to_nodes])
        G = construct_graph_from_edges(edges)
        self.w_tilde = self.w_tilde.gather(1, assignments.unsqueeze(-1).unsqueeze(-1).expand(self.w_tilde.shape[0], 1, 2)).squeeze(1)
        w = self.w_tilde # ?
        pseudotime = self.infer_pseudotime(G, root, edges, assignments, t, from_nodes, to_nodes)

        return G, w.detach().numpy(), pseudotime

    def infer_pseudotime(self, G, root, edges, assignments, t, from_nodes, to_nodes):
        shortest_paths = nx.single_source_shortest_path_length(G, root)
        distances_ = []
        for i in range(G.number_of_nodes()):
            if i in shortest_paths.keys():
                distances_.append(shortest_paths[i])
            else:
                distances_.append(-1)

        distances = torch.tensor(distances_).float()
        t_ = t.gather(1, assignments.unsqueeze(1)).flatten()
        pseudotimes = t_ * distances[from_nodes] + (1-t_) * distances[to_nodes]

        return pseudotimes.detach().numpy()

    def build_milestone_net(self, subgraph, init_node: int):
        '''Build the milestone network.

        Parameters
        ----------
        subgraph : nx.Graph
            The connected component of the backbone given the root vertex.
        init_node : int
            The root vertex.
        
        Returns
        ----------
        df_subgraph : pd.DataFrame 
            The milestone network.
        '''
        if len(subgraph)==1:
            warnings.warn('Singular node.')
            return []
        else:
            # Dijkstra's Algorithm
            unvisited = {node: {'parent':None,
                                'score':np.inf,
                                'distance':np.inf} for node in subgraph.nodes}
            current = init_node
            currentScore = 0
            currentDistance = 0
            unvisited[current]['score'] = currentScore

            milestone_net = []
            while True:
                for neighbour in subgraph.neighbors(current):
                    if neighbour not in unvisited: continue
                    newScore = currentScore + subgraph[current][neighbour]['weight']
                    if unvisited[neighbour]['score'] > newScore:
                        unvisited[neighbour]['score'] = newScore
                        unvisited[neighbour]['parent'] = current
                        unvisited[neighbour]['distance'] = currentDistance+1

                if len(unvisited)<len(subgraph):
                    milestone_net.append([unvisited[current]['parent'],
                                          current,
                                          unvisited[current]['distance']])
                del unvisited[current]
                if not unvisited: break
                current, currentScore, currentDistance = \
                    sorted([(i[0],i[1]['score'],i[1]['distance']) for i in unvisited.items()],
                            key = lambda x: x[1])[0]
            return np.array(milestone_net)


def construct_graph_from_edges(edges):
    return to_networkx(Data(edge_index=edges), to_undirected=False)

import torch
from torch import Tensor
from torch.linalg import vector_norm
from utils.graphs import Graph, pairwise_edge_distance

from typing import Tuple


def assign_to_edge():
    ...

def lineseg_projection(p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Project n points onto the closest point of s line segments in d dimensions.

    Args:
        p1 (Tensor): Shape s x d - The starting points of the line segments.
        p2 (Tensor): Shape s x d - The finishing points of the line segments.
        p3 (Tensor): Shape n x d - The points to be projected onto the segments.

    Returns:
        Tensor: Shape n x s x d - A 3d tensor of the projection points of each
        of the n points onto the s line segments.
    """
    l2 = torch.sum((p1 - p2)**2, dim=-1) # s
    p1p2 = p2 - p1  # s x d
    p1p3 = p3.unsqueeze(1) - p1.unsqueeze(0)  # n x 1 x d - 1 x s x d -> n x s x d
    
    t = torch.sum(p1p2 * p1p3, dim=-1)
    t = (t * (l2 > 0)) / (l2 + 1e-8)
    t = torch.clamp(t, min=0, max=1)
    projection  =  p1 + t.unsqueeze(-1) * p1p2
    # n x s x d =  s x d     n x s x d
    return projection, t


def project_onto_mst(
    latent: Tensor,  # n x d
    tree: Graph
) -> Tuple[Tensor, Tensor]:
    """Project each point in `latent` onto the MST.

    Args:
        latent (Tensor): The embedded cells, output of the encoder.

    Returns:
        Tuple[Tensor, Tensor]: The projection points of each cell onto the MST as
        well as the probability of each projection.
    """

    edges = tree.edge_index

    from_segments = tree.nodes[edges[0]]
    to_segments = tree.nodes[edges[1]]
    projection_coords, t = lineseg_projection(from_segments, to_segments, latent)  # n x s x d

    distances = vector_norm(latent.unsqueeze(1).broadcast_to(projection_coords.shape) - 
                            projection_coords, dim=-1)

    return distances, projection_coords, t


from numba import jit, float32
from numpy import int32

@jit((float32[:,:],), nopython=True, nogil=True)
def _check_expression(A):
    n_rows = A.shape[0]
    out = np.ones(n_rows, dtype=int32)
    for i in range(n_rows):        
        for j in A[i,:]:
            if j>0.0:
                break
        else:
            out[i] = 0
    return out

@jit((float32[:,:],), nopython=True, nogil=True)
def _check_variability(A):
    n_cols = A.shape[1]
    out = np.ones(n_cols, dtype=int32)
    for i in range(n_cols):
        init = A[0, i]
        for j in A[1:, i]:
            if j != init:
                break
        else:
            out[i] = 0
    return out


def log_norm(x, K = 1e4):
    '''Normalize the gene expression counts for each cell by the total expression counts, 
    divide this by a size scale factor, which is determined by total counts and a constant K
    then log-transforms the result.

    Parameters
    ----------
    x : np.array
        \([N, G^{raw}]\) The raw count data.
    K : float, optional
        The normalizing constant.

    Returns
    ----------
    x_normalized : np.array
        \([N, G^{raw}]\) The log-normalized data.
    scale_factor : np.array
        \([N, ]\) The scale factors.
    '''          
    scale_factor = np.sum(x,axis=1, keepdims=True)/K
    x_normalized = np.log(x/scale_factor + 1)
    print('min normailized value: ' + str(np.min(x_normalized)))
    print('max normailized value: ' + str(np.max(x_normalized)))
    return x_normalized, scale_factor


def feature_select(x, gene_num = 2000):
    '''Select highly variable genes (HVGs)
    (See [Stuart *et al*, (2019)](https://www.nature.com/articles/nbt.4096) and its early version [preprint](https://www.biorxiv.org/content/10.1101/460147v1.full.pdf)
    Page 12-13: Data preprocessing - Feature selection for individual datasets).

    Parameters
    ----------
    x : np.array
        \([N, G^{raw}]\) The raw count data.
    gene_num : int, optional
        The number of genes to retain.

    Returns
    ----------
    x : np.array
        \([N, G]\) The count data after gene selection.
    index : np.array
        \([G, ]\) The selected index of genes.
    '''     
    

    n, p = x.shape

    # mean and variance of each gene of the unnormalized data  
    mean, var = np.mean(x, axis=0), np.var(x, axis=0, ddof=1)

    # model log10(var)~log10(mean) by local fitting of polynomials of degree 2
    loess_model = loess.loess(np.log10(mean), np.log10(var), 
                    span = 0.3, degree = 2, family='gaussian'
                    )
    loess_model.fit()
    fitted = loess_model.outputs.fitted_values

    # standardized feature
    breakpoint()
    z = (x - mean)/np.sqrt(10**fitted)

    # clipped the standardized features to remove outliers
    z = np.clip(z, -np.inf, np.sqrt(n))

    # the variance of standardized features across all cells represents a measure of
    # single cell dispersion after controlling for mean expression    
    feature_score = np.sum(z**2, axis=0)/(n-1)
    
    # feature selection
    index = feature_score.argsort()[::-1][0:gene_num]

    return x[:, index], index


def preprocess(adata = None, processed: bool = False, dimred: bool = False, 
            x = None, c = None, label_names = None, raw_cell_names = None, raw_gene_names = None,  
            K: float = 1e4, gene_num: int = 2000, data_type: str = 'UMI', 
            npc: int = 64, random_state=0):
    '''Preprocess count data.

    Parameters
    ----------
    adata : AnnData, optional
        The scanpy object.
    processed : boolean
        Whether adata has been processed.
    dimred : boolean
        Whether the processed adata is after dimension reduction.
    x : np.array, optional
        \([N^{raw}, G^{raw}]\) The raw count matrix.
    c : np.array
        \([N^{raw}, s]\) The covariate matrix.
    label_names : np.array 
        \([N^{raw}, ]\) The true or estimated cell types.
    raw_cell_names : np.array  
        \([N^{raw}, ]\) The names of cells.
    raw_gene_names : np.array
        \([G^{raw}, ]\) The names of genes.
    K : int, optional
        The normalizing constant.
    gene_num : int, optional
        The number of genes to retain.
    data_type : str, optional
        'UMI', 'non-UMI', or 'Gaussian'.
    npc : int, optional
        The number of PCs to retain, only used if `data_type='Gaussian'`.
    random_state : int, optional
        The random state for PCA. With different random states, the resulted PCA scores are slightly different.

    Returns
    ----------
    x_normalized : np.array
        \([N, G]\) The preprocessed matrix.
    expression : np.array
        \([N, G^{raw}]\) The expression matrix after log-normalization and before scaling.
    x : np.array
        \([N, G]\) The raw count matrix after gene selections.
    c : np.array
        \([N, s]\) The covariates.
    cell_names : np.array
        \([N, ]\) The cell names.
    gene_names : np.array
        \([G^{raw}, ]\) The gene names.
    selected_gene_names : 
        \([G, ]\) The selected gene names.
    scale_factor : 
        \([N, ]\) The scale factors.
    labels : np.array
        \([N, ]\) The encoded labels.
    label_names : np.array
        \([N, ]\) The label names.
    le : sklearn.preprocessing.LabelEncoder
        The label encoder.
    gene_scalar : sklearn.preprocessing.StandardScaler
        The gene scaler.
    '''
    # if input is a scanpy data
    if adata is not None:
        import scanpy as sc
        
        # if the input scanpy is processed
        if processed: 
            x_normalized = x = adata.X
            gene_names = adata.var_names.values
            expression = None
            scale_factor = np.ones(x.shape[0])
        # if the input scanpy is not processed
        else: 
            dimred = False
            x = adata.X.copy()
            adata, expression, gene_names, cell_mask, gene_mask, gene_mask2 = _recipe_seurat(adata, gene_num)
            x_normalized = adata.X.copy()
            scale_factor = adata.obs.counts_per_cell.values / 1e4
            x = x[cell_mask,:][:,gene_mask][:,gene_mask2]
            
        if label_names is None:
            try:
                label_names = adata.obs.cell_types
            except:
                if label_names is not None and processed is False:
                    label_names = label_names[cell_mask]
            
        cell_names = adata.obs_names.values
        selected_gene_names = adata.var_names.values
        gene_scalar = None
    
    # if input is a count matrix
    else:
        # remove cells that have no expression
        expressed = _check_expression(x)
        print('Removing %d cells without expression.'%(np.sum(expressed==0)))
        x = x[expressed==1,:]
        if c is not None:
            c = c[expressed==1,:]
        if label_names is None:
            label_names = label_names[expressed==1]        
        
        # remove genes without variability
        variable = _check_variability(x)
        print('Removing %d genes without variability.'%(np.sum(variable==0)))
        x = x[:, variable==1]
        gene_names = raw_gene_names[variable==1]

        # log-normalization
        expression, scale_factor = log_norm(x, K)
        
        # feature selection
        x, index = feature_select(x, gene_num)
        selected_expression = expression[:, index]
        
        # per-gene standardization
        gene_scalar = preprocessing.StandardScaler()
        x_normalized = gene_scalar.fit_transform(selected_expression)
    
        cell_names = raw_cell_names[expressed==1]
        selected_gene_names = gene_names[index]

    if (data_type=='Gaussian') and (dimred is False):
        # use arpack solver and extend precision to get deterministic result
        pca = PCA(n_components = npc, random_state=random_state, svd_solver='arpack')
        x_normalized = x = pca.fit_transform(x_normalized.astype(np.float64)).astype(np.float32)

    if c is not None:
        c_scalar = preprocessing.StandardScaler()
        c = c_scalar.fit_transform(c)

    if label_names is None:
        warnings.warn('No labels for cells!')
        labels = None
        le = None
    else:
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(label_names)
        print('Number of cells in each class: ')
        table = pd.value_counts(label_names)
        table.index = pd.Series(le.transform(table.index).astype(str)) \
            + ' <---> ' + table.index
        table = table.sort_index()
        print(table)
        
    return (x_normalized, expression, x, c, 
        cell_names, gene_names, selected_gene_names, 
        scale_factor, labels, label_names, le, gene_scalar)


def _recipe_seurat(adata, gene_num):
    """
    Normalization and filtering as of Seurat [Satija15]_.
    This uses a particular preprocessing
    """
    import scanpy as sc
    cell_mask = sc.pp.filter_cells(adata, min_genes=200, inplace=False)[0]
    adata = adata[cell_mask,:]
    gene_mask = sc.pp.filter_genes(adata, min_cells=3, inplace=False)[0]
    adata = adata[:,gene_mask]
    gene_names = adata.var_names.values

    sc.pp.normalize_total(adata, target_sum=1e4, key_added='counts_per_cell')
    filter_result = sc.pp.filter_genes_dispersion(
        adata.X, min_mean=0.0125, max_mean=3, min_disp=0.5, log=False, n_top_genes=gene_num)
    
    sc.pp.log1p(adata)
    expression = adata.X.copy()
    adata._inplace_subset_var(filter_result.gene_subset)  # filter genes
    sc.pp.scale(adata, max_value=10)
    return adata, expression, gene_names, cell_mask, gene_mask, filter_result.gene_subset
