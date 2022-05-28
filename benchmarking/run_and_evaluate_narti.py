import numpy as np
from sklearn import preprocessing
import pandas as pd
import math
import h5py
from sklearn.preprocessing import LabelEncoder
import torch
import umap
import os 
import sys
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from benchmarking.NARTI import NARTI
from config import ExperimentConfig
from dataprocessing.dataset import RNASeqDataset
from numba import jit, float32
from numpy import int32

type_dict = {
    # dyno
    'dentate':'UMI', 
    'immune':'UMI', 
    'neonatal':'UMI', 
    'planaria_muscle':'UMI',
    'planaria_full':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human':'non-UMI', 
    'mesoderm':'non-UMI',
    
    # dyngen
    'bifurcating_2':'non-UMI',
    "cycle_1":'non-UMI', 
    "cycle_2":'non-UMI', 
    "cycle_3":'non-UMI',
    "linear_1":'non-UMI', 
    "linear_2":'non-UMI', 
    "linear_3":'non-UMI', 
    "trifurcating_1":'non-UMI', 
    "trifurcating_2":'non-UMI', 
    "bifurcating_1":'non-UMI', 
    "bifurcating_3":'non-UMI', 
    "converging_1":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}
source_dict = {
    'dentate':'dyno', 
    'immune':'dyno', 
    'neonatal':'dyno', 
    'planaria_muscle':'dyno',
    'planaria_full':'dyno',
    'aging':'dyno', 
    'cell_cycle':'dyno',
    'fibroblast':'dyno', 
    'germline':'dyno',    
    'human':'dyno', 
    'mesoderm':'dyno',
    
    'bifurcating_2':'dyngen',
    "cycle_1":'dyngen', 
    "cycle_2":'dyngen', 
    "cycle_3":'dyngen',
    "linear_1":'dyngen', 
    "linear_2":'dyngen', 
    "linear_3":'dyngen', 
    "trifurcating_1":'dyngen', 
    "trifurcating_2":'dyngen', 
    "bifurcating_1":'dyngen', 
    "bifurcating_3":'dyngen', 
    "converging_1":'dyngen',
    
    'linear':'our model',
    'bifurcation':'our model',
    'multifurcating':'our model',
    'tree':'our model',
}


def load_data(path, file_name):  
    '''Load h5df data.
    Parameters
    ----------
    path : str
        The path of the h5 files.
    file_name : str
        The dataset name.
    
    Returns:
    ----------
    data : dict
        The dict containing count, grouping, etc. of the dataset.
    '''     
    data = {}
    
    with h5py.File(os.path.join(path, file_name+'.h5'), 'r') as f:
        data['count'] = np.array(f['count'], dtype=np.float32)
        data['grouping'] = np.array(f['grouping']).astype(str)
        if 'gene_names' in f:
            data['gene_names'] = np.array(f['gene_names']).astype(str)
        else:
            data['gene_names'] = None
        if 'cell_ids' in f:
            data['cell_ids'] = np.array(f['cell_ids']).astype(str)
        else:
            data['cell_ids'] = None
            
        if 'milestone_network' in f:
            if file_name in ['linear','bifurcation','multifurcating','tree',                              
                            "cycle_1", "cycle_2", "cycle_3",
                            "linear_1", "linear_2", "linear_3", 
                            "trifurcating_1", "trifurcating_2", 
                            "bifurcating_1", 'bifurcating_2', "bifurcating_3", 
                            "converging_1"]:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype=str), 
                    columns=['from','to','w']
                ).astype({'w':np.float32})
            else:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype=str), 
                    columns=['from','to']
                )
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype(str)[0]            
        else:
            data['milestone_net'] = None
            data['root_milestone_id'] = None
            
        if file_name in ['mouse_brain', 'mouse_brain_miller']:
            data['grouping'] = np.array(['%02d'%int(i) for i in data['grouping']], dtype=object)
            data['root_milestone_id'] = dict(zip(['mouse_brain', 'mouse_brain_miller'], ['06', '05']))[file_name]
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)
        if file_name in ['mouse_brain_merged']:
            data['grouping'] = np.array(data['grouping'], dtype=object)
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype(str)[0]
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)

    data['type'] = type_dict[file_name]
    if data['type']=='non-UMI':
        scale_factor = np.sum(data['count'],axis=1, keepdims=True)/1e6
        data['count'] = data['count']/scale_factor
    
    return data  

def process_for_narti(data):
    return RNASeqDataset(X=data['count'], y=data['grouping'])

df = pd.DataFrame()
config = ExperimentConfig()

for file_name in type_dict.keys():
    print(file_name)
    no_loop = False if 'cycle' in file_name else True
    is_init = True
    data = load_data('data/',file_name)
    NUM_CLUSTER = len(np.unique(data['grouping']))
    config.number_of_centroids = NUM_CLUSTER
    model = NARTI(config, 2000)
    model.get_data(
        data['count'].copy(), 
        labels = data['grouping'].copy(), 
        gene_names=data['gene_names'])
    model.preprocess_data(                      
        gene_num = 2000,            # (optional) maximum number of influential genes to keep (the default is 2000)
        npc = 64                    # (optional) number of PCs to keep if data_type='Gaussian' (the default is 64)
    )
    num_simulation = 1
    # dim_latent = 8
    for n in range(num_simulation):
        with torch.no_grad():
            z = model.autoencoder.encoder(torch.tensor(model.X_normalized))
        labels = data['grouping'].copy()
        # mu = np.zeros((z.shape[1],NUM_CLUSTER))
        # breakpoint()
        # for i in np.arange(NUM_CLUSTER):
        #     mu[:,i] = np.mean(z[labels==i].detach().numpy(), axis=0)

        # model.init_latent_space(
        #         NUM_CLUSTER,                     # numebr of clusters
        #         cluster_labels=labels,           # (optional) names of the clustering labels for plotting
        #         mu=mu,                           # (optional) initial mean
        #         log_pi=None                    # (optional) initial pi
        # )   

        print('training')
        model.train()

        begin_node_true = model.le.transform([data['root_milestone_id']])[0]
        num_inference = 5
        L = 300
        for i in range(num_inference):
            if data['count'].shape[0]>15000:
                batch_size = 16
            else:
                batch_size = 64
            # model.init_inference(batch_size=batch_size, L=L, refit_dimred=False)

            for method in ['mean','modified_mean','map','modified_map']:
                _df = pd.DataFrame()            
                res = model.evaluate(data['milestone_network'].copy(),
                                    begin_node_true, 
                                    grouping=data['grouping'].copy(), 
                                    method=method,
                                    no_loop=no_loop,
                                    cutoff=None,
                )
                _df = _df.append(pd.DataFrame(res, index=[0]),ignore_index=True)
                _df['method'] = method
                _df['type'] = data['type']
                _df['data'] = file_name
                _df['source'] = source_dict[file_name]
                df = df.append(_df,ignore_index=True)

df = df.groupby('method').mean().sort_values(['data','method']).reset_index(drop=True)
df.to_csv('result/result_VITAE_%s.csv'%(file_name))