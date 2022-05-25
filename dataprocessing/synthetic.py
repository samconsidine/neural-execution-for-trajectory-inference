import os
import h5py
import requests
import sys
from dataprocessing.load import load_data
from dataprocessing.dataset import RNASeqDataset
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_synthetic_data(dataset_name):
    if not os.path.exists(f'data/tmp/{dataset_name}.h5'):
        download_bifurcating_data(dataset_name)

    with h5py.File(f'data/tmp/{dataset_name}.h5', 'r') as f:
        print(f)


def download_bifurcating_data(dataset_name):
    res = requests.get(f'https://github.com/jaydu1/VITAE/raw/master/data/{dataset_name}.h5')

    if not res.status_code == 200:
        raise requests.exceptions.HTTPError(f'File not found {dataset_name}')

    if not os.path.exists('data/tmp/'):
        os.makedirs('./data/tmp')

    with h5py.File(f'data/tmp/{dataset_name}.h5', 'w') as f:
        f.create_dataset(name=dataset_name, data=res.content)


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


def load_data_from_file(data_name):
    from sklearn.preprocessing import LabelEncoder
    data = load_data('data/', data_name)
    milestone_net = data['milestone_network']    
    le = LabelEncoder()
    grouping = le.fit_transform(data['grouping'])
    begin_node_true = le.transform([data['root_milestone_id']])[0]
    if 'w' in milestone_net.columns:
        grouping = None
    if milestone_net is not None:
        milestone_net['from'] = le.transform(milestone_net['from'])
        milestone_net['to'] = le.transform(milestone_net['to'])  

    milestone_net['label'] = np.where(milestone_net['w'] > 0.5, milestone_net['to'], milestone_net['from'])
    x = data['count']
    expression, scale_factor = log_norm(x)
    
    # feature selection
    selected_expression = expression
    
    # per-gene standardization
    gene_scalar = StandardScaler()
    x_normalized = gene_scalar.fit_transform(selected_expression)
    return RNASeqDataset(x_normalized, milestone_net['label'].values, milestone_net)


if __name__=="__main__":
    data = load_data_from_file(sys.argv[1])
    breakpoint()