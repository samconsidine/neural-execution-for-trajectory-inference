import torch
from torch.nn import Module, Parameter

import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from utils.torch import init_weights, cluster_acc
from typing import Optional
from sklearn.cluster import KMeans
from expression_matrix_encoder.models import StackedAE 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralisedClustering(Module):
    def __init__(self):
        super().__init__()

class CentroidPool(NeuralisedClustering):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.n_clusts = n_clusts
        self.n_dims = n_dims
        self.coords = Parameter(torch.rand(n_clusts, n_dims, requires_grad=True))
        self.to(device)

    def forward(self, X):
        return torch.cdist(X, self.coords)

class KMadness(NeuralisedClustering):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = Parameter(torch.rand(size=(n_clusts, n_dims)))
        self.to(device)

    @property
    def w(self):
        return f1(self.coords).T

    @property
    def b(self):
        return f2(sqnorm(self.coords))

    def fc(self, h):
        return (-(-h).topk(2, 1)[0])[:, 1, :]

    def forward(self, x):
        h = torch.matmul(self.w, x.T).T + self.b
        cluster_assignments = self.fc(h) > 0
        return cluster_assignments

    def closest_centroids(self, assignments):
        return self.coords[assignments.max(-1)[1]]


def f1(tensor):
    tensor = tensor.permute(1, 0)
    return torch.sub(tensor.unsqueeze(dim=2), tensor.unsqueeze(dim=1))


def f2(tensor):
    return tensor.unsqueeze(0).T - tensor


def sqnorm(tensor):
    return torch.linalg.norm(tensor, dim=1)**2


class IDEC(nn.Module):
    def __init__(self, n_input, encode, latent, 
                 decode, n_clusters, alpha=1, binary=True,
                 coords: Optional[torch.Tensor] = None):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.n_clusters = n_clusters
        self.ae = StackedAE(n_input, encode,latent, decode,binary=binary)
        self.y_pred_last = None
        self.convergence_iter = 0
        self.prop = None
        self.binary = binary

        if coords is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                latent,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = coords
        self.coords = nn.Parameter(initial_cluster_centers)

    def pretrain(self, train_loader, path, lr=0.001, num_epochs=50, cuda=False):
        self.ae.apply(init_weights)
        if self.binary == True:
            criterion = "cross-entropy"
        else: criterion = 'mse'
        self.ae.fit(train_loader, lr=lr, num_epochs=num_epochs, loss_type=criterion)
        self.ae.save_model(path)

    def target_distribution(self,batch_q):
        numerator = (batch_q**2) / torch.sum(batch_q,0)
        return numerator / torch.sum(numerator,dim=1, keepdim=True)

    def forward(self,batch):
        """
        Compute soft assignment for an input, returning the input's assignments
        for each cluster.
        batch: [batch_size, input_dimensions]
        output: [batch_size, number_of_clusters]
        """

        x_bar, z = self.ae.forward(batch)
        Frobenius_squared = torch.sum((z.unsqueeze(1)-self.coords)**2,2)
        numerator = 1.0 / (1.0+(Frobenius_squared/self.alpha))
        power = float(self.alpha+1)/2
        numerator = numerator**power
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return x_bar, q, z        

    def initialize_kmeans(self,valid_loader):
        print("=====Initializing KMeans Centers=======")
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        self.eval()
        data = []
        loop = tqdm(valid_loader)
        for batch_idx, (inputs,_,_) in enumerate(loop):
            if use_cuda:
                inputs = inputs.cuda() 
            _ , latent = self.ae.forward(inputs)
            data.append(latent.data.cpu().numpy())
        data = np.concatenate(data)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(data)
        self.y_pred_last = y_pred
        
        self.coords.data = torch.tensor(kmeans.cluster_centers_)

    def update_target_distribution(self,valid_loader,tol):
        data = []
        labels = []
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        for batch_idx, (inputs, tar,_) in enumerate(valid_loader):
            if use_cuda:
                inputs = inputs.cuda()
            _, tmp_q, _ = self.forward(inputs)
            data.append(tmp_q.data)
            labels.append(tar.cpu().numpy())
        tmp_q = torch.cat(data)
        labels = np.concatenate(labels)
        self.prop = self.target_distribution(tmp_q)

        #evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)

        labels_changed = np.sum(y_pred != self.y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        self.y_pred_last = y_pred

        if labels_changed < tol:
            self.convergence_iter+=1
        else:
            self.convergence_iter = 0 
        return labels_changed, cluster_acc(labels,y_pred)[0]

    def fit(self,valid_loader,train_loader,path,num_epochs=100,lr=1e-4,update_target=1,gama=0.1,tolerance=1e-4,loss_type="cross-entropy"):
        
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            
        if loss_type=="mse":
            criterion = nn.MSELoss()
        elif loss_type=="cross-entropy":
            criterion = nn.BCELoss()
        elif loss_type == 'Frobenius':
            criterion = self.Frobenius_norm()

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        self.train()
        train_loss = []

        for epoch in range(num_epochs):
            print("Executing IDEC",epoch,"...")
            if epoch % update_target == 0:
                labels_changed, acc = self.update_target_distribution(valid_loader,tol=tolerance)
            if self.convergence_iter >=10:
                print('percentage of labels changed {:.4f}'.format(labels_changed), '< tol',tolerance)
                print('Reached Convergence threshold. Stopping training.')
                break
            loop = tqdm(train_loader)
            total_loss = 0
            for batch_idx, (x, _, idx) in enumerate(loop):
                if use_cuda:
                    x = x.cuda()
                    idx = idx.cuda()

                x_bar, q, _ = self.forward(x)
                reconstr_loss = criterion(x_bar,x)
                kl_loss = F.kl_div(q.log(), self.prop[idx])
                loss = gama * kl_loss + reconstr_loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss = total_loss / (batch_idx + 1)
            train_loss.append(epoch_loss)
            print("epoch {} loss={:.4f}, accuracy={:.5f}".format(epoch,epoch_loss,acc))
            scheduler.step()
            self.save_model(path)

    def Frobenius_norm(self,approximation=None, input_matrix=None):
        #Minimizing the Frobenius norm
        Frobenius = 0.5*(torch.norm(input_matrix-approximation)**2)
        return Frobenius

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
