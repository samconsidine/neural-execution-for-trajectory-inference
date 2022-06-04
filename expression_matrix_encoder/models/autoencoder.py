from torch.nn import Sequential, Linear, ReLU, Sigmoid, Module
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm


from typing import List


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoEncoder(Module):
    """A standard pytorch autoencoder model.

    Args:
        emb_dim (int): The dimension of the latent space into which to find an encoding.
        input_dimension (int): The dimension of input data.
    """
    def __init__(self, in_dim, emb_dim: int):
        super().__init__()

        self.encoder = Sequential(
            Linear(in_dim, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, emb_dim),
            Sigmoid()
        )
        self.decoder = Sequential(
            Linear(emb_dim, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, in_dim)
        )

        self.to(device)

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)

        return encode, decode


class StackedAE(nn.Module):

    def __init__(self, n_input, encode, latent, decode, binary=True):
        """
        n_input: Int -> Dimensions of the dataset
        encode:  List -> Length of the list determines the 
                number of hidden layers for the encoder.
                Each element of the list determines the 
                size of each hidden layer
        latent: Int -> Size of latent feature space
        decode: List ->  Length of the list determines the 
                number of hidden layers for the decoder.
                Each element of the list determines the 
                size of each hidden layer
        binary: Activates a sigmoid to the output if the dataset
                is binary
        """
        super(StackedAE, self).__init__()
        self.binary = binary
        # encoder        
        self.enc_layers = nn.ModuleList()
        self.enc_Dims = encode.copy()
        self.enc_Dims.insert(0,n_input)
        for idx in range(len(self.enc_Dims)-1):
            self.enc_layers.append(nn.Linear(self.enc_Dims[idx],self.enc_Dims[idx+1]))
        self.z_layer = nn.Linear(self.enc_Dims[-1], latent)

        # decoder
        self.dec_layers = nn.ModuleList()
        self.dec_Dims = decode.copy()
        self.dec_Dims.insert(0,latent)
        for idx in range(len(self.dec_Dims)-1):
            self.dec_layers.append(nn.Linear(self.dec_Dims[idx],self.dec_Dims[idx+1]))
        
        self.fcout = nn.Linear(self.dec_Dims[-1],n_input)
        

    def forward(self, inputs):
        # encoder
        z = inputs
        for layers in self.enc_layers:
            z = layers(z)
            z = F.relu(z)
        z = self.z_layer(z)
        dec = z

        # decoder
        for layers in self.dec_layers:
            dec = layers(dec)
            dec = F.relu(dec)
        x_bar = self.fcout(dec)
        if self.binary:
            x_bar = F.sigmoid(x_bar)

        return x_bar, z

    def fit(self, train_loader, num_epochs=50, lr=1e-3, loss_type='cross-entropy'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        if loss_type=="mse":
            criterion = nn.MSELoss()
        elif loss_type=="cross-entropy":
            criterion = nn.BCELoss()
        
        train_loss = []
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
        for epoch in range(1, num_epochs + 1):
            print("Executing",epoch,"...")
            self.train()
            total_loss = 0
            loop = tqdm(train_loader) 
            for batch_idx, (data, _, _) in enumerate(loop):
                if use_cuda:
                    data = data.cuda()
                optimizer.zero_grad()
                x_bar, _ = self.forward(data)
                loss = criterion(x_bar,data)
                
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch>=50:
                scheduler.step()
            epoch_loss = total_loss / (batch_idx + 1)
            train_loss.append(epoch_loss)
            print("epoch {} loss={:.4f}".format(epoch, total_loss))

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
