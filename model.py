import torch
from torch import nn, optim
from torch.nn import functional as F

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.SiLU()
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    