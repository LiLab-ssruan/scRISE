import torch
import torch.nn as nn
import torch.nn.functional as F

from AGE.layers import *

class LinTrans(nn.Module):
    def __init__(self, input_feat_dim, hid_dim, n_clusters):
        super(LinTrans, self).__init__()
        self.fc = nn.Sequential(
              nn.Linear(input_feat_dim, hid_dim[0]),
              nn.SiLU(),
              nn.Linear(hid_dim[0], hid_dim[1])
            )
        
        self.dcs = SampleDecoder(act=lambda x: x)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, hid_dim[1]))
        nn.init.xavier_normal_(self.cluster_layer.data)
        self.v =1.0
    

    def forward(self, x):
        out = self.fc(x)
        q = self.soft_assign(out)
        return out, q
    

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.v)
        q = q ** ((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q






