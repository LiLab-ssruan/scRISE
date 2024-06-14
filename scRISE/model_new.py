import torch
import torch.nn as nn
import torch.nn.functional as F

from AGE.layers import *


class LinTrans(nn.Module):
    def __init__(self, inputSize, outputSize, layers, dims):
        super(LinTrans, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, 500),  # 经过一个线性层
            nn.ReLU(inplace=True),  # 经过激活函数
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),  # inplace=True表示进行覆盖运算，这样可以节约内存空间
            nn.Linear(200, outputSize)
        )
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):

        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std

        return z_scaled

    def forward(self, x,y):
        y = self.model(y)
        for layer in self.layers:
            out = layer(out)
        out = x+y
        out = self.scale(out)
        out = F.normalize(out)

        return out

'''
定义神经网络结构
'''
class MLP(nn.Module):
    def __init__(self, inputSize, outputSize) -> object:
        super(MLP, self).__init__()
        '''
        定义网络结构
        修改此处可修改网络结构
        '''
        self.model = nn.Sequential(
            nn.Linear(inputSize, 200),  # 经过一个线性层
            nn.ReLU(inplace=True),  # 经过激活函数
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),  # inplace=True表示进行覆盖运算，这样可以节约内存空间
            nn.Linear(200, outputSize),
            nn.ReLU(inplace=True)
        )

        self.dcs = SampleDecoder(act=lambda x: x)
    def scale(self, z):

        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std

        return z_scaled

    "定义前传"

    def forward(self, x):
        x = self.model(x)
        x = self.scale(x)
        x = F.normalize(x)
        return x


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
