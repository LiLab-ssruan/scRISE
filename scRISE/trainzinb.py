import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from pre_processing import load_data_origin_data
from modelzinb import ZINBAE,ZINBLoss
import torch
torch.cuda.empty_cache()
from torch import nn, optim
import pandas as pd
from sklearn.cluster import KMeans
from AGE.train import gae_for, smooth_for, Graph_Build,grid_search_loss_fn
from graph_function import *
from sklearn.metrics.cluster import *
import scanpy as sc
import random
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

# {'k': 15, 'run': 4, 'w1': 0.8, 'w2': 0.8} 0.4911859561276497
def cluster_acc(y_true, y_pred, name=None, path=None):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, 1



def eva(z,y_true, y_pred, epoch=0, pp=True, name=None, path=None):
    acc, f1 = cluster_acc(y_true, y_pred, name=name, path=path)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    #nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = ari_score(y_true, y_pred)
    silhouette = silhouette_score(z, y_pred)
    if pp:
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),'silhouette {:.4f}'.format(silhouette))
    return acc, nmi, ari, silhouette


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def normalize(adata, copy=True, highly_genes=2000, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    count_X = adata.X
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
    count_X = count_X[:, high_variable]
    return adata, count_X

class SquareRegularizeLoss(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    ##input:samples*features
    ##regularized loss:1/n*(|1-(x1^2+...+xn^2)|^p)
    def forward(self, input):
        feature_num = input.size(1)
        input = torch.pow(input, 2).sum(dim=1)
        if self.p == 1:
            loss = torch.abs(1 - input)
        else:
            loss = torch.pow(1 - input, self.p)
        loss = loss.mean() / feature_num

        return loss

parser = argparse.ArgumentParser(description='Main entrance of AGE')
#Preprocess-------------------------------------------------------------------------------------------
parser.add_argument('--data', type=str, default='Deng',
                    help='Baron_mouse.csv')
parser.add_argument('--delim', type=str, default='comma',
                    help='File delim type, comma or space: default(comma)')
parser.add_argument('--highly_genes', type=int, default=500)
#AE----------------------------------------------------------------------------------------------------
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 12800)')
parser.add_argument('--Regu-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train in Autoencoder initially (default: 500)')
parser.add_argument('--run', type=int, default=4, metavar='N',
                    help='(default: 5)')
parser.add_argument('--reduction', type=str, default='mean',
                    help='reduction type: mean/sum, default(sum)')
# Build cell graph-------------------------------------------------------------------------------------
parser.add_argument('--k', type=int, default=15,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--graph', default="knn",
                    help='the type of KNN graph')
# Clustering related-------------------------------------------------------------------------------------
parser.add_argument('--n-clusters', type=int, default=10, help='number of clusters')
# AGE related---------------------------------------------------------------------------------------------
parser.add_argument('--gnnlayers', type=int, default=3,
                    help='Number of gnn layers 61')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.default:500')
parser.add_argument('--hidden_dim1', type=int, default=256,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--hidden_dim2', type=int, default=32,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.0035,
                    help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.3, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold e')
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
parser.add_argument('--seed', type=int, default=1111, help='seed')
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

setup_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('Using device:'+str(device))
#args.datasetDir = "count/splatter/"+args.data+"/"
args.datasetDir = "count/"+args.data+"/"
print(args.datasetDir)

start_time = time.time()
expressionFilename = args.datasetDir + args.data + "_car.csv"

if not os.path.exists(expressionFilename):
    print('Dataset ' + expressionFilename + ' not exists!')

print('Input scRNA data in CSV format is validated, start reading...')

df = pd.read_csv(expressionFilename, index_col=0)
dataset = load_data_origin_data(expressionFilename)
x= dataset.x
y = dataset.y
args.n_clusters = int(max(y) - min(y) + 1)
print(args.n_clusters )
print('Data loaded, start filtering...')


adata = sc.AnnData(x)
adata, count_X = normalize(adata, copy=True, highly_genes=args.highly_genes, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True)
data = adata.X
sf= adata.obs.size_factors
print("Preprocessing Done. Total Running Time: %s seconds" %
      (time.time() - start_time))


def train(epoch, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, (data, x_raw, sf) in enumerate(dataloader):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        x_raw = x_raw.type(torch.FloatTensor)
        x_raw = x_raw.to(device)
        sf = sf.type(torch.FloatTensor)
        sf = sf.to(device)
        optimizer.zero_grad()
        recon_batch, z, mean, disp, pi = model(data)
        # RegularizeLoss = SquareRegularizeLoss()
        # re_loss = RegularizeLoss(z)
        zinb_loss = ZINBLoss()
        zinb_loss = zinb_loss(x=x_raw, mean=mean, disp=disp, pi=pi,
                                   scale_factor=sf)
        loss = zinb_loss 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx == 0:
            recon_batch_all = recon_batch
            data_all = data
            z_all = z
        else:
            recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
            data_all = torch.cat((data_all, data), 0)
            z_all = torch.cat((z_all, z), 0)


        recon_batch_all = recon_batch_all
        data_all = data_all
        z_all = z_all

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))

    return recon_batch_all, data_all, z_all


if __name__ == "__main__":
    for i in range(1, args.run + 1):
        print(i)
        print(data.shape)
        adj = Graph_Build(data, args.k)
        data = smooth_for(data, adj, args.gnnlayers)  
        dataset = TensorDataset(torch.Tensor(data), torch.Tensor(count_X), torch.Tensor(sf))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model = AE1(dim=data.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(1, args.Regu_epochs + 1):
            recon, original, z = train(epoch, dataloader)
        Z = z.detach().cpu().numpy()
        R = recon.detach().cpu().numpy()
        data = R

    sm_fea = Z
    zOut, y_prep = gae_for(features=sm_fea, adj=adj, y_true=y, n_clusters=args.n_clusters, lr=args.lr,
                           upth_ed=args.upth_ed, upth_st=args.upth_st, lowth_ed=args.lowth_ed, lowth_st=args.lowth_st,
                           epochs=args.epochs, upd=args.upd, bs=args.bs, w1=1, w2=1, device=device)


    print("Clustering Done. Total Running Time: %s seconds" %
          (time.time() - start_time))
    embedding_df = pd.DataFrame(zOut)
    embedding_df.to_csv(args.datasetDir + args.data + '_embedding.csv')
    results_df = pd.DataFrame(y_prep, columns=["Celltype"])
    results_df.to_csv(args.datasetDir + args.data + '_results.csv')
    y_true = y
    acc, nmi, ari, silhouette = eva(zOut, y_true, y_prep)

    clustering = KMeans(n_clusters=args.n_clusters,
                        random_state=0).fit(sm_fea)
    listResult = clustering.predict(sm_fea)
    print('Kmeans zOut')
    acc, nmi, ari, silhouette = eva(sm_fea, y_true, listResult)

    clustering = KMeans(n_clusters=args.n_clusters,
                        random_state=0).fit(adata.X)
    listResult = clustering.predict(adata.X)
    print('Kmeans raw')
    acc, nmi, ari, silhouette = eva(adata.X, y_true, listResult)

