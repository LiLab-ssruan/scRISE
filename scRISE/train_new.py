from __future__ import division
from __future__ import print_function
import os, sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
import argparse
import time

from torch import optim
from AGE.model_new import LinTrans, LogReg,MLP
from AGE.optimizer import loss_function
from AGE.utils import *
from sklearn.cluster import SpectralClustering, KMeans
from AGE.clustering_metric import clustering_metrics
from AGE.autoencoders import autoen, ClusteringLayer
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling

from numpy.linalg import norm


def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth


class AGE(torch):
    def __init__(self,X,features, adj, args, device):
        super(AGE, self).__init__()
        self.X = X
        self.adj = adj
        self.n_sample = features.shape[0] # n_nodes
        self.in_dim = features.shape[1]# feat_dim
        self.dim = [self.in_dim] + args.dims
        self.layers = args.linlayers
        self.gnnlayers = args.gnnlayers
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_orig = adj

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train
        n = adj.shape[0]
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

        # Some preprocessing
        adj_norm_s = preprocess_graph(adj, self.gnnlayers, norm='sym', renorm=True)
        adj_1st = (adj + sp.eye(n)).toarray()
        adj_label = torch.FloatTensor(adj_1st)

        latend= MLP(X.shape[1],128)
        model = LinTrans(self.layers, self.dim)

        pos_num = len(adj.indices)
        neg_num = self.n_sample * self.n_sample - pos_num
        # Thresholds Update.
        up_eta = (args.upth_ed - args.upth_st) / (args.epochs / args.upd)
        low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs / args.upd)



    def smooth(features,adj_norm_s):
        sm_fea_s = sp.csr_matrix(features).toarray()
        print('Laplacian Smoothing...')
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        return sm_fea_s

    def train(self):

        pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num,
                                               upth,
                                               lowth=update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)
        bs = min(args.bs, len(pos_inds))
        length = len(pos_inds)


def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    data_norm = data_norm.astype('float32')
    if cellwise_norm:
        libs = data.sum(axis=1)
        print(len(libs))
        norm_factor = np.diag(np.median(libs) / libs)
        print(norm_factor)
        data_norm = np.dot(norm_factor, data_norm)

    if log1p:
        data_norm = np.log2(data_norm + 1.)
    return data_norm

def find_hv_genes(X, top=2000):
    ngene = X.shape[1]
    CV = []
    for i in range(ngene):
        x = X[:, i]
        x = x[x != 0]
        mu = np.mean(x)
        var = np.var(x)
        CV.append(var / mu)
    CV = np.array(CV)
    rank = CV.argsort()
    hv_genes = np.arange(len(CV))[rank[:-1 * top - 1:-1]]
    return hv_genes

data = pd.read_csv(expressionFilename,index_col=0)

data = data.loc[~(data==0).all(axis=1)]
data = data.values.astype('float32')

data = take_norm(data.transpose(), cellwise_norm = True, log1p = True)
# hv_genes = find_hv_genes(data)
# print(hv_genes)
# data = data[:, hv_genes]
print("Preprocessing Done. Total Running Time: %s seconds" %
      (time.time() - start_time))

