import scipy.sparse as sp
from sklearn import preprocessing
import networkx as nx
import pickle as pkl
import pandas as pd
import numpy as np
from igraph import *
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
import scanpy as sc


def readLTMG(LTMGDir, ltmgfile):
    '''
    Read LTMG matrix as the regularizor. sparseMode for huge datasets sparse coding, now only use sparseMode
    '''
    # sparse mode
    # if sparseMode:
    df = pd.read_csv(LTMGDir+ltmgfile, header=None,
                     skiprows=1, delim_whitespace=True)
    for row in df.itertuples():
        # For the first row, it contains the number of genes and cells. Init the whole matrix
        if row[0] == 0:
            matrix = np.zeros((row[2], row[1]))
        else:
            matrix[row[2]-1][row[1]-1] = row[3]
    # nonsparse mode: read in csv format, very very slow when the input file is huge, not using
    # else:
    #     matrix = pd.read_csv(LTMGDir+ltmgfile,header=None, index_col=None, delimiter='\t', engine='c')
    #     matrix = matrix.to_numpy()
    #     matrix = matrix.transpose()
    #     matrix = matrix[1:,1:]
    #     matrix = matrix.astype(int)
    return matrix





def loadscExpression(csvFilename, sparseMode=True):
    '''
    Load sc Expression: rows are genes, cols are cells, first col is the gene name, first row is the cell name.
    sparseMode for loading huge datasets in sparse coding
    Transpose
    '''
    if sparseMode:
        print('Load expression matrix in sparseMode')
        genelist = []
        celllist = []
        with open(csvFilename.replace('.csv', '_sparse.npy'), 'rb') as f:
            objects = pkl.load(f, encoding='latin1')
        matrix = objects.tolil()
        matrix = np.transpose(matrix) #转置

        with open(csvFilename.replace('.csv', '_cell.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                celllist.append(line)

        with open(csvFilename.replace('.csv', '_gene.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                genelist.append(line)

    else:
        print('Load expression in csv format')
        matrix = pd.read_csv(csvFilename, index_col=0)
        genelist = matrix.index.tolist()
        celllist = matrix.columns.values.tolist()
        matrix = matrix.to_numpy()
        matrix = matrix.astype(float)

    return matrix, genelist, celllist

class scDataset(Dataset):
    def __init__(self, data=None, transform=None):
        """
        Args:
            data : sparse matrix.
            transform (callable, optional):
        """
        # Now lines are cells, and cols are genes
        self.features = data
        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx

def preprocess_features(features):
    """Row-normalize feature matrix"""
    features = preprocessing.minmax_scale(features,feature_range=(0, 1), axis=0, copy=True)
    return features


class scDatasetInter(Dataset):
    def __init__(self, features, transform=None):
        """
        Internal scData
        Args:
            construct dataset from features
        """
        self.features = features
        # Now lines are cells, and cols are genes
        # self.features = self.features.transpose()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        sample = torch.from_numpy(sample.toarray())

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample


def generateLouvainCluster(edgeList):
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size

def trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30):
    '''
    If the clustering numbers larger than certain number, use this function to trim. May have better solution
    '''
    numDict = {}
    for item in listResult:
        if not item in numDict:
            numDict[item] = 0
        else:
            numDict[item] = numDict[item]+1

    size = len(set(listResult))
    changeDict = {}
    for item in range(size):
        if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
            changeDict[item] = ''

    count = 0
    for item in listResult:
        if item in changeDict:
            listResult[count] = maxClusterNumber
        count += 1

    return listResult


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
    silhouette =  metrics.silhouette_score(z, y_pred)
    if pp:
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),'silhouette {:.4f}'.format(silhouette))
    return acc, nmi, ari, silhouette

def scnormalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        # threshold = np.percentile(adata.X.sum(axis=0), 1)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
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
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata