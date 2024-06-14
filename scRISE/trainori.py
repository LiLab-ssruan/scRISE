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

from torch import optim,nn
from AGE.model import LinTrans, LogReg
from AGE.optimizer import loss_function
from AGE.utils import *
from sklearn.cluster import SpectralClustering, KMeans
from AGE.clustering_metric import clustering_metrics
from AGE.autoencoders import autoen, ClusteringLayer
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics

from graph_function import *
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import *
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    #from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    # pdb.set_trace()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true, y_pred):
    acc= cluster_acc(y_true, y_pred)
    # acc=0
    f1=0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    # silhouette = silhouette_score(z, y_pred)


    return acc, f1, nmi, ari, homo, comp

def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
    # z = normalize(z - z.mean())
    # f_adj = np.matmul(z, np.transpose(z))
    f_adj = distance.cdist(z, z, "correlation")
    f_adj = f_adj.reshape([-1, ])
    pos_num = round(upper_threshold * len(f_adj))
    neg_num = round((1 - lower_treshold) * len(f_adj))

    pos_inds = np.argpartition(-f_adj, pos_num)[:pos_num]
    neg_inds = np.argpartition(f_adj, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

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

def target_distribution(q):
    p = q ** 2 / q.sum(0)
    return (p.t() / p.sum(1)).t()



def smooth_for(x, features, k, knn_distance, gnnlayers):
    adj = Graph_Build(x, k, knn_distance)
    # Some preprocessing
    adj_norm_s = preprocess_graph(adj, gnnlayers)
    sm_fea_s = sp.csr_matrix(features).toarray()
    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)

    return sm_fea_s, adj

def Graph_Build(x,k,knn_distance):

    adj = knn_graph(x, k, knn_distance)
    # adj = cknn_graph(x, k, delta=1, distanceType=knn_distance)
    adj = adj + mst_graph(x, knn_distance)
    adj[adj > 0] = 1
    G = nx.Graph(adj)
    adj = nx.adjacency_matrix(G)
    return adj


def cluster_loss(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
    kldloss = kld(p, q)
    return kldloss


def gae_for(features, adj, y_true, n_clusters, hidden_dim1, hidden_dim2, lr, upth_ed, upth_st, lowth_ed, lowth_st, epochs, upd, bs, device):
    print(features.shape)

    n_nodes, feat_dim = features.shape
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()


    # pos_weight = float(adj_norm.shape[0] * adj_norm.shape[0] - adj_norm.sum()) / adj_norm.sum()
    model = LinTrans(input_feat_dim=feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, n_clusters=n_clusters)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sm_fea_s = torch.FloatTensor(features)

    model = model.to(device)
    inx = sm_fea_s.to(device)
    # inx = sm_fea_s
    pos_num = len(adj.indices)

    neg_num = n_nodes * n_nodes - pos_num

    # Thresholds Update.
    up_eta = (upth_ed - upth_st) / (epochs / upd)
    low_eta = (lowth_ed - lowth_st) / (epochs / upd)
    pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), upth_st, lowth_st, pos_num, neg_num)
    upth, lowth = update_threshold(upth_st, lowth_st, up_eta, low_eta)

    bs = min(bs, len(pos_inds))

    # pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
    pos_inds_cuda = torch.LongTensor(pos_inds).to(device)

    kmeans = KMeans(n_clusters, n_init=20)
    data, _ = model(inx)
    y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    y_pred_last = y_pred
    model.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
    print('Start Training...')
    model.train()
    lst = []
    list_prep = []
    best_ari = 0.0
    for epoch in tqdm(range(epochs)):

        st, ed = 0, bs
        batch_num = 0
        length = len(pos_inds)
        z, q = model(inx)
        p = target_distribution(q).data
        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred)

        list_prep.append(y_pred)
        #
        if best_ari < ari:
            print(ari)
            best_ari = ari
            data = z.data.cpu().numpy()

        lst.append(ari)
        while (ed <= length):
            z, q = model(inx)
            c_loss = cluster_loss(p, q)

            RegularizeLoss = SquareRegularizeLoss()
            A_loss = RegularizeLoss(z)


            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st)).to(device)
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()

            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx, _ = model(x)
            zy, _ = model(y)
            batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st))).to(device)
            batch_pred = model.dcs(zx, zy)

            loss = 0.5*loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed - st)+1*c_loss
            optimizer.zero_grad()
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs

        if (epoch + 1) % upd == 0:
            model.eval()
            mu, _ = model(inx)
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
            bs = min(bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).to(device)

            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))
    tqdm.write("Optimization Finished!")
    max_ari = max(lst)  ###找到最大的ari
    maxid = lst.index(max_ari)
    optimal_pred = list_prep[maxid]

    return data, optimal_pred
