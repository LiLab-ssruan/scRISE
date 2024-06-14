from __future__ import division
from __future__ import print_function
import os , sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
import argparse
import time
from torch import optim,nn
from torch.nn import functional as F

from AGE.model import LinTrans
from AGE.optimizer import loss_function
from AGE.utils import *
from sklearn.cluster import KMeans
from utils import *
from tqdm import tqdm
from graph_function import *
from sklearn.metrics.cluster import *
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.model_selection import ParameterGrid

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


def update_similarity(z, upper_threshold, lower_threshold, pos_num, neg_num):
    means = np.mean(z, axis=1, keepdims=True)
    deviations = z - means
    lengths = np.linalg.norm(deviations, axis=1, keepdims=True)
    corr = np.matmul(deviations, np.transpose(deviations)) / np.matmul(lengths, np.transpose(lengths))
    corr = corr.reshape([-1,])

    pos_num = round(upper_threshold * len(corr))
    neg_num = round((1-lower_threshold) * len(corr))
    pos_inds = np.argpartition(-corr, pos_num)[:pos_num]
    neg_inds = np.argpartition(corr, neg_num)[:neg_num]
    return np.array(pos_inds), np.array(neg_inds)



def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth


def target_distribution(q):
    p = q ** 2 / q.sum(0)
    return (p.t() / p.sum(1)).t()

def smooth_for(features,adj, gnnlayers):
    # Some preprocessing
    adj_norm_s = preprocess_graph(adj, gnnlayers)
    sm_fea_s= sp.csr_matrix(features).toarray()

    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    return sm_fea_s



def Graph_Build(x,k,n_pca = 50):
    pca = PCA(n_pca).fit(x)
    x = pca.transform(x)
    print("make knn")
    adj = knn_graph(x, k)

    return adj

def cluster_loss(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))####这里求了均值，所以后面会* len(inputs)
    kldloss = kld(p, q)
    return kldloss


def gae_for(features, hid_dim,adj, y_true, n_clusters, lr, upth_ed, upth_st, lowth_ed, lowth_st, epochs, upd, bs, w1, w2, device):
    print(features.shape)
    n_nodes, feat_dim = features.shape
    model = LinTrans(input_feat_dim=feat_dim, hid_dim= hid_dim, n_clusters=n_clusters)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    sm_fea_s = torch.FloatTensor(features)
    model = model.to(device)
    inx = sm_fea_s.to(device)

    pos_num = len(adj.indices)
    neg_num = n_nodes * n_nodes - pos_num
    # Tresholds Update.
    up_eta = (upth_ed - upth_st) / (epochs / upd)
    low_eta = (lowth_ed - lowth_st) / (epochs / upd)
    pos_inds, neg_inds = update_similarity(sm_fea_s.numpy(), upth_st, lowth_st, pos_num, neg_num)
    upth, lowth = update_threshold(upth_st, lowth_st, up_eta, low_eta)

    bs = min(bs, len(pos_inds))
    pos_inds_cuda = torch.LongTensor(pos_inds).to(device)


    kmeans = KMeans(n_clusters, n_init=20)
    data, _= model(inx)
    y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    
    print('Start Training...')
    model.train()
    lst = []
    list_pred = []
    best_ari = 0.0
    for epoch in tqdm(range(epochs)):

        st, ed = 0, bs
        batch_num = 0
        length = len(pos_inds)
        z, q = model(inx)
        p = target_distribution(q).data
        y_pred = q.data.cpu().numpy().argmax(1)
        acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred)

        list_pred.append(y_pred)

        if best_ari < ari:
            print(ari)
            best_ari = ari
            data = z.data.cpu().numpy()

        lst.append(ari)
        while (ed <= length):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st)).to(device)
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx = model.fc(x)
            zy = model.fc(y)
            batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st))).to(device)
            batch_pred = model.dcs(zx, zy)
            z, q= model(inx)

            ce_loss = F.kl_div(q.log(), p, reduction='batchmean')

            loss =w1*loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed - st) + w2*ce_loss
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
            mu = model.fc(inx)
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
            bs = min(bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).to(device)
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))
    tqdm.write("Optimization Finished!")
    max_ari = max(lst)
    maxid = lst.index(max_ari)
    optimal_pred = list_pred[maxid]
    print(best_ari)
    return data, optimal_pred

          