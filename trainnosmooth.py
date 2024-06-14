import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from utils import *
from torch.utils.data import DataLoader
from pre_processing import load_data_origin_data
from model import AE
import torch
torch.cuda.empty_cache()
from torch import nn, optim
import pandas as pd
from sklearn.cluster import KMeans
from AGE.train import gae_for, smooth_for, Graph_Build
from graph_function import *
import scanpy as sc
import random
from sklearn.cluster import SpectralClustering


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(epoch, dataloader):
    model.train()
    train_loss = 0
    for batch_idx, (data, dataindex) in enumerate(dataloader):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z = model(data)
        loss_function = nn.MSELoss(reduction='sum')
        loss = 0.1*loss_function(recon_batch,data.view(-1, recon_batch.shape[1]))

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
        epoch, train_loss / len(dataloader.dataset)))
    
    return recon_batch_all, data_all, z_all



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main entrance of scGASA')
    #Preprocess-------------------------------------------------------------------------------------------
    parser.add_argument('--data', type=str, default='mESC',
                        help='Baron_mouse.csv')
    parser.add_argument('--delim', type=str, default='comma',
                        help='File delim type, comma or space: default(comma)')
    parser.add_argument('--highly_genes', type=int, default=2000)
    #AE----------------------------------------------------------------------------------------------------
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 12800)')
    parser.add_argument('--Regu-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train in Autoencoder initially (default: 500)')
    parser.add_argument('--run', type=int, default=5, metavar='N',
                        help='(default: 5)')
    # Build cell graph-------------------------------------------------------------------------------------
    parser.add_argument('--k', type=int, default=15,
                        help='parameter k in KNN graph (default: 10)')
    # Clustering related-------------------------------------------------------------------------------------
    parser.add_argument('--n-clusters', type=int, default=10, help='number of clusters')
    # AGE related---------------------------------------------------------------------------------------------
    parser.add_argument('--gnnlayers', type=int, default=5,
                        help='Number of gnn layers 61')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.default:500')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--upth_st', type=float, default=0.0015,
                        help='Upper Threshold start.')
    parser.add_argument('--lowth_st', type=float, default=0.3, help='Lower Threshold start.')
    parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
    parser.add_argument('--lowth_ed', type=float, default=0.7, help='Lower Threshold end')
    parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
    parser.add_argument('--seed', type=int, default=1111, help='seed')
    parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    setup_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    print('Using device:'+str(device))
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
    x = np.ceil(x).astype(int)
    y = dataset.y
    args.n_clusters = int(max(y) - min(y) + 1)
    print(x.shape)
    print("num_clusters",args.n_clusters)
    print('Data loaded, start filtering...')


    adata = sc.AnnData(x)
    adata = scnormalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    data = adata.X
    print("Preprocessing Done. Total Running Time: %s seconds" %
        (time.time() - start_time))

    hidden_dim = [256, 64, 32]
    for i in range(1, args.run + 1):
        print(i)
        adj = Graph_Build(data, args.k)
        dataset = scDataset(data)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model = AE(input_dim=data.shape[1], hidden_dims = hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1, args.Regu_epochs + 1):
            recon, original, z = train(epoch, dataloader)
        Z = z.detach().cpu().numpy()
        R = recon.detach().cpu().numpy()
        data = data = smooth_for(R, adj, args.gnnlayers)      

    # sm_fea = data
    # hid_dim =[256, 32]
    # zOut, y_prep = gae_for(features=sm_fea, hid_dim=hid_dim,adj=adj, y_true=y, n_clusters=args.n_clusters, lr=args.lr,
    #                        upth_ed=args.upth_ed, upth_st=args.upth_st, lowth_ed=args.lowth_ed, lowth_st=args.lowth_st,
    #                        epochs=args.epochs, upd=args.upd, bs=args.bs, w1=10, w2=10,device=device)
    zOut = data
    print("Clustering Done. Total Running Time: %s seconds" %
          (time.time() - start_time))
    embedding_df = pd.DataFrame(zOut)
    embedding_df.to_csv(args.datasetDir + args.data + '_embeddingSMOOTH.csv')
    results_df = pd.DataFrame(y_prep, columns=["Celltype"])
    results_df.to_csv(args.datasetDir + args.data + '_resultsSMOOTH.csv')
    y_true = y
    # acc, nmi, ari, silhouette = eva(zOut, y_true, y_prep)

    # clustering = KMeans(n_clusters=args.n_clusters,
    #                     random_state=args.seed).fit(sm_fea)
    # listResult = clustering.predict(sm_fea)
    # print('Kmeans zOut')
    # acc, nmi, ari, silhouette = eva(sm_fea, y_true, listResult)

    # clustering = KMeans(n_clusters=args.n_clusters,
    #                     random_state=args.seed).fit(adata.X)
    # listResult = clustering.predict(adata.X)
    # print('Kmeans raw')
    # acc, nmi, ari, silhouette = eva(adata.X, y_true, listResult)
    
    # print('SpectralClustering raw')
    # sc = SpectralClustering(n_clusters=args.n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
    # listResult = sc.fit_predict(adata.X)
    # acc, nmi, ari, silhouette = eva(adata.X, y_true, listResult)
    
