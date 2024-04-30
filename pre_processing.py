# !/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py


def read_csv(filename):
    """ Read TPM data of a dataset saved in csv format
    Format of the csv:
    first row: sample labels
    second row: cell labels
    third row: cluster labels from Seurat
    first column: gene symbols
    Args:
        filename: name of the csv file
        take_log: whether do log-transformation on input data
    Returns:
        dataset: a dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    """
    dataset = {}
    df = pd.read_csv(filename,low_memory=False,error_bad_lines=False)
    dat = df[df.columns[1:]].values
    dataset['sample_labels'] = dat[0, :].astype(int)
    dataset['cell_labels'] = dat[1, :].astype(int)
    # dataset['cluster_labels'] = dat[2, :].astype(int)
    gene_sym = df[df.columns[0]].tolist()[2:]
    gene_exp = dat[2:, :]

    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    return dataset


def pre_processing_single(dataset_file_list, type='csv'):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    dataset_list = []
    data_file = dataset_file_list

    if type == 'csv':
        dataset = read_csv(data_file)
    dataset['gene_exp'] = dataset['gene_exp'].astype(np.float)

    dataset_list.append(dataset)
    return dataset_list



class load_data_origin_data(Dataset):
    def __init__(self, dataset, load_type="csv"):
        def load_txt():
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

        def load_h5():
            data_mat = h5py.File(dataset)
            self.x = np.array(data_mat['X'])
            self.y = np.array(data_mat['Y'])

        def load_csv():
            dataset_list = pre_processing_single(dataset, type='csv')
            self.x = dataset_list[0]['gene_exp'].transpose().astype(np.float32)
            self.y = dataset_list[0]['cell_labels'].astype(np.int32)


        if load_type == "csv":
            load_csv()
        elif load_type == "h5":
            load_h5()
        elif load_type == "txt":
            load_txt()



    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


