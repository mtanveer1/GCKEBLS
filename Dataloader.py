import torch
import random
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize


def load_data(path, dataset, rand_split, device):
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    if ss.isspmatrix(features):
        features = features.todense()
    features = np.asarray(features)
    features = normalize(features)
    adj = data['adj']
    if ss.isspmatrix(adj):
        adj = adj.todense()

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    n_class = len(np.unique(labels))
    print('dataset: {}\n# node: {}\n# class: {}'.format(
        dataset, features.shape[0], n_class
    ))
    if rand_split:
        idx_train, idx_test, idx_val, idx_unlabeled = generate_partition_random(data, 20)
    else:
        idx_train, idx_test, idx_val, idx_unlabeled = generate_partition(data)
    adj_hat = torch.from_numpy(construct_adj_hat(adj).todense()).float().to(device)
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).long()
    return adj_hat, features, labels, n_class, idx_train, idx_val, idx_test


def construct_adj_hat(adj):
    adj = ss.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = 2 * ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) 
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  
    adj_hat = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_hat


def generate_partition(data):
    idx_train = data['train'].flatten()
    idx_val = data['val'].flatten()
    idx_test = data['test'].flatten()
    print('train: {}, val: {}, test: {}'.format(len(idx_train), len(idx_val), len(idx_test)))
    return idx_train, idx_test, idx_val, []


def generate_partition_random(labels, num_perclass):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}
    for label in each_class_num.keys():
        labeled_each_class_num[label] = num_perclass
    num_test = 1000
    num_val = 500
    idx_train = []
    idx_test = []
    idx_val = []
    idx_unlabeled = []
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            idx_train.append(index[idx])
        else:
            idx_unlabeled.append(index[idx])
            if num_test > 0:
                num_test -= 1
                idx_test.append(index[idx])
            elif num_val > 0:
                num_val -= 1
                idx_val.append(index[idx])
    print('train: {}, val: {}, test: {}'.format(len(idx_train), len(idx_val), len(idx_test)))
    return idx_train, idx_test, idx_val, idx_unlabeled

def count_each_class_num(labels):
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


