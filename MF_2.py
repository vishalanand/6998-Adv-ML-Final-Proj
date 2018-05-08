
# coding: utf-8

# In[1]:


import sys, os
sys.path.insert(0, '/Users/Sp0t/Desktop/6998-Adv-ML-Final-Proj/wmf')

import time
import wmf
import numpy as np
import pandas as pd
import bottleneck as bn
from scipy import sparse
from sklearn import decomposition


# In[2]:

PREFIX = 'artist_'
IID = 'iid' # item ID
PID = 'pid' # playlist ID

DATA_DIR = 'mpd_proc/data_2'

unique_iid = list()
with open(os.path.join(DATA_DIR, PREFIX + 'unique_' + PREFIX[0] + IID[1:] + '.txt'), 'r') as f:
    for line in f:
        unique_iid.append(line.strip())

n_items = len(unique_iid)


# In[3]:


df = pd.read_csv(os.path.join(DATA_DIR, PREFIX + 'train.csv'))
n_playlists = df[PID].max() + 1

rows, cols = df[PID], df[IID]
train_data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)),
                               dtype='float64',
                               shape=(n_playlists, n_items))


# In[4]:


sparsity = 1. * df.shape[0] / (n_playlists * n_items)

print("%d playlist inclusion events from %d playlist and %d %s (sparsity: %.3f%%)" %
      (df.shape[0], n_playlists, n_items, PREFIX[:-1] + 's', sparsity * 100))


# In[5]:


train_data


# Ensure the data is binary

# In[6]:


train_data.data[np.where(train_data.data > 1)] = 1.0


# In[7]:


np.where(train_data.data > 1)


# In[8]:


def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr[PID].min(), tp_te[PID].min())
    end_idx = max(tp_tr[PID].max(), tp_te[PID].max())

    rows_tr, cols_tr = tp_tr[PID] - start_idx, tp_tr[IID]
    rows_te, cols_te = tp_te[PID] - start_idx, tp_te[IID]

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


# In[9]:


valid_train, valid_test = load_tr_te_data(os.path.join(DATA_DIR, PREFIX + 'validation_tr.csv'),
                                          os.path.join(DATA_DIR, PREFIX + 'validation_te.csv'))


# In[10]:


idcg = lambda x, k: np.sum(1/np.log2(np.arange(2, min(k, x)+2)))
vidcg = np.vectorize(idcg)
    
def ndcg_recall_on_batch(preds, holdouts, k=100):
    N, M = preds.shape
    total_items = holdouts.getnnz(axis=1)
    
    top_inds = bn.argpartition(-preds, k, axis=1)[:,:k]
    top_items = preds[np.arange(N)[:, np.newaxis],top_inds]
    ranked_inds = np.argsort(-top_items, axis=1)[:,:k]
    ranked_items = top_inds[np.arange(N)[:,np.newaxis],ranked_inds]

    matches = holdouts[np.expand_dims(np.arange(N), 1),ranked_items]
    dcg = np.sum(matches/np.log2(np.arange(k)+2), axis=1)
    idcg = vidcg(total_items, k)
    ndcg = dcg/idcg
    
    recalls = np.sum(matches, axis=1)/np.minimum(k, total_items)
    
    return ndcg, recalls


# In[11]:


def calculate_rmse(W, H, data):
    squared_error = 0
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    for row in range(n_rows):
        data_row_pred = W[row].dot(H.T)
        data_row = data[0].toarray()[0]
        squared_error += np.sum(np.square(data_row_pred - data_row))
    rmse = np.sqrt(squared_error) * 1/(n_rows * n_cols)
    return rmse


# In[12]:


train_data[train_data.nonzero()[0][0], train_data.nonzero()[1][0]]


# In[13]:


# S = wmf.linear_surplus_confidence_matrix(train_data, alpha=100)


# In[14]:


# S[train_data.nonzero()[0][0], train_data.nonzero()[1][0]]


# In[15]:


class MF():
    def __init__(self):
        self.W = None
        self.H = None
        self.trained = False
        
    def train(
        self, 
        data, 
        num_factors=25,
        lambda_reg=1e-3,
        num_iterations=2,
        init_std=0.01,
        verbose=True
    ):
        self.data = data
        self.W, self.H = wmf.factorize(
            data,
            num_factors=num_factors,
            lambda_reg=lambda_reg,
            num_iterations=num_iterations,
            init_std=init_std,
            verbose=verbose,
            dtype=np.float64,
            recompute_factors=wmf.recompute_factors_bias
        )
        self.trained = True
        
    def predict(self, X):
        if self.trained:
            most_similar_rows = self.find_most_similar_rows(X)
            return self.W[most_similar_rows].dot(self.H.T)
            # predictions = np.array([], dtype=np.float64)
            # for row in range(X.shape[0]):
            #     features = X[row].toarray()[0]
            #     prediction = self._predict_single(features)
            #     if predictions.size == 0:
            #         predictions = prediction
            #     else:
            #         predictions = np.vstack((predictions, prediction))
            # return predictions

    def find_most_similar_rows(self, X):
        most_similar_rows = np.array([], dtype=np.int64)
        for row in range(X.shape[0]):
            features = X[row].toarray()[0]
            most_similar_row = self.find_most_similar_row(features)
            most_similar_rows = np.append(most_similar_rows, most_similar_row)
        return most_similar_rows
    
    def find_most_similar_row(self, features):
        return np.argmax(self.data.dot(features.T))

    def _predict_single(self, features):
        playlist_similarity_matrix = self.data.dot(features.T)
        most_similar_playlists = np.argwhere(playlist_similarity_matrix == np.max(playlist_similarity_matrix))
        return np.sum(self.W[most_similar_playlists], axis=0).dot(self.H.T)


# In[16]:


def calculate_metrics(model, val_train_data, val_test_data, k=100, logs={}):
    best_ndcg = 0.0
    N = val_train_data.shape[0]
    ndcg_batch_list = []
    recall_batch_list = []

    for start in range(0, N, 500):
        end = min(N, start+500)
        X = val_train_data[start:end]
#         X = X.toarray().astype('float32')
#         preds = np.asarray(model.predict(X))
        preds = model.predict(X)
        preds[X.nonzero()] = -np.inf
        ndcg_batch, recall_batch = ndcg_recall_on_batch(preds, val_test_data[start:end], k)
        ndcg_batch_list.append(ndcg_batch)
        recall_batch_list.append(recall_batch)
        print('Start: {}'.format(start))

    ndcg = np.mean(ndcg_batch_list)
    recall = np.mean(recall_batch_list)

    print("NDCG@{}: {}".format(k, ndcg))
    print("Recall@{}: {}".format(k, recall))
    return ndcg, recall
#     ndcg_hist.append(ndcg)
#     recall_hist.append(recall)

#     if ndcg > best_ndcg: 
#         self.model.save(os.path.join(self.model_dir, 'model.h5'))
#         best_ndcg = ndcg

#     return best_ndcg


# In[22]:


num_factors_list = [75]
num_iterations_list = [5]
lambda_reg_list = [1e-5] #, 1e-1, 1e1, 1e2]
linear_surplus_alpha_list = [5]


# In[23]:


def run_parameters_sweep():
    metric_hist = []
    for num_factors in num_factors_list:
        for lambda_reg in lambda_reg_list:
            for num_iterations in num_iterations_list:
                for alpha in linear_surplus_alpha_list:
                    print('Test:')
                    print('\tnum_factors   : {}'.format(num_factors))
                    print('\tlambda_reg    : {}'.format(lambda_reg))
                    print('\tnum_iterations: {}'.format(num_iterations))
                    print('\talpha         : {}'.format(alpha))
                    print('================')
                    start_time = time.time()
                    S = wmf.linear_surplus_confidence_matrix(train_data, alpha=alpha)
                    model = MF()
                    model.train(
                        S,
                        num_factors=num_factors,
                        lambda_reg=lambda_reg,
                        num_iterations=num_iterations
                    )
                    for k in [20, 50, 100]:
                        ndcg, recall = calculate_metrics(model, valid_train, valid_test, k=k)
                        metric_hist += [{
                            'num_factors': num_factors,
                            'lambda_reg': lambda_reg,
                            'num_iterations': num_iterations,
                            'alpha': alpha,
                            'ndcg@' + str(k): ndcg,
                            'recall@' + str(k): recall,
                            'k': k
                        }]
                    end_time = time.time()
                    print ("Time elapsed: {}s".format(round(end_time - start_time, 1)))
                    print('================')
    return metric_hist


# In[24]:


run_parameters_sweep()
