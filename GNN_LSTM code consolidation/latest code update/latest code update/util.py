import pickle

import networkx
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import networkx as nx
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class DataLoaderM_hosp(object):
    def __init__(self, data, train, valid, device, horizon, window, prediction_window,normalize=2):
        self.P = window
        self.prediction_window = prediction_window
        self.h = horizon
        self.predict_data = None
        self.rawdat = data
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 0
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        self.scale = torch.from_numpy(self.scale).float()

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)
        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / self.scale[i]

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h,self.m))
        if idx_set[-1] == self.n - 1:
            data_split = 'test'
        else:
            data_split = None
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:idx_set[i]+1, :])
        second_half = torch.zeros((self.h, self.P, self.m))
        if data_split == 'test':
            first_half = X[self.h:,:,:]
            for index in range(self.h):
                second_half[index,:,:] = torch.from_numpy(self.dat[(start+index+1):(start+index+self.P+1), :])
            self.predict_data = torch.cat((first_half, second_half), dim=0)
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class DataLoaderS_hosp(object):
    def __init__(self, data, train, valid, device,train_horizon, horizon, window, prediction_window,normalize=2):
        self.P = window
        self.prediction_window = prediction_window
        self.train_horizon = train_horizon
        self.h = horizon
        self.predict_data = None
        self.rawdat = data
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 0
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        self.scale = torch.from_numpy(self.scale).float()

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)
        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / self.scale[i]

    def _split(self, train, valid, test):
        train_set = range(self.P + self.train_horizon - 1, train)
        valid_set = range(train, self.n)
        self.train = self._batchify(train_set, self.train_horizon)
        self.valid = self._batchify(valid_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, horizon,self.m))
        if idx_set[-1] == self.n - 1:
            data_split = 'test'
        else:
            data_split = None
        for i in range(n):
            end = idx_set[i] - horizon + 1 # 12 - 4 + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:idx_set[i]+1, :])
        second_half = torch.zeros((self.h, self.P, self.m))
        if data_split == 'test':
            first_half = X[self.h:,:,:]
            for index in range(self.h):
                second_half[index,:,:] = torch.from_numpy(self.dat[(start+index+1):(start+index+self.P+1), :])
            self.predict_data = torch.cat((first_half, second_half), dim=0)
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size



def load_TE_matrix(MT):
    # This hard coded dictionary is results from MultivariateTE() in idtxl, using timestamp =20, and with a lag of 2
    if MT:
        TE = {
            48: [{6: 0.05232551, 23: 0.0584635, 20: 0.04492083}],
            47: [{21: 0.08801546, 45: 0.00131597, 8: 0.05139504, 5: 0.03611704}],
            46: [{32: 0.03563225, 39: 0.0365081, 41: 0.06169809, 24: 0.01717334, }],
            45: [{33: 0.06535965, 18: 0.02604022, 43: 0.02584262, 1: 0.03773758}],
            44: [{19: 0.10079411, 41: 0.08187466}],
            43: [{35: 0.05960492, 32: 0.06574153, 38: 0.0463955}],
            42: [{5: 0.07943887, 10: 0.05639639, 31: 0.03940296}],
            41: [{8: 0.0161751, 36: 0.03147701, 42: 0.02613964, 39: 0.03079603}],
            40: [{8: 0.10055947, 45: 0.05808693, 46: 0.04656119}],
            39: [{5: 0.01564659, 8: 0.0599683, 47: 0.03606368, 18: 0.02816105}],
            38: [{37: 0.02290785, 8: 0.07354595, 35: 0.0300532, 21: 0.0250704}],
            37: [],
            36: [{12: 0.02162307, 48: 0.01762945, 17: 0.01323069, 34: 0.03055586, 19: 0.02244652}],
            35: [{33: 0.06626002, 48: 0.04498229, 0: 0.04887697}],
            34: [{36: 0.04452033, 11: 0.01194885, 31: 0.0290801, 38: 0.02026184, 26: 0.00360195, 39: 0.02670655}],
            33: [{41: 0.08630029, 39: 0.06516149, 19: 0.05373189}],
            32: [{21: 0.02942886, 14: 0.00546266, 44: 0.02946866, 8: 0.01965424}],
            31: [{12: 0.04744158, 26: 0.06684893, 38: 0.03360169}],
            30: [{8: 0.07242087, 43: 0.05123109, 39: 0.05116866}],
            29: [{36: 0.00020584, 35: 0.01375407, 22: 0.01712204, 17: 0.03659704, 21: 0.03715727}],
            28: [{6: 0.05534549, 13: 0.06461178, 5: 0.07039714}],
            27: [{5: 0.06260995, 8: 0.0922456, 46: 0.03807409}],
            26: [{29: 0.0482842, 41: 0.06130106, 28: 0.02561589, 39: 0.02648196}],
            25: [{22: 0.07855455, 12: 0.06867876}],
            24: [{20: 0.0006549, 4: 0.01690069, 29: 0.03141054, 14: 0.00809592, 23: 0.0329617, 25: 0.02369082}],
            23: [{12: 0.02147475, 47: 0.04573759, 8: 0.05374737, 37: 0.03847727}],
            22: [{8: 0.06254485, 19: 0.06857342}],
            21: [{35: 0.15052277}],
            20: [{2: 0.00350296, 45: 0.02624339, 27: 0.05763167, 32: 0.04455308}],
            19: [{24: 0.06551457, 0: 0.04100909, 31: 0.02833728, 44: 0.03880833}],
            18: [{20: 0.03210129, 34: 0.03582298, 19: 0.0624421}],
            17: [{14: 0.04025358, 35: 0.03588485, 47: 0.03104353}],
            16: [{35: 0.10067553, 22: 0.06108929, 6: 0.04069267}],
            15: [{36: 0.03998054, 16: 0.12034552, 39: 0.04223743}],
            14: [{25: 0.00245498, 32: 0.04486371, 19: 0.04405419, 42: 0.04380478}],
            13: [{7: 0.00157065, 8: 0.06505025, 36: 0.00270655, 48: 0.01940436, 19: 0.01753651, 46: 0.0266226}],
            12: [{19: 0.04729226, 24: 0.04915915, 1: 0.04030601}],
            11: [{8: 0.02239159, 15: 0.03120977, 28: 0.01372593, 5: 0.02857623}],
            10: [{17: 0.05480369, 11: 0.0416419, 46: 0.03400587, 9: 0.03106367}],
            9: [{22: 0.09915284, 45: 0.08817049}],
            8: [{0: 0.01131866, 35: 0.03032135, 39: 0.03527701}],
            7: [],
            6: [{19: 0.05523206, 30: 0.03191467, 15: 0.03856314}],
            5: [{21: 0.06448187, 31: 0.03817584, 44: 0.04005356}],
            4: [{1: 0.06810937, 10: 0.03005981, 28: 0.03989548}],
            3: [{1: 0.08438589, 20: 0.07231518, 6: 0.03148677}],
            2: [{12: 0.0421206, 1: 0.04326041, 24: 0.01011066, 42: 0.01724356}],
            1: [{9: 0.01956007, 35: 0.03194242, 21: 0.01596129, 29: 0.03365001, 39: 0.02652162}],
            0: [{8: 0.06908246, 29: 0.04237187, 22: 0.04386211}],

    }
    adj = np.zeros((49, 49))
    for i in range(49):
        if i not in TE.keys():
            continue
        else:
            for source_dict in TE[i]:
                for src in source_dict.keys():
                    adj[src, i] = source_dict[src]
                    # source_dict[src]
    with open('MTE_matrix_using_real_value','wb') as f:
        pickle.dump(adj, f)
    return adj



