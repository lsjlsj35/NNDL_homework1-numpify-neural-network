import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random


def train_val_split(*data, val_size=0.2):
    random.seed(0)
    num = data[0].shape[0]
    idx = np.arange(num)
    random.shuffle(idx)
    val_size = int(val_size * num)
    # train-x, val-x, train-y, val-y, ...
    return [d for dataset in data for d in [dataset[val_size:], dataset[:val_size]]]


def load(path="./cifar-10-batches-py"):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    X_train = []
    y_train = []
    for i in range(1,6):
        data = unpickle(os.path.join(path, f'data_batch_{i}'))
        X_train.append(data[b'data'])
        y_train.extend(data[b'labels'])
    
    test_data = unpickle(os.path.join(path, 'test_batch'))
    X_test = test_data[b'data']
    y_test = test_data[b'labels']
    
    X_train = np.vstack(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    X_train, X_val, y_train, y_val = train_val_split(
        X_train, y_train, val_size=0.2)
    
    return X_test, y_test, X_train, y_train, X_val, y_val


class NormalizePolicy:
    def __init__(self, dataloader=None, u=None, var=None):
        if dataloader is not None:
            X = dataloader.data
            self.u = np.mean(X)
            self.var = np.var(X)
        else:
            self.u = u
            self.var = var

    def fit_transform(self, dataloader):
        X = dataloader.data
        self.u = np.mean(X)
        self.var = np.var(X)
        dataloader.normalize(self.u, self.var)

    def transform(self, dataloader):
        dataloader.normalize(self.u, self.var)

    def transformMat(self, X):
        M = X - self.u
        M = M / np.sqrt(self.var)
        return M

    @staticmethod
    def norm_for_each(*mats):
        result = []
        for mat in mats:
            u = np.mean(mat, axis=1, keepdims=True)
            var = np.var(mat, axis=1, keepdims=True)
            result.append((mat-u) / np.sqrt(var))
        return result

    @staticmethod
    def max_to_1(*mats):
        return [mat/255 for mat in mats]

    @staticmethod
    def null(*mats):
        return mats


class Dataloader:
    def __init__(self, X, y, batch_size=16):
        """
            输入ndarray，输出tensor
        """
        self.len = X.shape[0]
        idx = np.arange(self.len)
        np.random.shuffle(idx)
        self.data = X[idx]
        self.label = y[idx]
        self.counts = 0
        self.bs = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor((self.len-1) / self.bs)) + 1

    def normalize(self, u=128, var=10):
        self.data -= u
        self.data /= np.sqrt(var)

    def __next__(self):
        if self.counts + self.bs < self.len:
            self.counts += self.bs
            output_X, output_y = self.data[self.counts - self.bs: self.counts], self.label[
                                                                                self.counts - self.bs: self.counts]
            return output_X, output_y
        elif self.counts < self.len:
            output_X, output_y = self.data[self.counts:], self.label[self.counts:]
            self.counts = self.len
            return output_X, output_y
        else:
            idx = np.arange(self.len)
            np.random.shuffle(idx)
            self.data = self.data[idx]
            self.label = self.label[idx]
            self.counts = 0
            raise StopIteration


if __name__ == "__main__":
    xt, yt, xtr, ytr, xv, yv = load()
    print(xtr.shape, ytr.shape)
    print(xtr[:3])
    print(ytr[:3])
    print(np.unique(ytr))
    print(xt.shape[0], xtr.shape[0], xv.shape[0])
