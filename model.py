import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from data import Dataloader, NormalizePolicy
from module import *
import os
from abc import ABCMeta, abstractmethod


class NNClassifier:
    def __init__(self, batch_size=200, num_epochs=500, lr=0.002, optim_method="null", momentum_param=0.9,
                 eva_epoch=1, Model=None, loss_thre=1e12, reg=1e-5, lr_decay=0.99, lr_decay_interval=2):
        self.bs = batch_size
        self.epoch = num_epochs
        self.lr = lr
        self.Model = MLPModel() if Model is None else Model
        self.loss = CEwithSoftmax()
        self.optimizer = SimpleBatchGD(lr, self.Model, optim_method=optim_method,
                                       momentum_param=momentum_param, reg=reg)
        self.val_epoch = eva_epoch
        self.best_params = []
        self.loss_rec = []
        self.val_rec = []
        self.val_acc = []
        self.norm = NormalizePolicy()
        self.lr_decay = lr_decay
        assert type(lr_decay_interval) is int
        self.lr_decay_interval = lr_decay_interval

    # def set_model(self, linearSize, activation="LeakyReLU", **kwargs):
    #     self.Model.set_model_by_linear_size(linearSize, activation, **kwargs)
    def set_model(self, **kwargs):
        self.Model.set_model_by_linear_size(**kwargs)

    def fit(self, X, y, X_val, y_val, X_test, y_test, log=None, do_test=True, return_best_val_acc=True):
        dataloader = Dataloader(X, y, self.bs)
        self.steps_epoch = len(dataloader)
        self.num_train = X.shape[0]
        # self.norm.fit_transform(dataloader)
        best_score = 0
        self.best_params.clear()
        print(self.Model)
        self.loss_rec = []
        self.loss_acc = []
        self.val_rec = []
        self.val_acc = []
        pbar = tqdm(range(self.epoch))
        score = None
        for epoch in pbar:
            if epoch % self.lr_decay_interval == 0:
                self.lr *= self.lr_decay
            # if epoch == 150:
            # self.lr /= 10.
            cor = 0
            pbar_inner = tqdm(dataloader, total=len(dataloader), leave=False)
            for data, label in pbar_inner:
                self.Model.train()
                pre = self.Model(data)
                trn_loss = self.loss(pre, label)
                grad = self.loss.backward()
                self.Model.backward(grad)
                self.optimizer.step()
                cor += np.sum(np.argmax(pre, axis=1) == np.argmax(label, axis=1))
                self.loss_rec.append(trn_loss)
            self.loss_acc.append(cor / self.num_train)
            if not epoch % self.val_epoch:
                self.Model.eval()
                val_loss, score = self.eva(X_val, y_val)
                self.val_acc.append(score)
                if score > best_score:
                    best_score = score
                    self.best_params = []
                    for module in self.Model.modules:
                        if hasattr(module, "params") and "W" in module.params.keys():
                            self.best_params.append([module.params["W"], module.params["b"]])
                        else:
                            self.best_params.append(None)
            pbar.postfix = f"loss: {trn_loss:.2f}, acc: {100 * cor / self.num_train:.2f}%"
            if not epoch % self.val_epoch:
                pbar.postfix += f", val_loss: {val_loss:.2f}, val_acc: {100*score:.2f}%"

        print(f"best val: {100 * best_score: .3f}%")
        if log is not None:
            with open(os.path.join("./result/", log["path"]+".txt"), "w") as f:
                for k, v in log.items():
                    f.write(f"{k}: {v}\n")
                f.write(str(self.Model)+"\n")
                f.write(f"best val: {100 * best_score: .3f}%\n")

        if do_test:
            self.predict(X_test, y_test)
        RETURN_dict = {
            "model": self.best_params
        }
        if return_best_val_acc:
            RETURN_dict["best_acc"] = best_score
        return RETURN_dict

    def predict(self, X_test, y_test, params=[]):
        self.Model.eval()
        best_params = params or self.best_params
        if best_params:
            for i in range(len(self.Model.modules)):
                if best_params[i] is not None:
                    self.Model.modules[i].params["W"] = best_params[i][0]
                    self.Model.modules[i].params["b"] = best_params[i][1]

        loss, best_test_score = self.eva(X_test, y_test)
        print(f"Test: Loss {loss:.3f}; Acc {100 * best_test_score: .3f}%")

    # def predict(self, X, norm=False):
    #     if norm:
    #         X = self.norm.transformMat(X)
    #     pre = np.argmax(self.Model(X), axis=1)
    #     return pre

    def eva(self, X, y):
        self.Model.eval()
        pred = self.Model(X)
        val_loss = self.loss(pred, y)
        self.val_rec.append(val_loss)
        pre = np.argmax(pred, axis=1)
        return val_loss, np.sum(pre == np.argmax(y, axis=1)) / X.shape[0]
    
    def save(self, params=[], log=None):
        params = params or self.best_params
        if params:
            with open(os.path.join("./result/", log["path"]+"_best_param.json"), 'w') as f:
                json.dump([[i[0].tolist(), i[1].tolist()] if i else None for i in params], f, indent=4)
        else:
            raise RuntimeError("invalid model params")
        
    def load(self, params=[], path=None):
        if params:
            for i in range(len(self.Model.modules)):
                if params[i] is not None:
                    self.Model.modules[i].params["W"] = params[i][0]
                    self.Model.modules[i].params["b"] = params[i][1]
        elif path:
            with open(path) as f:
                params = json.load(f)
            for i in range(len(self.Model.modules)):
                if params[i] is not None:
                    self.Model.modules[i].params["W"] = np.array(params[i][0])
                    self.Model.modules[i].params["b"] = np.array(params[i][1])
        else:
            raise RuntimeError("Invalid model params, no param file.")

    def plot_loss(self, flag="all", log=None, imshow=True):
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot([50 * i + 1 for i in range(len(self.loss_rec[::50]))], self.loss_rec[::50], label="train")
        plt.plot([self.val_epoch * self.steps_epoch * i + 1 for i in range(len(self.val_rec))], self.val_rec, label="validation")
        plt.legend()
        plt.xlabel("steps")
        plt.ylabel("loss")
        if flag == "all":
            plt.subplot(122)
            plt.plot([self.val_epoch * i + 1 for i in range(len(self.val_acc))], self.val_acc)
            plt.plot([i+1 for i in range(len(self.loss_acc))], self.loss_acc)
            plt.xlabel("epoch")
            plt.ylabel("acc on validation set")
        plt.savefig(os.path.join("./result/", log["path"]+".png"))
        if imshow:
            plt.show()


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.modules = None
    
    def eval(self):
        for module in self.modules:
            module.eval()
        return self
    
    def train(self):
        for module in self.modules:
            module.train()
        return self

    @abstractmethod
    def set_model_by_linear_size(self, *args, **kwargs):
        pass

    def forward(self, X):
        for module in self.modules:
            X = module(X)
        return X

    def __call__(self, X):
        return self.forward(X)

    def backward(self, grad):
        for module in self.modules[::-1]:
            grad = module.backward(grad)

    def __str__(self):
        res = "Model: Input -> "
        for module in self.modules:
            res += f"[{str(module)}] -> "
        res += "Output"
        return res


class MLPModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.modules = []
        self.layer = 0
        if args:
            self.set_model_by_linear_size(args, **kwargs)

    def set_model_by_linear_size(self, linearSize, activation="LeakyReLU", normalize=True,
                                 dropout_rate=0.5, **kwargs):
        assert linearSize[0] == 3072
        assert linearSize[-1] == 10
        self.modules.clear()
        self.layer = len(linearSize) - 1
        # self.modules.append(PreNormalization())
        for i in range(self.layer):
            self.modules.append(Linear(linearSize[i], linearSize[i + 1], **kwargs))
            if i < self.layer - 1:
                if normalize:
                    self.modules.append(BatchNormalization(linearSize[i + 1]))
                self.modules.append(eval(activation)(**kwargs))
                if dropout_rate != 0:
                    self.modules.append(Dropout(dropout_rate=dropout_rate))
            # else:
            #     self.modules.append(Softmax())

    def forward(self, X):
        for module in self.modules:
            X = module(X)
        return X

    def __call__(self, X):
        return self.forward(X)

    def backward(self, grad):
        for module in self.modules[::-1]:
            grad = module.backward(grad)


class CNNModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.modules = []
        self.linear_layer = 0
        self.conv_channel = 0
        if args:
            self.set_model_by_linear_size(*args, **kwargs)

    def set_model_by_linear_size(self, convChannel: int, linearSize: list,
                                 activation="LeakyReLU", normalize=True, **kwargs):
        self.conv_channel = convChannel
        self.modules.clear()
        self.modules.extend([Conv2d_square_stride1(channel=convChannel, **kwargs),
                             LeakyReLU(eps=0),
                             Pooling(**kwargs),
                             Conv2d_square_stride1(channel=convChannel, **kwargs),
                             LeakyReLU(eps=0),
                             Pooling(**kwargs)])
        self.layer = len(linearSize) - 1
        for i in range(self.layer):
            self.modules.append(Linear(linearSize[i], linearSize[i + 1], **kwargs))
            if i < self.layer - 1:
                if normalize:
                    self.modules.append(BatchNormalization(linearSize[i + 1]))
                self.modules.append(eval(activation)(**kwargs))

    def forward(self, X):
        for module in self.modules[:6]:
            X = module(X)
        X = X.reshape((-1, self.conv_channel * 16))
        for module in self.modules[6:]:
            X = module(X)
        return X

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def backward(self, grad):
        for module in self.modules[-1:5:-1]:
            grad = module.backward(grad)
        grad = grad.reshape((-1, self.conv_channel, 4, 4))
        for module in self.modules[5::-1]:
            grad = module.backward(grad)


class SimpleBatchGD:
    def __init__(self, init_lr, model, optim_method="null", momentum_param=0.9, reg=1e-5):
        self.init_lr = init_lr
        self.model = model
        self.momentum = False
        self.momentum_param = momentum_param
        if optim_method == "momentum":
            self.momentum = True
            print("=== use momentum ===")
        self.last_params = []
        self.reg = reg

    def _update_model(self):
        for module in self.model.modules:
            tmp = {}
            if hasattr(module, "grads"):
                for key in module.params.keys():
                    tmp[key] = module.params[key]
            self.last_params.append(tmp)

    def step(self):
        if not self.last_params:
            self._update_model()
        for module, param_dict in zip(self.model.modules, self.last_params):
            if hasattr(module, "grads"):
                for key in module.params.keys():
                    if self.momentum:
                        module.params[key] -= self.init_lr * module.grads[key] \
                                              -self.momentum_param * (module.params[key] - param_dict[key])\
                                              + self.reg * module.params[key]
                        param_dict[key] = module.params[key]
                    else:
                        module.params[key] -= self.init_lr * module.grads[key]\
                                              + self.reg * module.params[key]
