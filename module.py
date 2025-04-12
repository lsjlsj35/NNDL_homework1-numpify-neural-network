import numpy as np
from abc import ABCMeta, abstractmethod


class Module(metaclass=ABCMeta):
    _eval = False

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        return self.__class__.__name__
    
    def eval(self):
        self._eval = True
        return self

    def train(self):
        self._train = False
        return self


# Normalization
class BatchNormalization(Module):
    def __init__(self, dim, **kwargs):
        self.d = dim
        self.params = {}
        self.params["beta"] = np.zeros((dim,)).reshape((1, -1))
        self.params["gamma"] = np.ones((dim,)).reshape((1, -1))
        self.grads = {}
        self.x_hat = 0

    def forward(self, X):
        self.u = np.mean(X, axis=0, keepdims=True)
        self.var = np.var(X, axis=0, unbiased=False, keepdims=True)
        self.x_hat = (X - self.u) / np.sqrt(self.var + 1e-8)
        return self.params["gamma"] * self.x_hat + self.params["beta"]

    def backward(self, grad):
        self.grads["gamma"] = np.mean(grad * self.x_hat, axis=0, keepdims=True)
        self.grads["beta"] = np.mean(grad, axis=0, keepdims=True)
        return self.params["gamma"] / np.sqrt(self.var + 1e-8) * grad

    def __str__(self):
        return f"BatchNormalization[{self.d}]"


class PreNormalization(Module):
    def __init__(self, **kwargs):
        pass

    def forward(self, X):
        u = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, unbiased=False, keepdims=True)
        return (X - u) / (np.sqrt(var))

    def backward(self, grad):
        return grad


class LayerNormalization(Module):
    def __init__(self, **kwargs):
        self.var = 0

    def forward(self, X):
        u = np.mean(X)
        self.var = np.sqrt(X)
        return (X - u) / (1e-6 + self.var)

    def backward(self, grad):
        return grad / (self.var + 1e-6)


# Activation and Softmax
class Softsign(Module):
    def __init__(self, **kwargs):
        self.res = None

    def forward(self, X):
        self.res = 1 / (1 + np.abs(X))
        return X * self.res

    def backward(self, grad):
        return np.square(self.res) * grad


class Logistic(Module):
    def __init__(self, **kwargs):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + np.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        outputs_grad_inputs = np.multiply(self.outputs, (1.0 - self.outputs))
        return np.multiply(grads, outputs_grad_inputs)


class LeakyReLU(Module):
    def __init__(self, eps=0.01, **kwargs):
        self.eps = eps
        self.input = None

    def forward(self, X):
        self.input = X
        zeros = np.zeros_like(X)
        return np.maximum(zeros, X) + self.eps * np.minimum(zeros, X)

    def backward(self, grad):
        input_grad = (self.input >= 0).astype(np.float32)
        return grad * np.clip(input_grad, self.eps, 1.)

    def __str__(self):
        return f"LeakyReLU({self.eps})"


class Softmax(Module):
    def __init__(self, **kwargs):
        self.output = None

    def forward(self, X):
        self.output = np.exp(X - np.max(X, axis=1, keepdims=True))  # 减最大值防止溢出
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        return self.output * (1 - self.output) * grad


# baseline
class Linear(Module):
    def __init__(self, input_size, output_size, initial="randn", **kwargs):
        self.input = None
        self.params = {}
        if initial == "all_zero":
            self.params["W"] = np.zeros((input_size, output_size)).astype(np.float32) * 0.01
            self.params["b"] = np.zeros((1, output_size)).astype(np.float32)
        if initial == "randn":
            self.params["W"] = np.random.randn(input_size, output_size).astype(np.float32) * 0.01
            self.params["b"] = np.zeros((1, output_size)).astype(np.float32)
        self.grads = {}

    def forward(self, X):
        self.input = X
        res = X @ self.params["W"] + self.params["b"]
        return res

    def backward(self, grad):
        """
            grad: Shape[N, output_size]
            return: Shape[N, input_size,]
        """
        self.grads["b"] = np.mean(grad, axis=0)
        self.grads["W"] = np.matmul(self.input.T, grad) / self.input.shape[0]
        return np.matmul(grad, self.params["W"].T)

    def __str__(self):
        return f"Linear({self.params['W'].shape[0]}->{self.params['W'].shape[1]})"


class Dropout(Module):
    def __init__(self, dropout_rate=0.5):
        self.dr = dropout_rate
        self.factor = 1 / (1 - dropout_rate + 1e-8)
        self.mask = None

    def forward(self, X):
        if self._eval:
            return X
        self.mask = np.random.rand(*X.shape) > self.dr
        return np.where(self.mask, X, 0) * self.factor

    def backward(self, grad):
        if self._eval:
            return grad
        return np.where(self.mask, grad, 0) * self.factor

    def __str__(self):
        return f"Dropout({self.dr:.2f})"


class Conv2d_square_stride1(Module):
    def __init__(self, kernel_size=3, channel=32, padding=0, **kwargs):
        # padding: None, 数字, "nearest"
        # stride (1, 1)
        # kernel shape (kernel_size, kernel_size)
        self.params = {"k": np.random.randn(channel, kernel_size, kernel_size) * 0.2 + np.random.randn(1) * 0.1}
        self.grad = {"k": np.zeros_like(self.params["k"])}
        self.channel = channel
        self.pad = padding
        self.size = kernel_size
        self.X = None
        self.multichannel = False

    def prePadding(self, X):
        if len(X.shape) == 3:
            X = X[:, None, ...]
            self.multichannel = True
        else:
            self.multichannel = False
        if self.pad is None:
            Xpad = X
        elif type(self.pad) != str:
            Xpad = np.full((X.shape[0], X.shape[1], X.shape[2] - 1 + self.size, X.shape[3] - 1 + self.size), 1.0*self.pad)
            t = self.size // 2
            Xpad[..., t:-t, t:-t] = X
        elif self.pad == "nearest":
            Xpad = np.zeros((X.shape[0], X.shape[1], 15 + self.size, 15 + self.size))
            t = self.size // 2
            Xpad[..., t:-t, t:-t] = X
            Xpad[..., t:-t, 0:t] = Xpad[..., t:-t, t]
            Xpad[..., t:-t, -t:] = Xpad[..., t:-t, -t-1]
            Xpad[:, :, 0:t] = Xpad[:, :, t]
            Xpad[:, :, -t:] = Xpad[:, :, -t-1]
        else:
            assert "invalid padding"
        return Xpad

    def forward(self, X):
        X = self.prePadding(X)
        self.X = X
        s = X.shape[-1]
        result = np.zeros((self.X.shape[0], self.channel, s-self.size+1, s-self.size+1))
        for i in range(self.size):
            for j in range(self.size):
                result += self.params["k"][:, i:i+1, j:j+1][None, ...] * X[..., i:s-self.size+i+1, j:s-self.size+j+1]
        return result

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def backward(self, grad):
        """
        :param grad: [B, C, O, O]
        :return: [B, [C], I, I]
        :self.X: [B, 1/C, P, P]
        """
        self.grad["k"] = np.zeros_like(self.params["k"])
        O = grad.shape[-1]
        result = np.zeros((grad.shape[0], grad.shape[1], self.X.shape[-2], self.X.shape[-1]))
        for i in range(self.size):
            for j in range(self.size):
                result[..., i:O+i, j:O+j] += self.params["k"][:, i, j].reshape((1, -1, 1, 1)) * grad
                self.grad["k"][:, i, j] = np.sum(grad*self.X[..., i:i+O, j:j+O], axis=(0, 2, 3)) / grad.shape[0]
        if self.pad is not None:
            t = self.size // 2
            result = result[..., t:-t, t:-t]
        if self.multichannel:
            result = np.sum(result, axis=1)
        return result

    def __str__(self):
        return f"Conv2d{(self.channel, self.size, self.size)}"


class Pooling(Module):
    def __init__(self, pool_kernel=(2, 2), method="max", **kwargs):
        if method != "max":
            raise NotImplementedError
        self.k = pool_kernel
        self.m = method
        self.mask = None
        self.shape = None
        self.stride = None

    def forward(self, X):
        self.shape = X.shape
        B, C, H, W = X.shape
        H_out, W_out = H // self.k[0], W // self.k[1]
        self.stride = X.strides
        stride = X.strides
        new_shape = (B, C, H_out, self.k[0], W_out, self.k[1])
        new_stride = (stride[0], stride[1], stride[2]*self.k[0], stride[2], stride[3]*self.k[1], stride[3])
        x_view = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_stride)
        res = np.max(x_view, axis=(3, 5))
        self.mask = res[..., None, :, None] == x_view
        return np.max(x_view, axis=(3, 5))

    def backward(self, grad):
        g = np.tile(grad[..., None, :, None], (1, 1, 1, self.k[0], 1, self.k[1]))
        res = g * np.where(self.mask, 1, 0)
        return np.lib.stride_tricks.as_strided(res, shape=self.shape, strides=self.stride)

    def __str__(self):
        return f"{self.m.capitalize()} Pooling{self.k}"


# Loss function
class MultiCrossEntropyLoss:
    def __init__(self, thre=1e3):
        self.predicts = None
        self.labels = None
        self.num = None
        self.threshold = thre

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
            predicts：[N, C]
            labels：[N, 1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]
        loss = 0
        valid_pre = self.predicts[list(range(self.num)), (self.labels-1).reshape([-1]).tolist()]
        loss = -np.mean(np.log(valid_pre))
        return loss

    def backward(self):
        K = np.eye(self.predicts.shape[1])[(self.labels-1).flatten()]
        return np.clip(K / (1e-12 + self.predicts) / (-self.labels.shape[0]), -self.threshold, 0)


class CEwithSoftmax:
    def __init__(self, thre=1e3):
        self.predicts = None
        self.labels = None
        self.num = None
        self.threshold = thre
        self.sm = Softmax()

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
            predicts：[N, C]
            labels：[N, D]
        """
        self.gt = labels
        y_shift = predicts - np.max(predicts, axis=-1, keepdims=True)
        y_exp = np.exp(y_shift)
        self.prob = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
        loss = np.mean(np.sum(-labels * np.log(self.prob), axis=-1))
        return loss

    def backward(self):
        return self.prob - self.gt
