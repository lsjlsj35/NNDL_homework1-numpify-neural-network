from data import load, NormalizePolicy
from model import NNClassifier, CNNModel, MLPModel
import numpy as np

Normalization_Method = "null"
BASE_PATH = "mlp_"

Xt, yt, X, y, Xv, yv = load()

# one-hot
yt = np.eye(10)[(yt-1).flatten()]
y = np.eye(10)[(y-1).flatten()]
yv = np.eye(10)[(yv-1).flatten()]
# normalize
Xt, X, Xv = getattr(NormalizePolicy(), Normalization_Method)(Xt, X, Xv)

overall_best_acc = 0.0
best_path = ""
idx = 0

for lr in [0.05, 0.2, 1]:
    for bs in [1024, 256]:
        for ld in [0.99, 0.95]:
            PATH = BASE_PATH + str(idx)
            idx += 1
            CONFIG = {
                "lr": lr,
                "batch_size": bs,
                "num_epochs": 30,
                "optim_method": "null",
                "momentum_param": 0.9,
                "reg": 0,
                "lr_decay": ld
            }

            # convChannel = 16
            MODEL_CONFIG = {
                # "linearSize": [convChannel * 16, 512, 10],
                "linearSize": [3072, 512, 256, 10],
                "normalize": False,
                "activation": "LeakyReLU",
                "eps": 0.01,
                # "convChannel": convChannel,
                "dropout_rate": 0.05
            }

            LOG = {
                "path": PATH,
                "normalization": Normalization_Method
            }

            LOG.update(CONFIG)
            LOG.update(MODEL_CONFIG)

            mlp = NNClassifier(Model=MLPModel(), **CONFIG)
            mlp.set_model(**MODEL_CONFIG)
            best_acc = mlp.fit(X, y, Xv, yv, Xt, yt, log=LOG, do_test=False)["best_acc"]
            mlp.plot_loss(log=LOG, imshow=False)
            mlp.save(log=LOG)

            if best_acc > overall_best_acc:
                overall_best_acc = best_acc
                best_path = f'result/{LOG["path"]}_best_param.json'
                print(f"Update best path. Now: {overall_best_acc}")
            else:
                print(f"acc: {overall_best_acc}")


mlp.load(path=best_path)
mlp.predict(Xt, yt)
