import matplotlib.pyplot as plt
import numpy as np
from model import NNClassifier, CNNModel, MLPModel


def visualize_first_layer_random(W, img_shape=(32, 32, 3), grid_shape=(16, 32)):
    assert W.shape[0] == np.prod(img_shape), "维度不匹配"
    fig, axes = plt.subplots(*grid_shape, figsize=(4*grid_shape[1], 4*grid_shape[0]))
    for i in range(grid_shape[0] * grid_shape[1]):
        ax = axes[i // grid_shape[1]][i % grid_shape[1]]
        img = W[:, i].reshape(img_shape)
        img = (img - img.min()) / (img.max() - img.min())
        img = 1.0 - img
        ax.imshow(img)
        ax.axis("off")
    # replace the last one
    img = np.mean(W, axis=1).reshape(img_shape)
    img = (img - img.min()) / (img.max() - img.min())
    img = 1.0 - img
    ax.imshow(img)
    plt.tight_layout()
    plt.savefig("RAND_VIS_PARAM.png")


MODEL_CONFIG = {
    # "linearSize": [convChannel * 16, 512, 10],
    "linearSize": [3072, 512, 256, 10],
    "normalize": False,
    "activation": "LeakyReLU",
    "eps": 0.01,
    # "convChannel": convChannel,
    "dropout_rate": 0.05
}

mlp = NNClassifier()
mlp.set_model(**MODEL_CONFIG)
mlp.load(path="result/mlp_3_best_param.json")
W = mlp.Model.modules[0].params["W"]
visualize_first_layer_random(W, grid_shape=(5, 4))
