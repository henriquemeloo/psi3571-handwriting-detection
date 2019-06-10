import os

from matplotlib import pyplot as plt
from mnist import MNIST
import numpy as np


def load_train_and_test_dataset(data_path='data/gzip'):
    data = MNIST(data_path)

    X_train, y_train = data.load(
        os.path.join(data_path, 'emnist-byclass-train-images-idx3-ubyte'),
        os.path.join(data_path, 'emnist-byclass-train-labels-idx1-ubyte'))

    X_test, y_test = data.load(
        os.path.join(data_path, 'emnist-byclass-test-images-idx3-ubyte'),
        os.path.join(data_path, 'emnist-byclass-test-labels-idx1-ubyte'))

    # Normalização
    X_train = np.array(X_train) / 255.0
    y_train = np.array(y_train)
    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)

    # Formatação
    X_train = X_train.reshape(X_train.shape[0], 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 28, 28)

    X_train = X_train.reshape(X_train.shape[0], 784, 1)
    X_test = X_test.reshape(X_test.shape[0], 784, 1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def display_random_image(data):
    #Display a random image
    plt.imshow(np.random.randint(len(data["X_train"])))
    plt.show()
