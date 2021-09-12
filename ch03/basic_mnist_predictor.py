import os
import pickle
from dataset.mnist import load_mnist
import numpy as np
from activationFunc import (sigmoid, softmax)

class mnist_predictor:

    def __init__(self):
        with open(os.getcwd()+'\ch03'+"\sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)
        self.network = network
        self.x, self.t = self.get_data()

    def get_data(self):
        (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test

    def predict(self, x):
        a1 = np.dot(x, self.network['W1']) + self.network['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.network['W2']) + self.network['b2']
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.network['W3']) + self.network['b3']
        y = softmax(a3)
        return y

