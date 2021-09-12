import numpy as np
from activationFunc import (sigmoid, identity_func)

class neural_network:

    def __init__(self):
        self.x = np.random.rand(1,2)
        self.network = {}
        self.network['W1'] = np.random.rand(2,3)
        self.network['b1'] = np.random.rand(1,3)
        self.network['W2'] = np.random.rand(3,2)
        self.network['b2'] = np.random.rand(1,2)
        self.network['W3'] = np.random.rand(2,2)
        self.network['b3'] = np.random.rand(1,2)

    def forward(self):
        a1 = np.dot(self.x, self.network['W1']) + self.network['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.network['W2']) + self.network['b2']
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.network['W3']) + self.network['b3']
        y = identity_func(a3)
        return y

