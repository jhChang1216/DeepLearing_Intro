import sys, os
sys.path.append(os.path)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from ch06.dropout import Dropout

weight_decay_lambda = 0.1

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, Xavier = False, He=False, l2=False, dropout=False, l1=False):
        self.params = {}
        self.l1 = l1
        self.l2 = l2

        if not Xavier and not He:
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
            self.params['b2'] = np.zeros(output_size)
        if Xavier:
            self.params['W1'] = (1 / np.sqrt(input_size)) * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = (1 / np.sqrt(hidden_size)) * np.random.randn(hidden_size, output_size)
            self.params['b2'] = np.zeros(output_size)
        if He:
            self.params['W1'] = (2 / np.sqrt(input_size)) * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = (2 / np.sqrt(hidden_size)) * np.random.randn(hidden_size, output_size)
            self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        if dropout:
            self.layers['Dropout'] = Dropout(0.1)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)

        if self.l1:
            weight_decay = 0
            for idx in range(1, 3):
                W = self.params['W' + str(idx)]
                weight_decay += 0.5 * weight_decay_lambda * np.sum(np.abs(W))
            return self.lastlayer.forward(y, t) + weight_decay

        if self.l2:
            weight_decay = 0
            for idx in range(1, 3):
                W = self.params['W' + str(idx)]
                weight_decay += 0.5 * weight_decay_lambda * np.sum(W ** 2)
            return self.lastlayer.forward(y, t) + weight_decay

        y = self.predict(x)
        return self.lastlayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # score로 표현된 y값에서 가장 대표(큰) 값의 인덱스를 뽑아냄.
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        # 타깃이 한 데이터만 갖는게 아니라면, t에도 마찬가지로 대표값 뽑는다.
        accuracy = np.sum(y == t)/float(x.shape[0])
        # y와 t가 일치하는 횟수를 count하고 배치 수로 나눠 평균 get
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)
        # 함수 W에 대해서 선언.
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


    def gradient(self, x, t):

        self.loss(x, t)

        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads



