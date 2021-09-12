from matplotlib import pyplot as plt
import numpy as np
from activationFunc import (step_func, sigmoid, ReLU)
from basic_neural_network import neural_network
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from helper import img_show
from ch03.basic_mnist_predictor import mnist_predictor

if __name__ == '__main__':
    # n_network = neural_network()
    # y = n_network.forward()
    # print(n_network.x, y)

    (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

    predictor = mnist_predictor()
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(predictor.x), batch_size):
        predicted_val = predictor.predict(predictor.x[i:i+batch_size])
        predicted_val = np.argmax(predicted_val, axis=1)
        accuracy_cnt += np.sum(predicted_val == predictor.t[i:i+batch_size])
    print(accuracy_cnt/len(predictor.x))









