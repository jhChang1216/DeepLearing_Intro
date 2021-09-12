import numpy as np

def step_func(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def ReLU(x):
    return np.maximum(0,x)

def identity_func(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    modified_x = exp_x / np.sum(exp_x)
    return modified_x