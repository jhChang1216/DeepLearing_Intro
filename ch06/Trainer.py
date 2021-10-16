import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from common import optimizer
from ch06.dropout import Dropout


class Trainer:

    def __init__(self, optimizer, network):
        self.optimizer = optimizer
        self.network = network

        self.result = {}

    def train(self, iternum, Xavier = False, He=False, l2=False, dropout=False, l1=False):

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        predictor = self.network(784, 50, 10, Xavier=Xavier, He = He, l2=l2, dropout=dropout, l1=l1)

        (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, one_hot_label=True)

        iternum = 10000
        train_size = x_train.shape[0]
        batch_size = 100

        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iternum):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            grads = predictor.gradient(x_batch, t_batch)
            self.optimizer['SGD'].update(predictor.params, grads)
            loss = predictor.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = predictor.accuracy(x_train, t_train)
                test_acc = predictor.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print(train_acc, test_acc)

        self.result['train_loss_list'] = train_loss_list
        self.result['train_acc_list'] = train_acc_list
        self.result['test_acc_list'] = test_acc_list

    def display_loss_chart(self, label):
        train_loss_list = self.result['train_loss_list']

        plt.plot(train_loss_list, label = label)
        plt.title('train_loss_list')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.grid(linestyle='dotted')

    def display_acc_chart(self, label_list):
        train_acc_list, test_acc_list = self.result['train_acc_list'], self.result['test_acc_list']

        plt.plot(train_acc_list, label=label_list[0])
        plt.plot(test_acc_list, label=label_list[1])
        plt.ylim(0.85, 0.95)
        plt.title('accuracy in train VS accuracy in test')
        plt.xlabel('iter')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.grid(linestyle='dotted')


