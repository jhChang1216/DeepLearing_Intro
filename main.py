import numpy as np
from ch05.MulLayer import MulLayer
from ch05.AddLayer import AddLayer
from ch05.TwoLayerNet import TwoLayerNet
from dataset.mnist import load_mnist
from matplotlib import pyplot as plt
from common import optimizer
from ch06.dropout import Dropout
from ch06.Trainer import Trainer

if __name__ == '__main__':

    optimizers = {}
    optimizers['SGD'] = optimizer.SGD()
    # optimizers['Momentum'] = optimizer.Momentum()
    # optimizers['Adagrad'] = optimizer.AdaGrad()
    # optimizers['Adam'] = optimizer.Adam()
    #
    # train_loss_list = {}
    # train_loss_list['SGD'] = []
    # train_loss_list['Momentum'] = []
    # train_loss_list['Adagrad'] = []
    # train_loss_list['Adam'] = []
    #
    # train_acc_list = []
    # test_acc_list = []
    #
    #
    # for key in optimizers:
    #     predictor = TwoLayerNet(784, 50, 10, l2=False, dropout=True)
    #
    #     (x_train, t_train), (x_test, t_test) = \
    #         load_mnist(normalize=True, one_hot_label=True)
    #
    #     iternum = 10000
    #     train_size = x_train.shape[0]
    #     batch_size = 100
    #     learning_rate = 0.1
    #
    #     iter_per_epoch = max(train_size / batch_size, 1)
    #
    #     for i in range(iternum):
    #         batch_mask = np.random.choice(train_size, batch_size)
    #         x_batch = x_train[batch_mask]
    #         t_batch = t_train[batch_mask]
    #         grads = predictor.gradient(x_batch, t_batch)
    #         optimizers[key].update(predictor.params, grads)
    #         loss = predictor.loss(x_batch, t_batch)
    #         train_loss_list[key].append(loss)
    #
    #         if i % iter_per_epoch == 0:
    #             train_acc = predictor.accuracy(x_train, t_train)
    #             test_acc = predictor.accuracy(x_test, t_test)
    #             train_acc_list.append(train_acc)
    #             test_acc_list.append(test_acc)
    #             print(train_acc, test_acc)
    #
    # loss_array = {}
    #
    # for key in train_loss_list:
    #     loss_array[key] = np.array(train_loss_list[key])
    #     print(key," : ",np.mean(loss_array[key]))
    #
    # plt.plot(train_acc_list, 'b-', label = 'train accuracy')
    # plt.plot(test_acc_list, 'g--', label = 'test accuracy')
    # plt.ylim(0.85, 0.95)
    # plt.title('accuracy in train VS accuracy in test')
    # plt.xlabel('iter')
    # plt.ylabel('accuracy')
    # plt.legend(loc='best')
    # plt.grid(linestyle='dotted')
    # plt.show()
    #
    # train_acc_list_np = np.array(train_acc_list)
    # test_acc_list_np = np.array(test_acc_list)

    ####결과값
    #SGD  :  0.5394052491776588
    #Momentum  :  0.1710974149078524
    #Adagrad  :  0.20954943225360934
    #Adam  :  0.12610166012684842

    trainer = Trainer(optimizers, TwoLayerNet)
    trainer.train(iternum=10000, Xavier=False, He=False, l2=False, dropout=False, l1=True)

    comparison = 'l1'

    plt.subplot(1,2,1)
    trainer.display_acc_chart(['train_acc('+comparison+')', 'test_acc('+comparison+')'])
    plt.subplot(1,2,2)
    trainer.display_loss_chart('train_loss('+comparison+')')

    regulation = 'l2'

    trainer = Trainer(optimizers, TwoLayerNet)
    trainer.train(iternum=10000, Xavier=False, He=True, l2=True, dropout=False)
    plt.subplot(1,2,1)
    trainer.display_acc_chart(['train_acc('+regulation+')', 'test_acc('+regulation+')'])
    plt.subplot(1,2,2)
    trainer.display_loss_chart('train_loss('+regulation+')')
    plt.show()



































