import numpy as np
from ch05.MulLayer import MulLayer
from ch05.AddLayer import AddLayer
from ch05.TwoLayerNet import TwoLayerNet
from dataset.mnist import load_mnist
from matplotlib import pyplot as plt

def func1(x):
    return (x[0]-2)**2 + 4*(x[1]-3)**2

if __name__ == '__main__':
    # apple_price = 100
    # apple_num = 4
    # mandarin_price = 70
    # mandarin_num = 3
    # tax = 1.1
    #
    # # mul_apple_layer = MulLayer()
    # # mul_tax_layer = MulLayer()
    # # total_price = mul_apple_layer.forward(apple_price, apple_num)
    # # final_apple_pay = mul_tax_layer.forward(total_price, tax)
    # #
    # # print(final_apple_pay)
    # #
    # # d_total_price, d_tax = mul_tax_layer.backward(1)
    # # d_apple_price, d_apple_num = mul_apple_layer.backward(d_total_price)
    # # print(d_apple_price, d_apple_num, d_total_price, d_tax)
    #
    # mul_apple_layer = MulLayer()
    # mul_mandarin_layer = MulLayer()
    # add_price_Layer = AddLayer()
    # mul_tax_layer = MulLayer()
    #
    # total_apple_price = mul_apple_layer.forward(apple_price, apple_num)
    # total_mandarin_price = mul_mandarin_layer.forward(mandarin_price, mandarin_num)
    # total_fruit_price = add_price_Layer.forward(total_apple_price, total_mandarin_price)
    # final_price = mul_tax_layer.forward(total_fruit_price, tax)
    #
    # print(final_price)
    #
    # d_final_price = 1
    # d_total_fruit_price, d_tax = mul_tax_layer.backward(d_final_price)
    # d_total_apple_price = add_price_Layer.backward(d_total_fruit_price)
    # d_total_mandarin_price = add_price_Layer.backward(d_total_fruit_price)
    # d_apple_price, d_apple_num = mul_apple_layer.backward(d_total_apple_price)
    # d_mandarin_price, d_mandarin_num = mul_mandarin_layer.backward(d_total_mandarin_price)
    #
    # d_var_list = [d_apple_price, d_apple_num, d_mandarin_price, d_mandarin_num, d_total_apple_price\
    #               ,d_total_mandarin_price, d_total_fruit_price, d_tax]
    #
    # for i in range(len(d_var_list)):
    #     print(f"{d_var_list[i]}")

    predictor = TwoLayerNet(784, 50, 10)

    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

    iternum = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/batch_size, 1)

    for i in range(iternum):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grads = predictor.gradient(x_batch, t_batch)
        for key in ('W1', 'b1', 'W2', 'b2'):
            predictor.params[key] -= 0.01*grads[key]
        loss = predictor.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i%iter_per_epoch == 0:
            train_acc = predictor.accuracy(x_train, t_train)
            test_acc = predictor.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    plt.plot(train_acc_list, 'b-', label = 'train accuracy')
    plt.plot(test_acc_list, 'g--', label = 'test accuracy')
    plt.title('accuracy in train VS accuracy in test')
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.show()

    plt.plot(train_loss_list, label='loss')
    plt.title("predictor's losses in train data")
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.show()



























