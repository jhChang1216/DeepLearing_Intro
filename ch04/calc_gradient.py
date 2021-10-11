import numpy as np
h = 1e-4

def numerical_diff(f, x):
    return (f(x+h)-f(x-h))/(2*h)

def numerical_gradient(f, x):
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]  #x[i]의 원래값을 저장해둔다.
        x[i] = tmp_val + h
        fxh1 = f(x) #x[i]가 약간 증가한 형태일 때의 f값을 계산

        x[i] = tmp_val - h
        fxh2 = f(x) #x[i]가 약간 감소한 형태일 때의 f값을 계산

        grad[i] = (fxh1-fxh2)/(2*h) #x[i]만 바뀌었을 때의 기울기 계산
        x[i] = tmp_val
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x
