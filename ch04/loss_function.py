import numpy as np

def sum_squares_error(y, t):
    error = 0.5*np.sum((y-t)**2)
    return error

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    error = -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size
    #(one-hot인코딩 가정) 예측값인 y의 각 행에서 t열에 해당하는 확률만을 추출
    return error