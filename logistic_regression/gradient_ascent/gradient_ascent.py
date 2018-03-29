"""
程序说明：
    使用梯度上升算法计算回归系数
"""

import numpy as np


def load_data(file_name, n):
    A = np.zeros((99, n))
    row = 0
    with open(file_name, 'rb') as fd:
        for line in fd.readlines():
            data = line.strip('\r\n'.encode()).split()
            A[row:] = data[0:n]
            row += 1
    return A


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 梯度上升
# alpha 步长 max 迭代次数
def gradient_ascent(x, y, alpha, max):
    x = np.mat(x)
    y = np.mat(y)
    m, n = np.shape(x)
    weigh = np.ones((n, 1))
    for i in range(max):
        h = sigmoid(x * weigh)
        weigh = weigh + alpha * x.transpose() * (y - h)
    return weigh


def prediction(x):
    x = np.mat(x)
    x_train = load_data('train/logistic_x.txt', 2)
    y_train = load_data('train/logistic_y.txt', 1)
    result = gradient_ascent(x_train, y_train, 0.8, 100)
    return sigmoid(x * result)


if __name__ == '__main__':
    print(prediction([1.3432504e+00,-1.3311479e+00]))
