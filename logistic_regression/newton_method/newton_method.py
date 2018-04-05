"""
程序说明：
    使用牛顿算法计算回归系数
    当y值为0,1时的牛顿方法
"""

from numpy import *
from numpy import linalg as la


def load_data(file_name, n):
    A = zeros((99, n))
    row = 0
    with open(file_name, 'rb') as fd:
        for line in fd.readlines():
            data = line.strip('\r\n'.encode()).split()
            A[row:] = data[0:n]
            row += 1
    return A


def sigmoid(x):
    return 2.0 / (1 + exp(-x)) - ones((len(x), 1))


def newton_method(x, y, max):
    x = mat(x)
    y = mat(y)
    m, n = shape(x)
    weigh = zeros((n, 1))
    while max > 0:
        # 计算假设函数，得到一个列向量，每行为那个样本属于1的概率
        h = sigmoid(x * weigh)
        # 计算J对theta的一阶导数
        grad = x.transpose() * (h - y)
        # 计算海森矩阵即J对theta的二阶导数
        H = x.T * diagflat(h) * diagflat(1 - h) * x
        # 迭代求出theta
        weigh = weigh - la.inv(H) * grad
        max -= max
    return weigh


def prediction(x):
    x = mat(x)
    x_train = load_data('train/logistic_x.txt', 2)
    y_train = load_data('train/logistic_y.txt', 1)
    theta = newton_method(x_train, y_train)
    return sigmoid(x * theta)


if __name__ == '__main__':
    print(prediction([2.7458671e+00, -2.7100561e+00]))
