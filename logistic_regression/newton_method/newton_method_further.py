"""
程序说明：
    使用牛顿算法计算回归系数
    当y值为-1,1时的牛顿方法
    对于y值为-1,1的逻辑回归，cost function 变为 J = log(1+exp(-yx))
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
    return 1.0 / (1 + exp(-x))


def newton_method(x, y):
    x = mat(x)
    y = mat(y)
    m, n = shape(x)
    # 牛顿方法中theta的取值不会影响y的值
    weigh = zeros((n, 1))
    grad = ones((n, 1))
    # 当梯度精度小于10E-6说明theta已经收敛
    while la.norm(grad) > 10E-6:
        # 计算假设函数，得到一个列向量，每行为那个样本属于1的概率
        h = sigmoid(-diagflat(y) * x * weigh)
        # 计算J对theta的一阶导数
        grad = -x.transpose() * diagflat(y) * h
        # 计算海森矩阵即J对theta的二阶导数
        H = x.T * diagflat(y) * diagflat(y) * diagflat(h) * diagflat(1 - h) * x
        # 迭代求出theta
        weigh = weigh - la.inv(H) * grad
    return weigh


def prediction(x):
    x = mat(x)
    x_train = load_data('train/logistic_x.txt', 2)
    y_train = load_data('train/logistic_y.txt', 1)
    theta = newton_method(x_train, y_train)
    # 输出y的预测值，也可以输出y的概率值
    if sigmoid(x * theta) > 0.5:
        return 1
    return -1


if __name__ == '__main__':
    print(prediction([8.6864964e-01,-2.4119348e+00]))
