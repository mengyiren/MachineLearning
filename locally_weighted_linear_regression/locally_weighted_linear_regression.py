from numpy import *
import numpy.linalg as la
import numpy.matlib as matlib

'''
    局部加权回归theta = (x.T*w*x)^-1*x.T*w*y 只是一个示意，并不是矩阵的运算
'''


def predict(x_input, y_input, x, tau):
    x_t = mat(x_input)
    y_t = mat(y_input)
    target = mat(x)
    m, n = x_t.shape
    w = get_weight(x_t, m, target, tau)
    theta = la.inv(x_t.transpose() * diagflat(w) * x_t) * x_t.transpose() * diagflat(w) * y_t
    return target * theta


def get_weight(x, m, target, tau):
    diff = x - matlib.repmat(target, m, 1)
    # 将每一行的权重系数相加，求出训练数据的权重系数
    return exp(sum(multiply(diff, diff), 1) / (-2 * tau ** 2))


def load_data(file_name, row_start, row_end, start, end):
    A = zeros((row_end - row_start + 1, end - start + 1))
    row = 0
    count = 0
    with open(file_name, 'rb') as fd:
        for line in fd.readlines():
            data = line.strip('\r\n'.encode()).split()
            count += 1
            if row_start <= count <= row_end:
                A[row:] = data[start:end + 1]
                row += 1
    return A


if __name__ == '__main__':
    # 获取从第0行到第1500行，第0列到第7列的数据作为训练输入数据
    x_train = load_data('train/quasar_train.txt', 0, 1500, 0, 7)
    # 获取从第0行到第1500行，第8列的数据作为训练输出数据
    y_train = load_data('train/quasar_train.txt', 0, 1500, 8, 8)
    x_test = load_data('train/quasar_train.txt', 1501, 1815, 0, 7)
    y_test = load_data('train/quasar_train.txt', 1501, 1815, 8, 8)
    print('预测值：' + str(predict(x_train, y_train, x_test[6], 3)))
    print('实际值：' + str(y_test[6]))
