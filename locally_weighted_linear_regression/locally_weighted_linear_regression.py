import pandas as pd

if __name__ == '__main__':
    x_train = pd.read_csv('train/quasar_train.csv')
    y_train = pd.read_csv('test/quasar_test.csv')