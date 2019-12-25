import multi_tread_RegressionTree as tr

import numpy as np
import matplotlib.pyplot as plt

class GBDT(object):
    def __init__(self):
        self.tree = None
        self.init_val = None
        self.fn = lambda x: x

    def _get_residuals(self, y, y_hat):
        return [yi - self.fn(y_hat_i) for yi, y_hat_i in zip(y, y_hat)]

    def fit(self, X, y, lr=1, estimator=10):
        self.init_val = sum(y)/len(y)
        y_hat = [self.init_val]*len(y)
        residual = self._get_residuals(y, y_hat)

        self.tree = []
        self.lr = lr
        for i in range(estimator):
            sub_tree = tr.RTree()
            sub_tree.fit(X, residual)
            y_hat = [y_hat_i + self.lr * res_hat_i for y_hat_i, res_hat_i in zip(y_hat, sub_tree.predict(X))]
            residual = self._get_residuals(y, y_hat)
            self.tree.append(sub_tree)
        return 0

    def _predict(self, x):
        return self.fn(self.init_val+sum(self.lr * tree._predict(x) for tree in self.tree))

    def predict(self, X):
        return [self._predict(x) for x in X]


def gen_data(x1, x2):
    y = np.sin(x1) * 1 / 2 + np.cos(x2) * 1 / 2 + 0.1 * x1
    return y

def load_data():
    x1_train = np.linspace(0, 50, 600)
    x2_train = np.linspace(-10, 10, 600)
    data_train = [[x1, x2, gen_data(x1, x2) + np.random.random(1)[0] - 0.5] for x1, x2 in zip(x1_train, x2_train)]
    x1_test = np.linspace(0, 50, 100) + np.random.random(100) * 0.5
    x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
    data_test = [[x1, x2, gen_data(x1, x2)] for x1, x2 in zip(x1_test, x2_test)]
    return np.array(data_train), np.array(data_test)

def main():
    train, test = load_data()
    x_train, y_train = train[:, :2], train[:, 2]
    x_test, y_test = test[:, :2], test[:, 2]  # 同上，但这里的y没有噪声

    rt = GBDT()
    rt.fit(x_train, y_train, estimator=40)
    result = rt.predict(x_train)
    plt.plot(result)
    plt.plot(y_train)
    plt.show()

    return 0
if __name__ =='__main__':
    main()