import sys
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

def multi_map(fun, param, thread=20):
    pool = Pool(thread)
    result = pool.starmap(fun, param)
    pool.close()
    pool.join()
    return result

class Node(object):
    def __init__(self, left=None, right=None, score=0, split=None):
        self.avg = score
        self.left = left
        self.right = right
        self.feature = None
        self.split = split


class RTree(object):
    def __init__(self):
        self.root = Node()

    def mse(self,x, y, split):
        spt_sum = [0,0]
        spt_cnt = [0,0]
        spt_sqrt = [0,0]

        for index, xi in enumerate(x):
            if xi < split:
                i = 0
            else:
                i = 1
            spt_cnt[i] += 1
            spt_sum[i] += y[index]
            spt_sqrt[i] += pow(y[index],2)

        spt_avg = [spt_sum[0]/spt_cnt[0],
                   spt_sum[1]/spt_cnt[1]]
        spt_mse = [spt_sqrt[0] - spt_sum[0]*spt_avg[0],
                   spt_sqrt[1] - spt_sum[1]*spt_avg[1]]
        return sum(spt_mse), spt_avg, split

    def best_split(self, x, y, feat):
        uniq = list(set(x))
        if len(uniq) == 1:
            return None
        uniq.remove(min(uniq))

        mse, avg, split = min(multi_map(self.mse, zip([x]*len(uniq), [y]*len(uniq), uniq)), key=lambda d:d[0])
        # mse, avg, split = min(list(map(lambda split: self.mse(x,y,split), uniq)), key=lambda d:d[0])
        return mse, avg, split, feat

    def best_feat(self, X, Y, idx):
        x = [X[i] for i in idx]
        y = [Y[i] for i in idx]

        param = list(map(list, zip(*x)))
        set = [x for x in map(lambda x, feat: self.best_split(x, y, feat), param, range(len(param))) if x is not None]
        if set == []:
            return None
        mse, avg, split, feature = min(set, key=lambda d:d[0])
        X = list(map(list, zip(*X)))
        idx = [[i for i, x in enumerate(X[feature]) if x < split],
               [i for i, x in enumerate(X[feature]) if x >= split]]
        return mse, feature, avg, split, idx

    def fit(self, X, y, max_depth=1, min_sample=2):
        que = [[0,self.root,list(range(len(y)))]]
        while (len(que)>0):
            depth, node, idx = que.pop(0)
            if depth >= max_depth:
                break;
            if len(idx)<min_sample:
                continue
            feature_set = self.best_feat(X, y, idx)
            if feature_set is None:
                continue
            mse, node.feature, avg, node.split, idx = feature_set
            node.left = Node(score=avg[0])
            node.right = Node(score=avg[1])
            que.append([depth+1, node.left, idx[0]])
            que.append([depth+1, node.right, idx[1]])
        return 0

    def _predict(self, x):
        node = self.root
        while node.left and node.right:
            if x[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
        return node.avg
    def predict(self, X):
        return [self._predict(i) for i in X]

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

    rt = RTree()
    rt.fit(x_train, y_train)
    result = rt.predict(x_train)
    plt.plot(result)
    plt.plot(y_train)
    plt.show()

    return 0
if __name__ =='__main__':
    # def test(x,y):
    #     print(y)
    #     return 1
    # param = zip([1,2,3,4],[1]*4)
    # print(multi_map(fun=test, param=param, thread=2))
    main()