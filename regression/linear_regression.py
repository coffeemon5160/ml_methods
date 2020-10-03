
import numpy as np


class LinearRegression():
    def __init__(self, lr, epoch):
        self.lr = lr
        self.epoch = epoch

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        学習用関数

        Args:
            x (np.ndarray): 学習用の説明変数の配列
            y (np.ndarray): 学習用のターゲットの配列
        """
        # FIXME: もうちょっとよい変形方法
        X = np.vstack([np.ones(len(x)), x]).T

        self.w = np.random.random(len(X[0]))

        for _ in range(self.epoch):
            y_pred = np.dot(X, self.w)

            dw = np.dot((y - y_pred), X) / len(X)
            self.w += self.lr * dw


if __name__ == '__main__':
    x = np.random.random(1000)

    y = x * 5 + 3

    linear_regression = LinearRegression(lr=0.1, epoch=1000)
    linear_regression.fit(x, y)
    print(linear_regression.w[0], linear_regression.w[1])
