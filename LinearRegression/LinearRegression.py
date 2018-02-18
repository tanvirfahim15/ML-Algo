import numpy as np


class LinearRegression:
    alpha = float(0)
    theta = np.array([0.0, 0.0, 0.0])
    x = np.array([[0.0, 0.0], [0.0, 0.0]])
    y = np.array([0.0, 0.0])

    def __init__(self, _theta, _x, _y, _alpha):
        _x = np.append(np.ones((_x.__len__(), 1)), _x, axis=1)
        if _x.__len__() != _y.__len__():
            raise ValueError('Incompatible matrix size : x y')
        if _x.shape[1] != _theta.__len__():
            raise ValueError('Incompatible matrix size: x theta')
        if type(_alpha) is not float:
            raise ValueError('_alpha should be float')
        self.theta = _theta
        self.x = _x
        self.y = _y
        self.alpha = _alpha

    def prediction(self):
        return np.matmul(self.x,self.theta)

    def predict(self, _x):
        _x = np.append(np.ones((_x.__len__(), 1)), _x, axis=1)
        return np.matmul(_x, self.theta)

    def cost(self):
        m=self.y.__len__()
        return (np.matmul(np.ones((1, m)), ((self.prediction()-self.y)**2))/(m*2))[0]

    def gradient_decent(self):
        m = self.y.__len__()
        one = np.ones((1, m))*(self.alpha/m)
        off = np.matmul(one, np.multiply(self.x, np.reshape((self.prediction()-self.y), (-1, 1)))).flatten()
        self.theta = self.theta-off

    def get_theta(self):
        return self.theta
