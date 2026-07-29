# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os


def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """
    Log sum.
    :param x: (batch, M)
    :return: (batch,)
    """
    a = np.max(x, axis=1, keepdims=True)
    return a + np.log(np.sum(np.exp(x - a), axis=1, keepdims=True))


# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))


class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.prediction = None

    # 交叉熵：
    #   loss = -sum(y_c * log(p_c))
    #   Softmax 概率：
    #   p_c = exp(z_c) / sum(exp(z_j))
    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        log_sum = log_sum_exp(x)
        self.logits = x
        self.labels = y
        log_sigma = x - log_sum
        self.prediction = np.exp(log_sigma)
        self.loss = -np.sum(y * log_sigma, axis=1)
        return self.loss

    # derivative = prediction - labels
    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        return self.prediction - self.labels


if __name__ == "__main__":
    x = np.random.random((10, 2))
    y = log_sum_exp(x)
    t = 1
