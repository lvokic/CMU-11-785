# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):
        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * norm + self.beta
            return self.out

        self.x = x
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.var = np.mean((x - self.mean) ** 2, axis=0, keepdims=True)
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = self.gamma * self.norm + self.beta
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
        return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        if self.x is None or self.norm is None:
            raise RuntimeError("Call training forward() before backward()")

        x = self.x
        mean = self.mean
        var = self.var
        norm = self.norm
        batch_size = x.shape[0]

        centered = x - mean
        inv_std = (var + self.eps) ** (-0.5)

        derivative_x_hat = delta * self.gamma
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        self.dgamma = np.sum(delta * norm, axis=0, keepdims=True)

        derivative_var = -0.5 * np.sum(
            derivative_x_hat
            * centered
            * (var + self.eps) ** (-1.5),
            axis=0,
            keepdims=True
        )
        derivative_mean = (
            -np.sum(derivative_x_hat * inv_std, axis=0, keepdims=True)
            - (2.0 / batch_size)
            * derivative_var
            * np.sum(centered, axis=0, keepdims=True)
        )
        derivative = (
            derivative_x_hat * inv_std
            + derivative_var * (2.0 / batch_size) * centered
            + derivative_mean / batch_size
        )
        return derivative


if __name__ == '__main__':
    x = np.random.random((10, 2))

    bn = BatchNorm(2)

    y = bn(x)

    t = 1
