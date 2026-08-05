import numpy as np


class MaxPoolLayer:
    """Max-pooling layer starter for the HW2P1 bonus."""

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.used = None
        self.in_w = None
        self.in_h = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        raise NotImplemented

    def backward(self, delta):
        raise NotImplemented


class MeanPoolLayer:
    """Mean-pooling layer starter for the HW2P1 bonus."""

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.in_w = None
        self.in_h = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def backward(self, delta):
        raise NotImplemented
