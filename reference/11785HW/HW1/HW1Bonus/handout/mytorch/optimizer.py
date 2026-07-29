# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

try:
    from hw1 import MLP
except ModuleNotFoundError:
    MLP = object

"""
In the linear.py file, attributes have been added to the Linear class to make
implementing Adam easier, check them out!

self.mW = np.zeros(None) #mean derivative for W
self.vW = np.zeros(None) #squared derivative for W
self.mb = np.zeros(None) #mean derivative for b
self.vb = np.zeros(None) #squared derivative for b
"""


class adam:
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0  # Number of Updates

    def step(self):
        """
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        """
        self.t += 1
        for linear in self.model.linear_layers:
            linear.mean_dW = self.beta1 * linear.mean_dW + (1 - self.beta1) * linear.dW
            linear.mean_db = self.beta1 * linear.mean_db + (1 - self.beta1) * linear.db

            linear.v_W = self.beta2 * linear.v_W + (1 - self.beta2) * ((linear.dW) ** 2)
            linear.v_b = self.beta2 * linear.v_b + (1 - self.beta2) * ((linear.db) ** 2)

            mean_tW = linear.mean_dW / (1 - self.beta1**self.t)
            var_tW = linear.v_W / (1 - self.beta2**self.t)
            mean_tb = linear.mean_db / (1 - self.beta1**self.t)
            var_tb = linear.v_b / (1 - self.beta2**self.t)

            linear.W = linear.W - (self.lr * mean_tW) / (np.sqrt(var_tW + self.eps))
            linear.b = linear.b - (self.lr * mean_tb) / (np.sqrt(var_tb + self.eps))
