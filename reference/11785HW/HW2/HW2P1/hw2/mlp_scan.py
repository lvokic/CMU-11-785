# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append("mytorch")
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP:
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24, 8, kernel_size=8, stride=4)
        self.conv2 = Conv1D(8, 16, kernel_size=1, stride=1)
        self.conv3 = Conv1D(16, 4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        w1, w2, w3 = weights
        # MLP weights are (input_features, output_neurons), while Conv1D
        # weights are (out_channel, in_channel, kernel_size).
        self.conv1.W = w1.T.reshape(8, 8, 24).transpose(0, 2, 1)
        self.conv2.W = w2.T.reshape(16, 1, 8).transpose(0, 2, 1)
        self.conv3.W = w3.T.reshape(4, 1, 16).transpose(0, 2, 1)

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP:
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # The distributed MLP reuses two first-layer neuron types over
        # adjacent pairs of frames.  The next layer combines two such
        # positions using eight shared filters, and the final layer combines
        # two adjacent positions into the four outputs of the MLP.
        self.conv1 = Conv1D(24, 2, kernel_size=2, stride=2)
        self.conv2 = Conv1D(2, 8, kernel_size=2, stride=2)
        self.conv3 = Conv1D(8, 4, kernel_size=2, stride=1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten(),
        ]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        # The MLP matrices are stored as (input_features, output_neurons),
        # whereas Conv1D expects (out_channel, in_channel, kernel_size).
        # Select one copy of each shared parameter group before reshaping.
        self.conv1.W = w1[:48, :2].T.reshape(2, 2, 24).transpose(0, 2, 1)
        self.conv2.W = w2[:4, :8].T.reshape(8, 2, 2).transpose(0, 2, 1)
        self.conv3.W = w3.T.reshape(4, 2, 8).transpose(0, 2, 1)

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
