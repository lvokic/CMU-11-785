"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

from mytorch.loss import *
from mytorch.activation import *
from mytorch.batchnorm import *
from mytorch.linear import *

from typing import List


class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(
        self,
        input_size,
        output_size,
        hiddens,
        activations: List[Activation],
        weight_init_fn,
        bias_init_fn,
        criterion,
        lr,
        momentum=0.0,
        num_bn_layers=0,
    ):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = []
        # Write your code here.
        layer_sizes = [input_size] + list(hiddens) + [output_size]
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.linear_layers.append(
                Linear(in_dim, out_dim, weight_init_fn, bias_init_fn)
            )
        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        self.bn_layers = []
        # Write your code here.
        if self.num_bn_layers > 0:
            assert self.num_bn_layers <= len(hiddens)
            self.bn_layers = [BatchNorm(hiddens[i]) for i in range(self.num_bn_layers)]

        self.output = None

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        for i, linear in enumerate(self.linear_layers):
            x = linear.forward(x)
            if i < self.num_bn_layers:
                x = self.bn_layers[i](x, eval=not self.train_mode)
            x = self.activations[i](x)
        self.output = x
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for linear in self.linear_layers:
            linear.dW.fill(0.0)
            linear.db.fill(0.0)
        for bn_layer in self.bn_layers:
            bn_layer.dgamma.fill(0.0)
            bn_layer.dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)
        if self.momentum == 0.0:
            for linear in self.linear_layers:
                linear.W -= self.lr * linear.dW
                linear.b -= self.lr * linear.db
        else:
            for linear in self.linear_layers:
                linear.momentum_W = (
                    self.momentum * linear.momentum_W - self.lr * linear.dW
                )
                linear.W += linear.momentum_W
                linear.momentum_b = (
                    self.momentum * linear.momentum_b - self.lr * linear.db
                )
                linear.b += linear.momentum_b

        for bn_layer in self.bn_layers:
            bn_layer.gamma -= self.lr * bn_layer.dgamma
            bn_layer.beta -= self.lr * bn_layer.dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        loss = self.total_loss(labels=labels)
        gradient = self.criterion.derivative()
        for i in reversed(range(len(self.linear_layers))):
            gradient = gradient * self.activations[i].derivative()
            if i < self.num_bn_layers:
                gradient = self.bn_layers[i].backward(gradient)
            gradient = self.linear_layers[i].backward(gradient)
        return loss

    def error(self, labels):
        return (np.argmax(self.output, axis=1) != np.argmax(labels, axis=1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


# This function does not carry any points. You can try and complete this function to train your
# network.
def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    n_train = len(trainx)
    n_val = len(valx)

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    for e in range(nepochs):
        idxs = np.arange(n_train)
        np.random.shuffle(idxs)

        mlp.train()
        train_loss_sum = 0.0
        train_error_sum = 0
        # Per epoch setup ...
        for b in range(0, len(trainx), batch_size):
            # Train ...
            batch_idxs = idxs[b : b + batch_size]
            batch_x = trainx[batch_idxs]
            batch_y = trainy[batch_idxs]
            mlp.forward(batch_x)
            mlp.zero_grads()
            batch_loss = mlp.backward(batch_y)
            mlp.step()
            train_loss_sum += batch_loss
            train_error_sum += mlp.error(batch_y)

        training_losses[e] = train_loss_sum / n_train
        training_errors[e] = train_error_sum / n_train

        mlp.eval()
        val_loss_sum = 0.0
        val_error_sum = 0
        for b in range(0, n_val, batch_size):
            batch_x = valx[b : b + batch_size]
            batch_y = valy[b : b + batch_size]
            mlp.forward(batch_x)
            val_loss_sum += mlp.total_loss(batch_y)
            val_error_sum += mlp.error(batch_y)

        validation_losses[e] = val_loss_sum / n_val
        validation_errors[e] = val_error_sum / n_val

        print(
            f"Epoch {e + 1}: "
            f"train_loss={training_losses[e]:.4f}, "
            f"train_error={training_errors[e]:.4f}, "
            f"val_loss={validation_losses[e]:.4f}, "
            f"val_error={validation_errors[e]:.4f}"
        )

    # Cleanup ...

    # Return results ...
    return (training_losses, training_errors, validation_losses, validation_errors)
