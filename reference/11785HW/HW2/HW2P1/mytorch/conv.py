# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D:
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        batch_size, _, input_size = x.shape
        output_size = (input_size - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, self.out_channel, output_size))
        for b in range(batch_size):
            for oc in range(self.out_channel):
                for i in range(output_size):
                    start = i * self.stride
                    end = start + self.kernel_size
                    window = x[b, :, start:end]
                    out[b, oc, i] = np.sum(window * self.W[oc]) + self.b[oc]
        self.output = out
        return self.output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        x = self.x
        batch_size, _, input_size = x.shape  # type: ignore
        output_size = delta.shape[2]
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dx = np.zeros((batch_size, self.in_channel, input_size))  # type: ignore
        for b in range(batch_size):
            for oc in range(self.out_channel):
                for i in range(output_size):
                    start = i * self.stride
                    end = start + self.kernel_size
                    grad = delta[b, oc, i]
                    dx[b, :, start:end] += grad * self.W[oc]
                    self.dW[oc] += grad * x[b, :, start:end]  # type: ignore
                    self.db[oc] += grad
        return dx


class Conv2D:
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        weight_init_fn=None,
        bias_init_fn=None,
    ):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channel, in_channel, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size, _, input_width, input_height = x.shape
        self.input_width = input_width
        self.input_height = input_height
        output_width = (input_width - self.kernel_size) // self.stride + 1
        output_height = output_width
        out = np.zeros((batch_size, self.out_channel, output_width, output_height))
        for b in range(batch_size):
            for oc in range(self.out_channel):
                for i in range(output_width):
                    for j in range(output_height):
                        start_w = i * self.stride
                        end_w = start_w + self.kernel_size
                        start_h = j * self.stride
                        end_h = start_h + self.kernel_size
                        region = x[b, :, start_w:end_w, start_h:end_h]
                        out[b, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        x = self.x
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        batch_size, _, output_width, output_height = delta.shape
        dx = np.zeros(
            (batch_size, self.in_channel, self.input_width, self.input_height)
        )
        for b in range(batch_size):
            for oc in range(self.out_channel):
                for i in range(output_width):
                    for j in range(output_height):
                        grad = delta[b, oc, i, j]
                        start_w = i * self.stride
                        end_w = start_w + self.kernel_size
                        start_h = j * self.stride
                        end_h = start_h + self.kernel_size
                        dx[b, :, start_w:end_w, start_h:end_h] += grad * self.W[oc]
                        self.dW[oc] += grad * x[b, :, start_w:end_w, start_h:end_h]  # type: ignore
                        self.db[oc] += grad
        return dx


class Flatten:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c * self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)
