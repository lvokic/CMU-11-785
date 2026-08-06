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
        batch_size, in_channel, self.in_w, self.in_h = x.shape
        out_w = (self.in_w - self.kernel) // self.stride + 1
        out_h = (self.in_h - self.kernel) // self.stride + 1
        out = np.zeros((batch_size, in_channel, out_w, out_h))
        self.used = np.zeros(
            (batch_size, in_channel, out_w, out_h, self.kernel, self.kernel),
            dtype=bool,
        )
        for b in range(batch_size):
            for ic in range(in_channel):
                for i in range(out_w):
                    for j in range(out_h):
                        start_w = i * self.stride
                        end_w = start_w + self.kernel
                        start_h = j * self.stride
                        end_h = start_h + self.kernel
                        window = x[b, ic, start_w:end_w, start_h:end_h]
                        max_value = np.max(window)
                        out[b, ic, i, j] = max_value
                        self.used[b, ic, i, j] = window == max_value
        return out

    def backward(self, delta):
        batch_size, channel, out_w, out_h = delta.shape
        dx = np.zeros((batch_size, channel, self.in_w, self.in_h))  # type: ignore
        for b in range(batch_size):
            for c in range(channel):
                for i in range(out_w):
                    for j in range(out_h):
                        start_w = i * self.stride
                        start_h = j * self.stride
                        end_w = start_w + self.kernel
                        end_h = start_h + self.kernel
                        mask = self.used[b, c, i, j]  # type: ignore
                        dx[b, c, start_w:end_w, start_h:end_h][mask] = delta[b, c, i, j]
        return dx


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
        batch_size, in_channel, self.in_w, self.in_h = x.shape
        out_w = (self.in_w - self.kernel) // self.stride + 1
        out_h = (self.in_h - self.kernel) // self.stride + 1
        out = np.zeros((batch_size, in_channel, out_w, out_h))
        self.used = np.zeros_like(x)
        for b in range(batch_size):
            for ic in range(in_channel):
                for i in range(out_w):
                    for j in range(out_h):
                        start_w = i * self.stride
                        end_w = start_w + self.kernel
                        start_h = j * self.stride
                        end_h = start_h + self.kernel
                        window = x[b, ic, start_w:end_w, start_h:end_h]
                        mean_value = np.mean(window)
                        out[b, ic, i, j] = mean_value
        return out

    def backward(self, delta):
        batch_size, channel, out_w, out_h = delta.shape
        dx = np.zeros((batch_size, channel, self.in_w, self.in_h))  # type: ignore
        for b in range(batch_size):
            for c in range(channel):
                for i in range(out_w):
                    for j in range(out_h):
                        start_w = i * self.stride
                        start_h = j * self.stride
                        end_w = start_w + self.kernel
                        end_h = start_h + self.kernel
                        dx[b, c, start_w:end_w, start_h:end_h] += delta[b, c, i, j] / (
                            self.kernel**2
                        )
        return dx
