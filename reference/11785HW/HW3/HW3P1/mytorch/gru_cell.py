import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r = self.r_act(self.Wrx @ self.x + self.bir + self.Wrh @ h + self.bhr)
        self.z = self.z_act(self.Wzx @ self.x + self.biz + self.Wzh @ h + self.bhz)
        self.n = self.h_act(
            self.Wnx @ self.x + self.bin + self.r * (self.Wnh @ h + self.bhn)
        )
        h_t = (1 - self.z) * self.n + self.z * h

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        # return h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # Keep the cached vectors one-dimensional for elementwise derivatives.
        # Use column views only when forming outer products for parameter gradients.
        delta = np.asarray(delta).reshape(self.h)

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        x_col = self.x.reshape(self.d, 1)
        h_col = self.hidden.reshape(self.h, 1)

        d_loss_n = delta * (1 - self.z) * self.h_act.derivative(state=self.n)
        d_loss_z = delta * (self.hidden - self.n) * self.z * (1 - self.z)
        d_loss_r = (
            d_loss_n
            * (self.Wnh @ self.hidden + self.bhn)
            * self.r
            * (1 - self.r)
        )

        self.dWrx += d_loss_r.reshape(self.h, 1) @ x_col.T
        self.dbir += d_loss_r
        self.dWrh += d_loss_r.reshape(self.h, 1) @ h_col.T
        self.dbhr += d_loss_r

        self.dWzx += d_loss_z.reshape(self.h, 1) @ x_col.T
        self.dbiz += d_loss_z
        self.dWzh += d_loss_z.reshape(self.h, 1) @ h_col.T
        self.dbhz += d_loss_z

        self.dWnx += d_loss_n.reshape(self.h, 1) @ x_col.T
        self.dbin += d_loss_n
        self.dWnh += (d_loss_n * self.r).reshape(self.h, 1) @ h_col.T
        self.dbhn += d_loss_n * self.r

        dx = self.Wrx.T @ d_loss_r + self.Wzx.T @ d_loss_z + self.Wnx.T @ d_loss_n
        dh = (
            delta * self.z
            + self.Wrh.T @ d_loss_r
            + self.Wzh.T @ d_loss_z
            + self.Wnh.T @ (d_loss_n * self.r)
        )

        dx = dx.reshape(1, self.d)
        dh = dh.reshape(1, self.h)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        # return dx, dh
        return dx, dh
