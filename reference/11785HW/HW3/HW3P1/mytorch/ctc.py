import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = 1)
                target output

        Return
        ------
        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks
        skipConnect: (np.array, dim = 1)
                    skip connections

        """
        extSymbols = None
        skipConnect = None

        # -------------------------------------------->

        # Your Code goes here
        target = np.asarray(target).reshape(-1)
        L = len(target)
        S = 2 * L + 1
        extSymbols = np.full(S, self.BLANK, dtype=target.dtype)
        extSymbols[1::2] = target
        skipConnect = np.zeros(S, dtype=int)
        for s in range(2, S):
            if extSymbols[s] != self.BLANK and extSymbols[s] != extSymbols[s - 2]:
                skipConnect[s] = 1
        # <---------------------------------------------

        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (output len, out channel))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        if T == 0 or S == 0:
            return alpha

        alpha[0, 0] = logits[0, extSymbols[0]]
        if S > 1:
            alpha[0, 1] = logits[0, extSymbols[1]]

        for t in range(1, T):
            for s in range(S):
                prev = alpha[t - 1, s]
                if s > 0:
                    prev += alpha[t - 1, s - 1]
                if s > 1 and skipConnect[s]:
                    prev += alpha[t - 1, s - 2]
                alpha[t, s] = prev * logits[t, extSymbols[s]]
        # <---------------------------------------------

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        beta: (np.array, dim = (output len, out channel))
                backward probabilities

        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        beta[T - 1, S - 1] = 1
        if S > 1:
            beta[T - 1, S - 2] = 1
        for t in reversed(range(T - 1)):
            for s in reversed(range(S)):
                after = beta[t + 1, s] * logits[t + 1, extSymbols[s]]
                if s < S - 1:
                    after += beta[t + 1, s + 1] * logits[t + 1, extSymbols[s + 1]]
                if s < S - 2 and skipConnect[s + 2]:
                    after += beta[t + 1, s + 2] * logits[t + 1, extSymbols[s + 2]]

                beta[t, s] = after
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array)
                forward probability

        beta: (np.array)
                backward probability

        Return
        ------
        gamma: (np.array)
                posterior probability

        """
        gamma = None

        # -------------------------------------------->
        T, S = alpha.shape
        gamma = alpha * beta
        column_sums = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(
            gamma, column_sums, out=np.zeros_like(gamma), where=column_sums != 0
        )
        # <---------------------------------------------

        return gamma
