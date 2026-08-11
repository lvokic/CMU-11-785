import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC(BLANK=self.BLANK)
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        self.gammas = []
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Extend Sequence with blank ->
            #     Compute forward probabilities ->
            #     Compute backward probabilities ->
            #     Compute posteriors using total probability function
            #     Compute Expected Divergence and take average on batches
            # <---------------------------------------------

            # -------------------------------------------->
            T_b = int(input_lengths[b])
            L_b = int(target_lengths[b])

            sample_logits = logits[:T_b, b, :]
            sample_target = target[b, :L_b]

            extSymbols, skipConnect = self.ctc.targetWithBlank(target=sample_target)
            alpha = self.ctc.forwardProb(
                logits=sample_logits, extSymbols=extSymbols, skipConnect=skipConnect
            )
            beta = self.ctc.backwardProb(
                logits=sample_logits, extSymbols=extSymbols, skipConnect=skipConnect
            )
            gamma = self.ctc.postProb(alpha=alpha, beta=beta)

            selected_probs = sample_logits[:, extSymbols]
            selected_probs = np.clip(selected_probs, 1e-12, None)

            totalLoss[b] = -np.sum(gamma * np.log(selected_probs))
            self.gammas.append(gamma)
            # <---------------------------------------------

        return np.mean(totalLoss)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: np.ndarray, shape = (seqlength, batch_size, len(Symbols))
            Derivative of the divergence with respect to the input symbols
            at each time step, sample, and class.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # <---------------------------------------------

            # -------------------------------------------->
            gamma = self.gammas[b]
            input_length = int(self.input_lengths[b])
            target_length = int(self.target_lengths[b])

            target_b = self.target[b, :target_length]
            extSymbols, _ = self.ctc.targetWithBlank(target_b)

            probs = np.clip(self.logits[:input_length, b, :], 1e-12, None)
            grad = np.zeros_like(probs)

            for t in range(input_length):
                for s, symbol in enumerate(extSymbols):
                    grad[t, symbol] -= gamma[t, s] / probs[t, symbol]

            dY[:input_length, b, :] = grad
            # <---------------------------------------------

        return dY
