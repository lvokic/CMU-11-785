import numpy as np

def collapse(path):
    result = []
    previous = None
    for symbol in path:
        if symbol != "-" and symbol != previous:
            result.append(symbol)
        previous = symbol
    return "".join(result)

def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    blank = 0
    S, T, B = y_probs.shape
    forward_path = []
    forward_prob = []

    for b in range(B):
        y_prob = y_probs[:, :, b]
        # y_prob has shape (symbols, time); choose one symbol per time step.
        best_symbols = np.argmax(y_prob, axis=0)
        best_prob = np.max(y_prob, axis=0)
        forward_prob.append(np.prod(best_prob))
        decoded = []
        prev = None
        for symbol in best_symbols:
            if symbol == prev:
                continue
            if symbol != blank:
                decoded.append(SymbolSets[symbol - 1])
            prev = symbol
        forward_path.append("".join(decoded))

    if B == 1:
        return (forward_path[0], forward_prob[0])
    return (forward_path, forward_prob)


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    blank = 0
    S, T, B = y_probs.shape

    best_paths = []
    merged_scores = []

    for b in range(B):
        # prefix -> (probability ending in blank, probability ending in symbol)
        bestPathsWithScores = {"": (1.0, 0.0)}
        y_prob = y_probs[:, :, b]

        for t in range(T):
            candidates = {}
            for prefix, (p_blank, p_nonblank) in bestPathsWithScores.items():
                total = p_blank + p_nonblank
                blank_prob = y_prob[blank, t]
                old_blank, old_nonblank = candidates.get(prefix, (0.0, 0.0))
                candidates[prefix] = (old_blank + total * blank_prob, old_nonblank)

                for idx, symbol in enumerate(SymbolSets, start=1):
                    symbol_prob = y_prob[idx, t]
                    if prefix and prefix[-1] == symbol:
                        # Repeating a symbol without a blank keeps the same prefix.
                        old_blank, old_nonblank = candidates.get(prefix, (0.0, 0.0))
                        candidates[prefix] = (
                            old_blank,
                            old_nonblank + p_nonblank * symbol_prob,
                        )
                        extended = prefix + symbol
                        old_blank, old_nonblank = candidates.get(extended, (0.0, 0.0))
                        candidates[extended] = (
                            old_blank,
                            old_nonblank + p_blank * symbol_prob,
                        )
                    else:
                        extended = prefix + symbol
                        old_blank, old_nonblank = candidates.get(extended, (0.0, 0.0))
                        candidates[extended] = (
                            old_blank,
                            old_nonblank + total * symbol_prob,
                        )

            if t < T - 1:
                bestPathsWithScores = dict(
                    sorted(
                        candidates.items(),
                        key=lambda item: item[1][0] + item[1][1],
                        reverse=True,
                    )[:BeamWidth]
                )
            else:
                finalCandidates = candidates

        scores = {}
        for prefix, (p_blank, p_nonblank) in finalCandidates.items():
            scores[prefix] = p_blank + p_nonblank
        merged_scores.append(scores)
        best_paths.append(max(scores, key=scores.get))

    if B == 1:
        return best_paths[0], merged_scores[0]
    return best_paths, merged_scores
