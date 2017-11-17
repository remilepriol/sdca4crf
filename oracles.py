import numpy as np

import utils


def chain_sum_product(uscores, bscores):
    """Apply the sum-product algorithm on a chain

    :param uscores: array T*K, (unary) scores on individual nodes
    :param bscores: array (T-1)*K*K, (binary) scores on the edges
    :return: log-marginals on nodes, log-marginals on edges, log-partition
    """

    # I keep track of the log messages instead of the messages
    # This is more stable numerically

    chain_length, nb_class = uscores.shape

    # backward pass
    backward_messages = np.zeros([chain_length - 1, nb_class])
    backward_messages[-1] = utils.logsumexp(bscores[-1] + uscores[-1])
    for t in range(chain_length - 3, -1, -1):
        backward_messages[t] = utils.logsumexp(bscores[t] + uscores[t + 1] + backward_messages[t + 1])

    # we compute the log-partition and include it in the forward messages
    log_partition = utils.logsumexp(backward_messages[0] + uscores[0])

    # forward pass
    forward_messages = np.zeros([chain_length - 1, nb_class])
    forward_messages[0] = utils.logsumexp(bscores[0].T + uscores[0] - log_partition)
    for t in range(1, chain_length - 1):
        forward_messages[t] = utils.logsumexp(bscores[t].T + uscores[t] + forward_messages[t - 1])

    unary_marginals = np.empty([chain_length, nb_class])
    unary_marginals[0] = uscores[0] + backward_messages[0] - log_partition
    unary_marginals[-1] = forward_messages[-1] + uscores[-1]
    for t in range(1, chain_length - 1):
        unary_marginals[t] = forward_messages[t - 1] + uscores[t] + backward_messages[t]

    binary_marginals = np.empty([chain_length - 1, nb_class, nb_class])
    binary_marginals[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] + backward_messages[1] - log_partition
    binary_marginals[-1] = forward_messages[-2, :, np.newaxis] + uscores[-2, :, np.newaxis] + bscores[-1] + uscores[-1]
    for t in range(1, chain_length - 2):
        binary_marginals[t] = forward_messages[t - 1, :, np.newaxis] + uscores[t, :, np.newaxis] + bscores[t] + uscores[
            t + 1] + backward_messages[t + 1]

    return unary_marginals, binary_marginals, log_partition


def chain_viterbi(uscores, bscores):
    # I keep track of the score instead of the potentials
    # because summation is more stable than multiplication
    chain_length, nb_class = uscores.shape

    # backward pass
    argmax_messages = np.empty([chain_length - 1, nb_class], dtype=int)
    max_messages = np.empty([chain_length - 1, nb_class], dtype=float)
    tmp = bscores[-1] + uscores[-1]
    # Find the arg max
    argmax_messages[-1] = np.argmax(tmp, axis=-1)
    # Store the max
    max_messages[-1] = tmp[np.arange(nb_class), argmax_messages[-1]]
    for t in range(chain_length - 3, -1, -1):
        tmp = bscores[t] + uscores[t + 1] + max_messages[t + 1]
        argmax_messages[t] = np.argmax(tmp, axis=-1)
        max_messages[t] = tmp[np.arange(nb_class), argmax_messages[t]]

    # Label to be returned
    global_argmax = np.empty(chain_length, dtype=int)

    # forward pass
    tmp = max_messages[0] + uscores[0]
    global_argmax[0] = np.argmax(tmp)
    global_max = tmp[global_argmax[0]]
    for t in range(1, chain_length):
        global_argmax[t] = argmax_messages[t - 1, global_argmax[t - 1]]

    return global_argmax, global_max
