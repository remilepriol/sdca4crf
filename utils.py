import numpy as np


def entropy(proba):
    epsilon = 1e-50
    return -np.sum(proba * np.log(np.maximum(epsilon, proba)))


def kullback_leibler(p, q):
    eps = 1e-50
    return np.maximum(0, np.sum(p * np.log(np.maximum(p, eps) / np.maximum(q, eps)), axis=-1))


def logsumexp(v):
    vmax = np.amax(v, axis=-1, keepdims=True)
    return vmax.squeeze(axis=-1) + np.log(np.sum(np.exp(v - vmax), axis=-1))


def boolean_encoding(y, k):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1  #
    return ans
