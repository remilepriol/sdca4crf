import numpy as np


def entropy(logproba, axis=None, returnlog=False):
    themax = np.amax(logproba)
    try:
        ans = themax + np.log(- np.sum(np.exp(logproba - themax) * logproba, axis=axis))
        if returnlog:
            return ans
        else:
            return np.exp(ans)
    except FloatingPointError:
        print("Entropy problem:",
              themax, "\n",
              logproba)
        raise


def kullback_leibler(logp, logq, axis=None, returnlog=False):
    themax = np.amax(logp)
    ans = themax + np.log(np.sum(np.exp(logp - themax) * (logp - logq), axis=axis))
    if returnlog:
        return ans
    else:
        try:
            return np.exp(ans)
        except FloatingPointError:
            print("too big ", ans)
            raise


def logsumexp(v, axis=-1):
    vmax = np.amax(v, axis=axis, keepdims=True)
    return vmax.squeeze(axis=axis) + np.log(np.sum(np.exp(v - vmax), axis=axis))


def boolean_encoding(y, k):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1  #
    return ans