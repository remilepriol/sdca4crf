import numpy as np


def entropy(proba, axis=None):
    epsilon = 1e-50
    return -np.sum(proba * np.log(np.maximum(epsilon, proba)), axis=axis)


def kullback_leibler(p, q, axis=None):
    epsilon = 1e-50
    return np.maximum(0, np.sum(p * np.log(np.maximum(p, epsilon) / np.maximum(q, epsilon)), axis=axis))


def log_entropy(logproba, axis=None):
    return -np.sum(np.exp(logproba) * logproba, axis=axis)


def log_kullback_leibler(logp, logq, axis=None):
    return np.sum(np.exp(logp) * (logp - logq), axis=axis)


def logsumexp(v, axis=-1):
    vmax = np.amax(v, axis=axis, keepdims=True)
    return vmax.squeeze(axis=axis) + np.log(np.sum(np.exp(v - vmax), axis=axis))


def boolean_encoding(y, k):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1  #
    return ans


def bounded_newton(evaluator, init, lowerbound, upperbound, precision=1e-12, max_iter=20):
    """Return the root x0 of a function u defined on [lowerbound, upperbound] with given precision,
    using Newton-Raphson method

    :param evaluator: function that return the values u(x) and u(x)/u'(x)
    :param init: initial point x
    :param lowerbound:
    :param upperbound:
    :param precision: on the value of |u(x)|
    :param max_iter: maximum number of iterations
    :return: x an approximate root of u
    """
    x = init
    fx, gx = evaluator(x)
    obj = [fx]
    count = 0
    while np.absolute(fx) > precision and count < max_iter:
        # stop condition to avoid cycling over an extremity of the segment
        count += 1
        x -= gx
        # Make sure x is in (lower bound, upper bound)
        x = max(lowerbound, x)
        x = min(upperbound, x)
        fx, gx = evaluator(x)
        obj.append(fx)
    return x, obj


def find_root_decreasing(evaluator, precision):
    """Return the root x0 of a decreasing function u defined on [0,1] with given precision.
    The root can be smaller than 0, in which case, return 0.
    The root can be larger than 1, in which case, return 1.

    :param evaluator: function that return the values u(x) and u(x)/u'(x)
    :param precision: maximum value of |u(x)| so that x is returned
    :return: x an approximate root of u
    """

    u0, _ = evaluator(0)
    if u0 <= precision:  # 0 is optimal
        return 0, [u0]

    u1, _ = evaluator(1)
    if u1 >= -precision:  # 1 is optimal
        return 1, [u1]

    return bounded_newton(evaluator, .5, 0, 1, precision=precision)
