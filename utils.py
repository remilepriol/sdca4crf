import numpy as np


def entropy(proba, axis=None):
    epsilon = 1e-50
    return -np.sum(proba * np.log(np.maximum(epsilon, proba)), axis=axis)


def kullback_leibler(p, q, axis=None):
    epsilon = 1e-50
    return np.maximum(0, np.sum(p * np.log(np.maximum(p, epsilon) / np.maximum(q, epsilon)), axis=axis))


def logsumexp(v):
    vmax = np.amax(v, axis=-1, keepdims=True)
    return vmax.squeeze(axis=-1) + np.log(np.sum(np.exp(v - vmax), axis=-1))


def boolean_encoding(y, k):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1  #
    return ans


def bounded_newton(func, grad, init, lowerbound, upperbound, precision=1e-12, max_iter=50):
    x = init
    fx = func(x)
    obj = [fx]
    count = 0
    while np.absolute(fx) > precision and count < max_iter:  # stop condition to avoid cycling over an extremity of 0,1?
        count += 1
        gx = grad(x)
        x -= fx / gx
        # Make sure x is in (lower bound, upper bound)
        x = max(lowerbound + precision, x)
        x = min(upperbound - precision, x)
        fx = func(x)
        obj.append(fx)
    return x, obj


def find_root_decreasing(u, gu, precision):
    """Return the root of a decreasing function u defined on [0,1] with given precision."""

    u0 = u(0)
    if u0 <= precision:  # 0 is optimal
        return 0, [u0]

    u1 = u(1)
    if u1 >= -precision:  # 1 is optimal
        return 1, [u1]

    return bounded_newton(u, gu, .5, 0, 1, precision=precision)
