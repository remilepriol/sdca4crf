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

    return safe_newton(evaluator, 0, 1, precision=precision)


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


# define MAXIT 100 Maximum allowed number of iterations.
def safe_newton(evaluator, lowerbound, upperbound, precision, max_iter=200):
    """Using a combination of Newton-Raphson and bisection, find the root of a function bracketed between lowerbound
    and upperbound.

    :param evaluator: user-supplied routine that returns both the function value u(x) and u(x)/u'(x).
    :param lowerbound: point smaller than the root
    :param upperbound: point larger than the root
    :param precision: accuracy on the root value rts
    :param max_iter: maximum number of iteration
    :return: The root, returned as the value rts
    """

    fl, _ = evaluator(lowerbound)
    fh, _ = evaluator(upperbound)
    if (fl > 0 and fh > 0) or (fl < 0 and fh < 0):
        raise ValueError("Root must be bracketed in [lower bound, upper bound]")
    if fl == 0:
        return lowerbound
    if fh == 0:
        return upperbound

    if fl < 0:  # Orient the search so that f(xl) < 0.
        xl, xh = lowerbound, upperbound
    else:
        xh, xl = lowerbound, upperbound

    rts = (xl + xh) / 2  # Initialize the guess for root
    dxold = abs(upperbound - lowerbound)  # the “stepsize before last"
    dx = dxold  # and the last step

    f, fdf = evaluator(rts)
    obj = [[f]]

    for _ in np.arange(max_iter):  # Loop over allowed iterations.
        rtsold = rts
        rts -= fdf
        if not ((rts - xh) * (rts - xl) < 0 and abs(fdf) > abs(dxold) / 2):
            # Bisect if Newton out of range, or not converging fast enough
            # we check with a negation to dissect in case fdf is NaN
            dxold = dx
            dx = (xh - xl) / 2
            rts = xl + dx
            if xl == rts:  # change in root is negligible
                return rts, np.array(obj)
        else:  # Newton step (already applied pn rts)
            dxold = dx
            dx = fdf
            if rtsold == rts:  # change in root is negligible
                return rts, np.array(obj)
        if abs(dx) < precision:  # Convergence criterion.
            return rts, np.array(obj)
        f, fdf = evaluator(rts)  # the one new function evaluation per iteration
        obj.append([f])
        if f < 0:  # maintain the bracket on the root
            xl = rts
        else:
            xh = rts

    raise RuntimeError("Maximum number of iterations exceeded in safe_newton")

