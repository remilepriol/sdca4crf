import numpy as np


def logsumexp(v):
    vmax = np.amax(v, axis=-1)
    return vmax + np.log(np.sum(np.exp(v - vmax[:, np.newaxis]), axis=-1))


def negloglikelihood(w, x, y, i=False):
    # regularization scheme : soustraire le maximum
    # dans ce cas simple, soustraire le score si il est positif.
    n, d = x.shape
    score = np.zeros([n, 2])
    score[:, 0] = -y * np.dot(x, w)
    return np.mean(logsumexp(score))


def sigmoid(z):
    ans = np.ones_like(z)
    ans[z > 0] = 1 / (1 + np.exp(-z[z > 0]))
    ans[z < 0] = np.exp(z[z < 0]) / (1 + np.exp(z[z < 0]))
    return ans


def stochastic_gradient(w, x, y, i, scalar=False):
    """Return the gradient of the i-th negative-log-likelihood with respect to w.
    If scalar is True, return the value that is to be multiplied by x[i] to get the gradient.
    """
    z = -y[i] * x[i]
    if scalar:
        return sigmoid((np.dot(w, z))) * (-y[i])
    else:
        return sigmoid((np.dot(w, z))) * z


def entropy_bernoulli(alpha):
    return - alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha)


def dual_to_primal(alpha, x, y, reg):
    n = x.shape[0]
    a = 1 / reg / n * y[:, np.newaxis] * x
    return np.dot(a.T, alpha)

# def primalobjective(w, x, y, reg, bias=False):
#     """Return the score of the classifier for a given parameter w
#     w = parameter vector of dimension d.
#     x = n*d design matrix : each line is a data point.
#     y = class vector with values in {-1,1} of dimension n
#     reg = l2 regularization parameter
#     bias = if true, then the last coordinate of w is not regularized.
#     """
#     return regularization(w, reg, bias) + negloglikelihood(w, x, y)


# def dualobjective(w, alpha, reg):
#     """Return the score of the dual of the logistic regression.
#     w = w(alpha) should be the dual vector of alpha. Passed in argument to reduce complexity.
#     alpha = dual parameter.
#     reg = l2 regularization parameter
#     """
#     entropy = np.mean(entropy_bernoulli(alpha))
#     return regularization(w, reg) + entropy
