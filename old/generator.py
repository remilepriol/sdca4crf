import numpy as np


def genrandomcov(d, square=False):
    """Generate a random positive semi definite matrix of dimension d."""
    # problem : the eigenvalues are squared from random numbers, hence they tend to yield larger condition numbers
    sig = np.random.randn(d, d)
    sig = np.dot(sig, sig.T)
    if not square:  # take the square root of the eigenvalues.
        eigval, eigvec = np.linalg.eigh(sig)
        eigval = np.sqrt(eigval)
        sig = np.dot(eigval * eigvec, eigvec.T)
    return sig


def gengaussians(n, d, theta=.5, scale=.1, shift=1, randomcov=False, gentest=False):
    """Generate n data points according to two random gaussians in dimension d.
    n = number of data points
    d = dimension of each data point
    theta = proportion of points in the first gaussian, between 0 and 1
    scale = scale of each gaussian
    shift = average horizontal shift of the first gaussian with respect to the second.

    :return
    x size n*d the data points
    y size n the index of the gaussian in {-1,1}
    mu 2*d the centers of the gaussians
    sig 2*d*d the covariances of the gaussians
    """
    # number of points drawn from each gaussian
    N1 = int(theta * n)
    N2 = n - N1
    # parameters of each gaussian
    mu = np.random.rand(2, d)
    # translate the centers to separate the gaussians
    mu[0] += shift
    if randomcov:
        sig1 = scale * genrandomcov(d)
        sig2 = scale * genrandomcov(d)
    else:
        sig1 = scale * np.eye(d)
        sig2 = sig1
    sig = [sig1, sig2]
    # draw the samples
    x = np.empty([n, d])
    x[:N1] = np.dot(np.random.randn(N1, d), sig[0]) + mu[0]
    x[N1:] = np.dot(np.random.randn(N2, d), sig[1]) + mu[1]
    # specify classes
    y = np.ones(n).astype(int)
    y[N1:] = -y[N1:]
    if gentest:
        xtest = np.empty([n, d])
        xtest[:N1] = np.dot(np.random.randn(N1, d), sig[0]) + mu[0]
        xtest[N1:] = np.dot(np.random.randn(N2, d), sig[1]) + mu[1]
        ytest = y.copy()
        return x, y, mu, sig, xtest, ytest
    else:
        return x, y, mu, sig


def random_parameters(d, k, scale, randomcov):
    """Return k random means and covariances in dimension d. The covariance are scaled by scale."""
    # parameters of each gaussian
    # means
    mu = np.random.rand(k, d)
    # covariances
    sig = np.zeros([k, d, d])
    if randomcov:
        for i in range(k):
            sig[i] = scale * genrandomcov(d)
    else:
        sig1 = scale * genrandomcov(d)
        for i in range(k):
            sig[i] = sig1

    return mu, sig


def draw_gmm_samples(mu, sig, cuts):
    n = cuts[-1]
    k, d = mu.shape
    x = np.empty((n, d))
    y = np.empty(n, dtype=int)
    lower = 0
    upper = cuts[0]
    for i in range(0, k):
        x[lower:upper] = np.random.multivariate_normal(mu[i], sig[i], size=upper - lower)
        y[lower:upper] = i
        if i < k - 1:
            lower = upper
            upper = cuts[i + 1]
    return x, y


def gengaussianmixture(n, d, k, uniform=True, scale=.1, randomcov=False, gentest=False):
    """Generate n data points drawn from a mixture of random gaussians in dimension d.
    n = number of data points
    d = dimension of each data point
    k = number of gaussians in the mixture
    uniform = if true, there will be as many points drawn from each gaussian.
    scale = average scale of the gaussians
    randomcov =  if true, the covariances of the gaussians are generated randomly

    :return
    x size n*d the data points
    y size n the index of the gaussian in [0,k-1]
    mu 2*d the centers of the gaussians
    sig 2*d*d the covariances of the gaussians
    """
    # number of points drawn from each gaussian
    if uniform:
        cuts = (np.arange(1, k + 1) * n / k).astype(int)
    else:
        cuts = np.random.randint(0, n, size=k)
        cuts[-1] = n
        cuts.sort()

    # parameters of each gaussian
    mu, sig = random_parameters(d, k, scale, randomcov)

    # draw the samples
    x, y = draw_gmm_samples(mu, sig, cuts)
    if gentest:
        xtest, ytest = draw_gmm_samples(mu, sig, cuts)
        return x, y, mu, sig, xtest, ytest
    else:
        return x, y, mu, sig


def standardize(x):
    """Return data points centered and reduced.

    :param x: size n*d data points
    :return: x'
    """
    ans = x - np.mean(x, axis=0, keepdims=True)
    ans /= np.std(ans, axis=0, keepdims=True)
    return ans
