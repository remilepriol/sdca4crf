import matplotlib.pyplot as plt
import numpy as np

import oracles
from chunk.parse import ALPHABET, ALPHALEN
from sequence import Sequence


def radius(xi, yi):
    """Return the maximum radius (not squared) of the corrected feature for tuple (xi, yi)"""
    feat = Features()

    # ground truth feature
    feat.add_datapoint(xi, yi)
    feat.multiply_scalar(-1, inplace=True)

    # the near-optimal y puts all the weight on one character
    # that is the least present in the true label

    ychars, ycounts = np.unique(yi, return_counts=True)
    diffchars = np.setdiff1d(np.arange(ALPHALEN), yi)
    if len(diffchars) > 0:
        char = diffchars[0]
    else:
        char = ychars[np.argmin(ycounts)]

    label2 = char * np.ones_like(yi)
    feat.add_datapoint(xi, label2)

    return np.sqrt(feat.squared_norm())


def radii(words, labels):
    rs = np.empty_like(words)
    for i, (word, label) in enumerate(zip(words, labels)):
        rs[i] = radius(word, label)
    return rs


class Features:
    """Features associated to a sample and a label.
    Taking the centroid of such features gives the weights of the primal model.
    
    Features are composed of:
    - sparse emission features (unary), which counts the number of apparitions of a each
    attribute for each tag. Because we use Features to represent the weights vector, that may
    not be sparse, at least not exactly, we represent emission as a dense matrix.
    - dense bias features (unary), which counts the number of apparition of each tag, 
    (1) in general, (2) at the beginning, (3) at the end.
    - dense transition features (binary), which counts the number of transition between every tags.
    
    Attributes are defined prior to parsing. The parser reads a file made of 'label\t attributes'.
    """

    def __init__(self, emission=None, bias=None, transition=None):

        self.emission = emission

        if bias is None:
            self.bias = np.zeros([ALPHALEN, 3])
        else:
            self.bias = bias

        if transition is None:
            self.transition = np.zeros([ALPHALEN, ALPHALEN])
        else:
            self.transition = transition

    #########################################
    # Construction operations
    #########################################
    def _init_emission(self, nb_features):
        if self.emission is None:
            self.emission = np.zeros((ALPHALEN, nb_features))

    def _add_unary(self, xi, yit, t, emission_counts):
        if emission_counts is None:
            self.emission[yit, xi[t].indices] += 1
        else:
            emission_counts[yit].extend(xi[t].indices)
        self.bias[yit] += [1, t == 0, t == xi.shape[0] - 1]

    def _add_binary(self, yit, yitp):
        self.transition[yit, yitp] += 1

    def add_datapoint(self, xi, yi, emission_counts=None):
        if xi.shape[0] != len(yi):
            raise ValueError(
                "Not the same number of tags (%i) and words (%i) in sentence."
                % (xi.shape[0], len(yi)))
        self._init_emission(xi[0].dimension)
        for t, label in enumerate(yi):
            self._add_unary(xi, label, t, emission_counts)
        for t in range(xi.shape[0] - 1):
            self._add_binary(yi[t], yi[t + 1])

    def add_dataset(self, x, y):
        if len(x) != len(y):
            raise ValueError(
                "Not the same number of labels (%i) and data points (%i)."
                % (len(x), len(y)))

        # to be fast, I have to avoid the addition of vectors of size nb_features = 75k at each
        # iteration. I use the same method as for the dictionary initialization with the numpy
        # unique method.
        emission_counts = np.empty(ALPHALEN, dtype=list)
        for i in range(ALPHALEN):
            emission_counts[i] = []

        for xi, yi in zip(x, y):
            self.add_datapoint(xi, yi, emission_counts)

        for i, em in enumerate(emission_counts):
            if len(em) > 0:
                indices, values = np.unique(em, return_counts=True)
                self.emission[i, indices] = values

    def _add_unary_centroid(self, xi, unary_marginals):
        self._init_emission(xi[0].dimension)
        # Important part. I hope it works
        for xit, mt in zip(xi, unary_marginals):
            self.emission[:, xit.indices] += mt[:, np.newaxis]

        self.bias[:, 0] += np.sum(unary_marginals, axis=0)
        self.bias[:, 1] += unary_marginals[0]
        self.bias[:, 2] += unary_marginals[-1]

    def _add_binary_centroid(self, binary_marginals):
        self.transition += np.sum(binary_marginals, axis=0)

    def add_centroid(self, xi, marginals):
        if marginals.islog:
            marginals = marginals.exp()
        self._add_unary_centroid(xi, marginals.unary)
        self._add_binary_centroid(marginals.binary)

    #########################################
    # From weights to probabilities
    #########################################
    def _unary_scores(self, xi):
        """Return the unary scores of word when self encode the weights of the model.

        :param xi: T*d, each column is a word embedding (csr matrix).
        :return: u unary scores T*K, u(t,k) is the score for word t and label k.
        """
        uscores = np.zeros([xi.shape[0], ALPHALEN])
        # Important part. I hope it works.
        # uscores = xi.dot(self.emission.T)
        for t, xit in enumerate(xi):
            uscores[t] += self.emission[:, xit.indices].sum(axis=1)

        uscores += self.bias[:, 0]  # model bias
        uscores[0] += self.bias[:, 1]  # beginning of word bias
        uscores[-1] += self.bias[:, 2]  # end of word bias
        return uscores

    def _binary_scores(self, xi):
        """Return the binary scores of a word when self encode the weights of the model.

        :param xi:
        :return: binary scores (T-1)*K*K, each case is the transition score between two labels
        for a given position.
        """
        return (xi.shape[0] - 1) * [self.transition]

    def infer_probabilities(self, xi):
        uscores = self._unary_scores(xi)
        bscores = self._binary_scores(xi)
        umargs, bmargs, log_partition = oracles.sequence_sum_product(uscores, bscores)
        umargs = np.minimum(umargs, 0)
        bmargs = np.minimum(bmargs, 0)
        ans = Sequence(umargs, bmargs, log=True)

        nans = ans.exp()
        assert nans.is_consistent()
        assert nans.is_density(1)

        return ans, log_partition

    def label_score(self, xi, yi):
        """Return the score <self,F(xi, yi)>."""
        label_feat = Features()
        label_feat.add_datapoint(xi, yi)
        return label_feat.inner_product(self)

    def predict(self, xi):
        uscores = self._unary_scores(xi)
        bscores = self._binary_scores(xi)
        return oracles.sequence_viterbi(uscores, bscores)

    def prediction_loss(self, x, y):
        loss01 = 0
        loss_hamming = 0
        nb_letters = 0

        for images, truth in zip(x, y):
            prediction = self.predict(images)[0]
            tmp = np.sum(truth != prediction)
            loss_hamming += tmp
            loss01 += (tmp > 0)
            nb_letters += len(truth)

        return loss01 / len(y), loss_hamming / nb_letters

    #########################################
    # Arithmetic operations
    #########################################
    def map(self, ufunc, inplace=False):
        """Map ufunc on each of the arrays inside self.

        We need ufunc(0) = 0 , otherwise emission won't be sparse anymore.
        """

        if not np.isclose(ufunc(0), 0):
            raise ValueError("ufunc %s should return 0 in 0." % (ufunc))

        if inplace:
            self.emission = ufunc(self.emission)
            self.bias = ufunc(self.bias)
            self.transition = ufunc(self.transition)
        else:
            return Features(ufunc(self.emission), ufunc(self.bias), ufunc(self.transition))

    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.emission = scalar * self.emission
            self.bias = scalar * self.bias
            self.transition = scalar * self.transition
        else:
            return Features(scalar * self.emission, scalar * self.bias, scalar * self.transition)

    def squared_norm(self):
        return np.sum(self.emission ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    def add(self, other):
        emission = self.emission + other.emission
        bias = self.bias + other.bias
        transition = self.transition + other.transition
        return Features(emission, bias, transition)

    def subtract(self, other):
        emission = self.emission - other.emission
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        return Features(emission, bias, transition)

    def multiply(self, other):
        emission = self.emission * other.emission
        bias = self.bias * other.bias
        transition = self.transition * other.transition
        return Features(emission, bias, transition)

    def reduce(self):
        """Return the addition of the features."""
        ans = self.emission.sum()
        ans += self.bias.sum()
        ans += self.transition.sum()
        return ans

    def inner_product(self, other):
        return self.multiply(other).reduce()

    def display(self):
        """Display bias and transition features as heatmaps."""
        cmap = "Greys"
        plt.matshow(self.transition, cmap=cmap)
        plt.grid()
        tags_range = range(ALPHALEN)
        plt.xticks(tags_range, ALPHABET, rotation='vertical')
        plt.yticks(tags_range, ALPHABET)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features", y=1.3)

        rescale_bias = np.array([1 / 23, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.xticks(tags_range, ALPHABET)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features", y=1.15)

        plt.show()
