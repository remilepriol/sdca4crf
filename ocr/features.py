import matplotlib.pyplot as plt
import numpy as np

import oracles
from ocr.parse import ALPHABET, ALPHALEN, NB_PIXELS, letters2wordimage
from sequence import Sequence


# RADIUS OF THE CORRECTED FEATURES
def radius(word, label):
    """Return \max_y \|F(x_i,y_i) - F(x_i,y) \|^2

    :param word: sequence of letter images x
    :param label: sequence of letter values y
    """
    feat = Features()

    # ground truth feature
    feat.add_datapoint(word, label)
    feat.multiply_scalar(-1, inplace=True)

    # the optimal y puts all the weight on one character
    # that is not included in the true label
    char = np.setdiff1d(np.arange(ALPHALEN), label)[0]
    label2 = char * np.ones_like(label)
    feat.add_datapoint(word, label2)

    return np.sqrt(feat.squared_norm())


def radii(words, labels):
    rs = np.empty_like(words)
    for i, (word, label) in enumerate(zip(words, labels)):
        rs[i] = radius(word, label)
    return rs


class Features:
    """Features associated to a certain word. also used to store the weights of the primal
    model.
    """

    def __init__(self, emission=None, bias=None, transition=None, random=False):
        if random:
            self.emission = np.random.randn(ALPHALEN, NB_PIXELS)
            self.bias = np.random.randn(ALPHALEN, 3)
            self.transition = np.random.randn(ALPHALEN, ALPHALEN)
            return

        if emission is None:
            self.emission = np.zeros([ALPHALEN, NB_PIXELS])
        else:
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
    def _add_unary(self, images, label, position):
        self.emission[label] += images[position]
        self.bias[label] += [1, position == 0, position == images.shape[0] - 1]

    def _add_binary(self, label, next_label):
        self.transition[label, next_label] += 1

    def add_datapoint(self, images, labels):
        for t in range(images.shape[0]):
            self._add_unary(images, labels[t], t)
        for t in range(images.shape[0] - 1):
            self._add_binary(labels[t], labels[t + 1])

    def _add_unary_centroid(self, images, unary_marginals=None):
        if unary_marginals is None:  # assume uniform marginal
            self.emission += np.sum(images, axis=0) / ALPHALEN
            self.bias[:, 0] += images.shape[0] / ALPHALEN
            self.bias[:, 1:] += 1 / ALPHALEN
        else:
            self.emission += np.dot(unary_marginals.T, images)
            self.bias[:, 0] += np.sum(unary_marginals, axis=0)
            self.bias[:, 1] += unary_marginals[0]
            self.bias[:, 2] += unary_marginals[-1]

    def _add_binary_centroid(self, images, binary_marginals=None):
        if binary_marginals is None:  # assume uniform marginal
            self.transition += (images.shape[0] - 1) / ALPHALEN ** 2
        else:
            self.transition += np.sum(binary_marginals, axis=0)

    def add_centroid(self, images, marginals=None):
        if marginals is None:  # assume uniform marginal
            self._add_unary_centroid(images, None)
            self._add_binary_centroid(images, None)
        else:
            if marginals.islog:
                marginals = marginals.exp()
            self._add_unary_centroid(images, marginals.unary)
            self._add_binary_centroid(images, marginals.binary)

    #########################################
    # From weights to probabilities
    #########################################
    def unary_scores(self, images):
        """Return the unary scores of word when self encode the weights of the model.

        :param images: T*d, each line is a letter image.
        :return: unary scores T*K, each case is a score for one image and one label.
        """
        uscores = np.dot(images, self.emission.T)
        uscores += self.bias[:, 0]  # model bias
        uscores[0] += self.bias[:, 1]  # beginning of word bias
        uscores[-1] += self.bias[:, 2]  # end of word bias
        return uscores

    def binary_scores(self, images):
        """Return the binary scores of a word when self encode the weights of the model.

        :param images: images T*d, each line is a letter image.
        :return: binary scores (T-1)*K*K, each case is the transition score between two labels
        for a given position.
        """
        return (images.shape[0] - 1) * [self.transition]

    def infer_probabilities(self, images):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        umargs, bmargs, log_partition = oracles.sequence_sum_product(uscores, bscores)
        umargs = np.minimum(umargs, 0)
        bmargs = np.minimum(bmargs, 0)
        ans = Sequence(umargs, bmargs, log=True)

        nans = ans.exp()
        assert nans.is_consistent()
        assert nans.is_density(1)

        return ans, log_partition

    def word_score(self, images, labels):
        """Return the score <self,F(images,labels)>."""
        ans = np.sum(images * self.emission[labels])
        ans += np.sum(self.bias[labels, 0])
        ans += self.bias[labels[0], 1]
        ans += self.bias[labels[-1], 2]
        ans += np.sum(self.transition[labels[:-1], labels[1:]])
        return ans

    def predict(self, images):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        return oracles.sequence_viterbi(uscores, bscores)[0]

    #########################################
    # Arithmetic operations
    #########################################
    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.emission *= scalar
            self.bias *= scalar
            self.transition *= scalar
        else:
            return Features(self.emission * scalar, self.bias * scalar, self.transition * scalar)

    def combine(self, other, ufunc):
        emission = ufunc(self.emission, other.emission)
        bias = ufunc(self.bias, other.bias)
        transition = ufunc(self.transition, other.transition)
        return Features(emission, bias, transition)

    def add(self, other):
        return self.combine(other, np.add)

    def subtract(self, other):
        return self.combine(other, np.subtract)

    def squared_norm(self):
        return np.sum(self.emission ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    def inner_product(self, other):
        return np.sum(self.emission * other.emission) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)

    def display(self):
        cmap = "Greys"
        emissions = letters2wordimage(self.emission)
        plt.matshow(emissions, cmap=cmap)
        ticks_positions = np.linspace(0, emissions.shape[1], ALPHALEN + 2).astype(int)[1:-1]
        plt.xticks(ticks_positions, list(ALPHABET))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.matshow(self.transition, cmap=cmap)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.yticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features")
        rescale_bias = np.array([1 / 7.5, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features")