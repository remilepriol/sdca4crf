import matplotlib.pyplot as plt
import numpy as np

import oracles
import parse
from chains import LogProbability, Probability
from constant import ALPHABET, ALPHABET_SIZE, NB_PIXELS


# RADIUS OF THE CORRECTED FEATURES

def radius(word, label):
    """Return \max_y \|F(x_i,y_i) - F(x_i,y) \|^2

    :param word: sequence of letter images x
    :param label: sequence of letter values y
    """
    feat = Features()

    # ground truth feature
    feat.add_word(word, label)
    feat.multiply_scalar(-1, inplace=True)

    # the optimal y puts all the weight on one character
    # that is not included in the true label
    char = np.setdiff1d(np.arange(ALPHABET_SIZE), label)[0]
    label2 = char * np.ones_like(label)
    feat.add_word(word, label2)

    return np.sqrt(feat.squared_norm())


def radii(words, labels):
    rs = np.empty_like(words)
    for i, word, label in enumerate(zip(words, labels)):
        rs[i] = radius(word, label)
    return rs


class Features:
    """Features associated to a certain word. also used to store the weights of the primal
    model.
    """

    def __init__(self, emission=None, bias=None, transition=None, random=False):
        if random:
            self.emission = np.random.randn(ALPHABET_SIZE, NB_PIXELS)
            self.bias = np.random.randn(ALPHABET_SIZE, 3)
            self.transition = np.random.randn(ALPHABET_SIZE, ALPHABET_SIZE)
            return

        if emission is None:
            self.emission = np.zeros([ALPHABET_SIZE, NB_PIXELS])
        else:
            self.emission = emission

        if bias is None:
            self.bias = np.zeros([ALPHABET_SIZE, 3])
        else:
            self.bias = bias

        if transition is None:
            self.transition = np.zeros([ALPHABET_SIZE, ALPHABET_SIZE])
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

    def add_word(self, images, labels):
        for t in range(images.shape[0]):
            self._add_unary(images, labels[t], t)
        for t in range(images.shape[0] - 1):
            self._add_binary(labels[t], labels[t + 1])

    def add_dictionary(self, images_set, labels_set):
        for images, labels in zip(images_set, labels_set):
            word_size = labels.shape[0]
            if word_size != images.shape[0]:
                raise ValueError("Not the same number of labels (%i) and images (%i) inside word."
                                 % (word_size, images.shape[0]))
            self.add_word(images, labels)

    def _add_unary_centroid(self, images, unary_marginals=None):
        if unary_marginals is None:  # assume uniform marginal
            self.emission += np.sum(images, axis=0) / ALPHABET_SIZE
            self.bias[:, 0] += images.shape[0] / ALPHABET_SIZE
            self.bias[:, 1:] += 1 / ALPHABET_SIZE
        else:
            self.emission += np.dot(unary_marginals.T, images)
            self.bias[:, 0] += np.sum(unary_marginals, axis=0)
            self.bias[:, 1] += unary_marginals[0]
            self.bias[:, 2] += unary_marginals[-1]

    def _add_binary_centroid(self, images, binary_marginals=None):
        if binary_marginals is None:  # assume uniform marginal
            self.transition += (images.shape[0] - 1) / ALPHABET_SIZE ** 2
        else:
            self.transition += np.sum(binary_marginals, axis=0)

    def add_centroid(self, images, marginals=None):
        if marginals is None:  # assume uniform marginal
            self._add_unary_centroid(images, None)
            self._add_binary_centroid(images, None)
        else:
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

    def infer_probabilities(self, images, log):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        umargs, bmargs, log_partition = oracles.chain_sum_product(uscores, bscores, log=log)
        umargs = np.minimum(umargs, 0)
        bmargs = np.minimum(bmargs, 0)
        if log:
            return LogProbability(umargs, bmargs), log_partition
        else:
            return Probability(umargs, bmargs), log_partition

    def word_score(self, images, labels):
        """Return the score <self,F(images,labels)>."""
        return \
            np.sum(images * self.emission[labels]) \
            + np.sum(self.bias[labels, np.zeros(labels.shape[0])]) \
            + np.sum(self.transition[labels[:-1], labels[1:]])

    def predict(self, images):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        return oracles.chain_viterbi(uscores, bscores)

    def prediction_score(self, x, y):
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
    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.emission *= scalar
            self.bias *= scalar
            self.transition *= scalar
        else:
            return Features(self.emission * scalar, self.bias * scalar, self.transition * scalar)

    def combine(self, other, operator):
        emission = operator(self.emission, other.emission)
        bias = operator(self.bias, other.bias)
        transition = operator(self.transition, other.transition)
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
        emissions = parse.letters2wordimage(self.emission)
        plt.matshow(emissions, cmap=cmap)
        ticks_positions = np.linspace(0, emissions.shape[1], ALPHABET_SIZE + 2).astype(int)[1:-1]
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
