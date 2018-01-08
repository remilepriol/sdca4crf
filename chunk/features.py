from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

import oracles
from sequence import Sequence

ALPHABET = [
    'I-INTJ', 'I-CONJP', 'O', 'I-ADVP', 'B-VP', 'B-LST', 'I-LST', 'I-PRT', 'I-SBAR', 'I-UCP',
    'B-CONJP', 'I-PP', 'B-ADVP', 'I-NP', 'I-VP', 'B-ADJP', 'B-PP', 'B-NP', 'I-ADJP', 'B-PRT',
    'B-SBAR', 'B-UCP', 'B-INTJ'
]
ALPHALEN = len(ALPHABET)
TAG2INT = {tag: i for i, tag in enumerate(ALPHABET)}
TEMPLATES = (
    ((0, -2),),
    ((0, -1),),
    ((0, 0),),
    ((0, 1),),
    ((0, 2),),
    ((0, -1), (0, 0)),
    ((0, 0), (0, 1)),
    ((1, -2),),
    ((1, -1),),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
    ((1, -2), (1, -1)),
    ((1, -1), (1, 0)),
    ((1, 0), (1, 1)),
    ((1, 1), (1, 2)),
    ((1, -2), (1, -1), (1, 0)),
    ((1, -1), (1, 0), (1, 1)),
    ((1, 0), (1, 1), (1, 2)),
)


def get_unary_attributes(sentence, t):
    keys = []
    for template in TEMPLATES:
        values = []
        for (field, offset) in template:
            p = t + offset
            if p < 0 or p >= len(sentence):
                values = []
                break
            values.append(sentence[p][field])
        if values:
            keys.append("%s=%s" % (str(template), '|'.join(values)))
    return keys


def radius(*args):
    """Not implemented yet."""
    return 1


def radii(words, labels):
    warn("Not implemented yet.")
    rs = np.empty_like(words)
    for i, (word, label) in enumerate(zip(words, labels)):
        rs[i] = radius(word, label)
    return rs


class Features:
    """Features associated to a sample and a label.
    Taking the centroid of such features gives the weights of the primal model.
    
    Features are composed of:
    - sparse emission features (unary), which counts the number of apparitions of a each
    attribute for each tag.
    - dense bias features (unary), which counts the number of apparition of each tag, 
    (1) in general, (2) at the beginning, (3) at the end.
    - dense transition features (binary), which counts the number of transition between every tags.
    
    Attributes are defined following the TEMPLATES. 
    Each template is made of a list of (field, offset).
    The field can be either 0 or 1, respectively 'word' and 'part of speech tag'.
    The offset is relative to the position of the word at stake.
    """

    def __init__(self, emission=None, bias=None, transition=None):

        if emission is None:
            self.emission = np.empty(ALPHALEN, dtype=object)
            for tag in range(ALPHALEN):
                self.emission[tag] = dict()
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
    def _add_unary(self, sentence, label, t):
        d = self.emission[label]
        for key in get_unary_attributes(sentence, t):
            d[key] = d.get(key, 0) + 1
        self.bias[label] += [1, t == 0, t == len(sentence) - 1]

    def _add_binary(self, label, next_label):
        self.transition[label, next_label] += 1

    def add_word(self, sentence, labels):
        if len(sentence) != len(labels):
            raise ValueError("Not the same number of tags (%i) and words (%i) in sentence."
                             % (len(sentence), len(labels)))
        for t, label in enumerate(labels):
            self._add_unary(sentence, label, t)
        for t in range(len(sentence) - 1):
            self._add_binary(labels[t], labels[t + 1])

    def add_dictionary(self, images_set, labels_set):
        if len(images_set) != len(labels_set):
            raise ValueError("Not the same number of labels (%i) and data points (%i)."
                             % (len(images_set), len(labels_set)))
        for images, labels in zip(images_set, labels_set):
            self.add_word(images, labels)

    def _add_unary_centroid(self, sentence, unary_marginals):
        for t, marg in enumerate(unary_marginals):
            for key in get_unary_attributes(sentence, t):
                for label, dic in enumerate(self.emission):
                    dic[key] = dic.get(key, 0) + marg[label]

        self.bias[:, 0] += np.sum(unary_marginals, axis=0)
        self.bias[:, 1] += unary_marginals[0]
        self.bias[:, 2] += unary_marginals[-1]

    def _add_binary_centroid(self, images, binary_marginals):
        self.transition += np.sum(binary_marginals, axis=0)

    def add_centroid(self, images, marginals):
        if marginals.islog:
            marginals = marginals.exp()
        self._add_unary_centroid(images, marginals.unary)
        self._add_binary_centroid(images, marginals.binary)

    #########################################
    # From weights to probabilities
    #########################################
    def unary_scores(self, sentence):
        """Return the unary scores of word when self encode the weights of the model.

        :param sentence: T*d, each line is a letter image.
        :return: unary scores T*K, each case is a score for one image and one label.
        """
        uscores = np.empty([len(sentence), ALPHALEN])
        for t in range(len(sentence)):
            for tag in range(ALPHALEN):
                uscores[t, tag] = sum(
                    [self.emission[tag].get(key, 0) for key in get_unary_attributes(sentence, t)])
        uscores += self.bias[:, 0]  # model bias
        uscores[0] += self.bias[:, 1]  # beginning of word bias
        uscores[-1] += self.bias[:, 2]  # end of word bias
        return uscores

    def binary_scores(self, sentence):
        """Return the binary scores of a word when self encode the weights of the model.

        :param sentence:
        :return: binary scores (T-1)*K*K, each case is the transition score between two labels
        for a given position.
        """
        return (len(sentence) - 1) * [self.transition]

    def infer_probabilities(self, images):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        umargs, bmargs, log_partition = oracles.sequence_sum_product(uscores, bscores)
        umargs = np.minimum(umargs, 0)
        bmargs = np.minimum(bmargs, 0)
        return Sequence(umargs, bmargs, log=True), log_partition

    def word_score(self, sentence, labels):
        """Return the score <self,F(images,labels)>."""
        ans = 0
        for t, label in enumerate(labels):
            d = self.emission[label]
            ans += sum([d.get(key, 0) for key in get_unary_attributes(sentence, t)])
        ans += np.sum(self.bias[labels, 0])
        ans += self.bias[labels[0], 1]
        ans += self.bias[labels[-1], 2]
        ans += np.sum(self.transition[labels[:-1], labels[1:]])
        return ans

    def predict(self, images):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
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
        emission = np.empty(ALPHALEN, dtype=object)
        for tag in range(ALPHALEN):
            d = emission[tag] = dict()
            d1 = self.emission[tag]
            for key, value in d1.items():
                emission[tag][key] = ufunc(value)

        if inplace:
            self.emission = emission
            self.bias = ufunc(self.bias)
            self.transition = ufunc(self.transition)
        else:
            return Features(emission, ufunc(self.bias), ufunc(self.transition))

    def multiply_scalar(self, scalar, inplace=False):
        func = lambda x: scalar * x
        return self.map(func, inplace)

    def combine(self, other, ufunc):
        emission = np.empty(ALPHALEN, dtype=object)
        for tag in range(ALPHALEN):
            d = emission[tag] = dict()
            d1 = self.emission[tag]
            d2 = other.emission[tag]
            for key in set(d1).union(d2):
                d[key] = ufunc(d1.get(key, 0), d2.get(key, 0))
        bias = ufunc(self.bias, other.bias)
        transition = ufunc(self.transition, other.transition)
        return Features(emission, bias, transition)

    def add(self, other):
        return self.combine(other, np.add)

    def subtract(self, other):
        return self.combine(other, np.subtract)

    def reduce(self):
        """Return the addition of the features."""
        ans = 0
        for dic in self.emission:
            ans += sum(dic.values())
        ans += self.bias.sum()
        ans += self.transition.sum()
        return ans

    def squared_norm(self):
        return self.map(lambda x: x ** 2).reduce()

    def inner_product(self, other):
        return self.combine(other, np.multiply).reduce()

    def display(self):
        """Display bias and transition features as heatmaps."""
        cmap = "Greys"
        plt.matshow(self.transition, cmap=cmap)
        plt.grid()
        tags_range = range(ALPHALEN)
        plt.xticks(tags_range, ALPHALEN, rotation='vertical')
        plt.yticks(tags_range, ALPHALEN)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features", y=1.3)

        rescale_bias = np.array([1 / 23, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.xticks(tags_range, ALPHALEN)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features", y=1.15)
