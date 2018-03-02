import matplotlib.pyplot as plt
import numpy as np

from sdca4crf.oracles import sequence_sum_product, sequence_viterbi
from sdca4crf.utils import letters2wordimage
from .sequence_marginals import SequenceMarginals


def radius(points_sequence, labels_sequence, data):
    """Return \max_y \|F(x_i,y_i) - F(x_i,y) \|

    :param points_sequence: sequence of points
    :param labels_sequence: sequence of labels
    :param data: data set with the specifications nb_features, nb_labels, is_sparse
    """
    featuremap = Weights(nb_features=data.nb_features, nb_labels=data.nb_labels,
                         is_sparse_features=data.is_sparse)

    # ground truth feature
    featuremap.add_datapoint(points_sequence, labels_sequence)
    featuremap.multiply_scalar(-1, inplace=True)

    # the optimal labels_sequence is made of only one label
    # that is the least present in the true labels_sequence

    ulabels, ucounts = np.unique(labels_sequence, return_counts=True)
    diff_labels = np.setdiff1d(np.arange(data.nb_labels), labels_sequence)
    if len(diff_labels) > 0:
        optlabel = diff_labels[0]
    else:
        optlabel = ulabels[np.argmin(ucounts)]

    optlabels_sequence = optlabel * np.ones_like(labels_sequence)
    featuremap.add_datapoint(points_sequence, optlabels_sequence)

    return np.sqrt(featuremap.squared_norm())


def radii(data):
    rs = np.empty(len(data))
    for i, (points_sequence, labels_sequence) in enumerate(data):
        rs[i] = radius(points_sequence, labels_sequence, data)
    return rs


class Weights:
    """Weights of the CRF model. It is a centroid of features and decomposes the same way.

        Features are composed of:
        - sparse or dense emission features (unary). Add the embeddings for each label. Because
        the weights vector is a centroid, it may not be sparse. Thus we represent emission as a
        dense matrix.
        - dense bias features (unary), which counts the number of apparition of each label,
        (1) in general, (2) at the beginning, (3) at the end.
        - dense transition features (binary), which counts the number of transition between each
        label.

        Caution: for sparse features, some indices are -1 meaning that the attribute is absent.
        They are filtered out in the process of updating the weights.
        """

    def __init__(self, emission=None, bias=None, transition=None,
                 nb_labels=0, nb_features=0, is_sparse_features=False):

        # sparse centroid replace self.emission
        self.emission = np.zeros([nb_labels, nb_features]) if emission is None else emission
        self.bias = np.zeros([nb_labels, 3]) if bias is None else bias
        self.transition = np.zeros([nb_labels, nb_labels]) if transition is None else transition
        self.nb_labels = self.transition.shape[0]

        # are the features dense or sparse?
        self.is_sparse_features = is_sparse_features

    def display(self):
        """Display emission (if dense) bias and transition features as heatmaps."""
        cmap = "Greys"

        if self.is_sparse_features:
            emissions = letters2wordimage(self.emission)
            plt.matshow(emissions, cmap=cmap)
            ticks_positions = np.linspace(0, emissions.shape[1],
                                          self.emission.shape[0] + 2).astype(int)[1:-1]
            plt.xticks(ticks_positions, np.arange(self.emission.shape[0]))
            plt.colorbar(fraction=0.046, pad=0.04)

        plt.matshow(self.transition, cmap=cmap)
        plt.grid()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features", y=1.3)

        rescale_bias = np.array([1 / 23, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features", y=1.15)

        plt.show()

    #########################################
    # Arithmetic operations
    #########################################
    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.emission = scalar * self.emission
            self.bias = scalar * self.bias
            self.transition = scalar * self.transition
        else:
            return Weights(scalar * self.emission, scalar * self.bias, scalar * self.transition,
                           is_sparse_features=self.is_sparse_features)

    def add(self, other):
        emission = self.emission + other.emission
        bias = self.bias + other.bias
        transition = self.transition + other.transition
        return Weights(emission, bias, transition,
                       is_sparse_features=self.is_sparse_features)

    def subtract(self, other):
        emission = self.emission - other.emission
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        return Weights(emission, bias, transition,
                       is_sparse_features=self.is_sparse_features)

    def inner_product(self, other):
        return np.sum(self.emission * other.emission) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)

    def squared_norm(self):
        return np.sum(self.emission ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    #########################################
    # Construction operations
    #########################################
    def add_datapoint(self, points_sequence, labels_sequence):
        for t, label in enumerate(labels_sequence):
            self.bias[label] += [1, t == 0, t == len(labels_sequence) - 1]
            self._add_labeled_point_emission(points_sequence[t], label)
        for t in range(points_sequence.shape[0] - 1):
            self.transition[labels_sequence[t], labels_sequence[t + 1]] += 1

    def _add_labeled_point_emission(self, point, label):
        if self.is_sparse_features:
            self.emission[label, point[point >= 0]] += 1
        else:
            self.emission[label] += point

    def add_centroid(self, xi, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        self._add_centroid_emission(xi, marginals.unary)

        self.bias[:, 0] += np.sum(marginals.unary, axis=0)
        self.bias[:, 1] += marginals.unary[0]
        self.bias[:, 2] += marginals.unary[-1]

        self.transition += np.sum(marginals.binary, axis=0)

    def _add_centroid_emission(self, points_sequence, unary_marginals):
        if self.is_sparse_features:  # slow!
            for point, unimarginal in zip(points_sequence, unary_marginals):
                self.emission[:, point[point >= 0]] += unimarginal[:, np.newaxis]
        else:
            self.emission += np.dot(unary_marginals.T, points_sequence)

    #########################################
    # From weights to probabilities
    #########################################
    def scores(self, points_sequence):
        """Return the scores of points_sequence.

        :param points_sequence: 2d array T*d representing a sequence of data point
        :return: tuple (unary_scores, binary_scores). unary_scores is a 2d array of shape
        T*K, where u(t,k) is the score for point t and label k. binary_scores is a 3d array of
        shape (T-1)*K*K, where u(t,k,k') is the transition score for label k on point t and
        label k' on point t+1.
        """

        if self.is_sparse_features:
            unary_scores = np.empty([points_sequence.shape[0], self.emission.shape[0]])
            for t, point in enumerate(points_sequence):
                unary_scores[t] = self.emission[:, point].sum(axis=1)
        else:
            unary_scores = np.dot(points_sequence, self.emission.T)

        unary_scores += self.bias[:, 0]  # model bias
        unary_scores[0] += self.bias[:, 1]  # beginning of word bias
        unary_scores[-1] += self.bias[:, 2]  # end of word bias

        binary_scores = (points_sequence.shape[0] - 1) * [self.transition]

        return unary_scores, binary_scores

    def infer_probabilities(self, points_sequence):
        uscores, bscores = self.scores(points_sequence)
        umargs, bmargs, log_partition = sequence_sum_product(uscores, bscores)
        umargs = np.minimum(umargs, 0)
        bmargs = np.minimum(bmargs, 0)
        ans = SequenceMarginals(umargs, bmargs, log=True)

        nans = ans.exp()
        assert nans.is_consistent()
        assert nans.is_density(1)

        return ans, log_partition

    def predict(self, points_sequence):
        uscores, bscores = self.scores(points_sequence)
        return sequence_viterbi(uscores, bscores)[0]

    def labeled_sequence_score(self, points_sequence, labels_sequence):
        """Return the score <self,F(points_sequence, labels_sequence)>."""

        if self.is_sparse_features:
            rows_id = np.repeat(labels_sequence, points_sequence.shape[1])
            ans = np.sum(self.emission[rows_id, np.reshape(points_sequence, (-1,))])
        else:
            ans = np.sum(points_sequence * self.emission[labels_sequence])

        ans += np.sum(self.bias[labels_sequence, 0])
        ans += self.bias[labels_sequence[0], 1]
        ans += self.bias[labels_sequence[-1], 2]

        ans += np.sum(self.transition[labels_sequence[:-1], labels_sequence[1:]])

        return ans
