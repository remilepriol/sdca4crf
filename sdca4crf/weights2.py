import matplotlib.pyplot as plt
import numpy as np

from .oracles import sequence_sum_product, sequence_viterbi
from .sequence import Sequence
from .utils import letters2wordimage


def radius(points_sequence, labels_sequence, data):
    """Return \max_y \|F(x_i,y_i) - F(x_i,y) \|

    :param points_sequence: sequence of points
    :param labels_sequence: sequence of labels
    :param data: data set with the specifications nb_features, nb_labels, is_sparse
    """
    Weights_ = SparseWeights if data.is_sparse else DenseWeights
    featuremap = Weights_(nb_features=data.nb_features, nb_labels=data.nb_labels)

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
    def __init__(self, emission=None, bias=None, transition=None,
                 nb_labels=0, nb_features=0):

        # sparse centroid replace self.emission
        self.emission = np.zeros([nb_labels, nb_features]) if emission is None else emission
        self.nb_features = nb_features
        self.bias = np.zeros([nb_labels, 3]) if bias is None else bias
        self.transition = np.zeros([nb_labels, nb_labels]) if transition is None else transition
        self.nb_labels = nb_labels

    def add_centroid(self, xi, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        self._add_centroid_emission(xi, marginals.unary)

        self.bias[:, 0] += np.sum(marginals.unary, axis=0)
        self.bias[:, 1] += marginals.unary[0]
        self.bias[:, 2] += marginals.unary[-1]

        self.transition += np.sum(marginals.binary, axis=0)

    def _add_centroid_emission(self, xi, marginals_unary):
        NotImplemented

    def _display(self):
        """Display emission (if dense) bias and transition features as heatmaps."""
        cmap = "Greys"

        plt.matshow(self.transition, cmap=cmap)
        plt.grid()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features", y=1.3)

        rescale_bias = np.array([1 / 23, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features", y=1.15)

        plt.show()

    def _scores(self, unary_scores, points_sequence):
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
        ans = Sequence(umargs, bmargs, log=True)

        nans = ans.exp()
        assert nans.is_consistent()
        assert nans.is_density(1)

        return ans, log_partition

    def predict(self, points_sequence):
        uscores, bscores = self.scores(points_sequence)
        return sequence_viterbi(uscores, bscores)[0]


class DenseWeights(Weights):
    def __init__(self, emission=None, bias=None, transition=None, nb_labels=0, nb_features=0, dataset_sparse=False):
        super().__init__(emission=emission, bias=bias, transition=transition, nb_features=nb_features, nb_labels=nb_labels)
        self.dataset_sparse = dataset_sparse  # only needed for weights

    def display(self):
        self._display()

    def squared_norm(self):
        return np.sum(self.emission ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    def _add_labeled_point_emission(self, point, label):
        if self.dataset_sparse:
            self.emission[label, point[point >= 0]] += 1
        else:
            self.emission[label] += point

    def add_datapoint(self, points_sequence, labels_sequence):
        for t, label in enumerate(labels_sequence):
            self.bias[label] += [1, t == 0, t == len(labels_sequence) - 1]
            self._add_labeled_point_emission(points_sequence[t], label)
        for t in range(points_sequence.shape[0] - 1):
            self.transition[labels_sequence[t], labels_sequence[t + 1]] += 1

    def _add_centroid_emission(self, points_sequence, unary_marginals):
        if self.dataset_sparse:  # slow!
            for point, unimarginal in zip(points_sequence, unary_marginals):
                self.emission[:, point[point >= 0]] += unimarginal[:, np.newaxis]
        else:
            self.emission += np.dot(unary_marginals.T, points_sequence)

    def scores(self, points_sequence):
        if self.dataset_sparse:
            unary_scores = np.empty([points_sequence.shape[0], self.emission.shape[0]])
            for t, point in enumerate(points_sequence):
                try:
                    unary_scores[t] = self.emission[:, point].sum(axis=1)
                except:
                    import ipdb; ipdb.set_trace()
        else:
            unary_scores = np.dot(points_sequence, self.emission.T)
        unary_scores, binary_scores = self._scores(unary_scores, points_sequence)
        return unary_scores, binary_scores

    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.emission = scalar * self.emission
            self.bias = scalar * self.bias
            self.transition = scalar * self.transition
        else:
            return DenseWeights(scalar * self.emission, scalar * self.bias, scalar * self.transition, dataset_sparse=self.dataset_sparse)

    def add(self, other):
        if isinstance(other, SparseWeights):
            return other.add_to_dense(self, True)

        emission = self.emission + other.emission
        bias = self.bias + other.bias
        transition = self.transition + other.transition
        return DenseWeights(emission, bias, transition)

    def subtract(self, other):
        if isinstance(other, SparseWeights):
            return other.subtract_to_dense(self, True)
        emission = self.emission - other.emission
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        return DenseWeights(emission, bias, transition, dataset_sparse=self.dataset_sparse)

    def inner_product(self, other):
        if isinstance(other, SparseWeights):
            return other.inner_product_to_dense(self)

        return np.sum(self.emission * other.emission) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)


class SparseWeights(Weights):
    def __init__(self, emission=None, bias=None, transition=None, active=None, nb_labels=0, nb_features=0):
        super().__init__(emission=emission, bias=bias, transition=transition, nb_features=nb_features, nb_labels=nb_labels)
        self.active = active

    def display(self):
        emissions = letters2wordimage(self.emission)
        plt.matshow(emissions, cmap=cmap)
        ticks_positions = np.linspace(0, emissions.shape[1],
                                      self.emission.shape[0] + 2).astype(int)[1:-1]
        plt.xticks(ticks_positions, np.arange(self.emission.shape[0]))
        plt.colorbar(fraction=0.046, pad=0.04)
        self._diplay()

    def scores(self, points_sequence):
        unary_scores = np.empty([points_sequence.shape[0], self.emission.shape[0]])
        for t, point in enumerate(points_sequence):
            # TODO verify with Remi
            unary_scores[t] = self.emission.sum(axis=1)
        unary_scores, binary_scores = self._scores(unary_scores, points_sequence)
        return unary_scores, binary_scores

    def add_datapoint(self, points_sequence, labels_sequence):
        alphalen = self.nb_labels  # marginal.nb_class

        active_attributes, inverse = np.unique(points_sequence, return_inverse=True)
        self.active = active_attributes[1:]
        self.emission = np.zeros([active_attributes.shape[0], alphalen])[1:]

    def _add_centroid_emission(self, points_sequence, unary_marginals):
        alphalen = self.nb_labels  # marginal.nb_class
        sentence_length, nb_attributes = points_sequence.shape

        active_attributes, inverse = np.unique(points_sequence, return_inverse=True)
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        # TODO reshape inverse
        for i, inv in enumerate(inverse):
            centroid[inv] += unary_marginals[i // nb_attributes]

        # Finally remove the zeros
        self.active = active_attributes[1:]
        # TODO: double check with Remi
        self.emission = centroid[1:].T

    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.bias = scalar * self.bias
            self.transition = scalar * self.transition
            self.emission = scalar * self.emission
        else:
            return SparseWeights(scalar * self.emission, scalar * self.bias, scalar * self.transition, self.active, self.nb_labels, self.nb_features)

    def add(self, other):
        if isinstance(other, DenseWeights):
            return self.add_to_dense(other)
        else:
            emission = self.emission + other.emission
            bias = self.bias + other.bias
            transition = self.transition + other.transition
            return SparseWeights(emission, bias, transition, self.active, self.nb_labels, self.nb_features)

    def add_to_dense(self, other, return_dense=False):
        emission = self.emission + other.emission[:, self.active]
        bias = self.bias + other.bias
        transition = self.transition + other.transition
        if return_dense:
            other.emission[:, self.active] = emission
            return DenseWeights(other.emission, bias, transition, dataset_sparse=other.dataset_sparse)
        else:
            return SparseWeights(emission, bias, transition, self.active, self.nb_labels, self.nb_features)

    def subtract(self, other):
        if isinstance(other, DenseWeights):
            emission = self.emission - other.emission[:, self.active]
        else:
            emission = self.emission - other.emission
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        return SparseWeights(emission, bias, transition, self.active, self.nb_labels, self.nb_features)

    def subtract_to_dense(self, other, return_dense=False):
        emission = self.emission - other.emission[:, self.active]
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        if return_dense:
            other.emission[:, self.active] = emission
            return DenseWeights(other.emission, bias, transition, dataset_sparse=other.dataset_sparse)
        else:
            return SparseWeights(emission, bias, transition, self.active, self.nb_labels, self.nb_features)

    def inner_product(self, other):
        if isinstance(other, DenseWeights):
            return self.inner_product_to_dense(other)
        else:
            return np.sum(self.emission * other.emission) + \
                   np.sum(self.bias * other.bias) + \
                   np.sum(self.transition * other.transition)

    def inner_product_to_dense(self, other):
        return np.sum(self.emission * other.emission[:, self.active]) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)

    def squared_norm(self):
        return np.sum(self.emission ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)


class SparseCentroid:

    def __init__(self, sentence, marginal, nb_labels):
        alphalen = nb_labels  # marginal.nb_class
        sentence_length, nb_attributes = sentence.shape

        active_attributes, inverse = np.unique(sentence, return_inverse=True)
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        # TODO reshape inverse
        for i, inv in enumerate(inverse):
            centroid[inv] += marginal[i // nb_attributes]

        # Finally remove the zeros
        self.active = active_attributes[1:]
        self.centroid = centroid[1:]
