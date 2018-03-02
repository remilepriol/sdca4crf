import matplotlib.pyplot as plt
import numpy as np

from .oracles import sequence_sum_product, sequence_viterbi
from .sequence import Sequence
from .utils import letters2wordimage


def radius(points_sequence, labels_sequence, data):
    """Return \max_y \|F(x_i,y_i) - F(x_i,y) \|.

    :param points_sequence: sequence of points
    :param labels_sequence: sequence of labels
    :param data: data set with the specifications nb_features, nb_labels, is_sparse
    """
    weights_cls = SparseWeights if data.is_sparse else DenseWeights
    featuremap = weights_cls(nb_features=data.nb_features, nb_labels=data.nb_labels)

    # ground truth feature
    featuremap.add_datapoint(points_sequence, labels_sequence)

    # the optimal labels_sequence is made of only one label
    # that is the least present in the true labels_sequence
    # First find the labels present in the sequence
    ulabels, ucounts = np.unique(labels_sequence, return_counts=True)
    # Second find the labels absent in the sequence
    diff_labels = np.setdiff1d(np.arange(data.nb_labels), labels_sequence)
    if len(diff_labels) > 0:  # if there are some take one
        optlabel = diff_labels[0]
    else:  # else find the label which appears the least
        optlabel = ulabels[np.argmin(ucounts)]

    # Finally create a sequence with only this label
    optlabels_sequence = optlabel * np.ones_like(labels_sequence)
    # Add it to the featuremap
    featuremap *= -1
    featuremap.add_datapoint(points_sequence, optlabels_sequence)

    return np.sqrt(featuremap.squared_norm())


def radii(data):
    """Return an array with the radius of the corrected features for the input data points."""
    rs = np.empty(len(data))
    for i, (points_sequence, labels_sequence) in enumerate(data):
        rs[i] = radius(points_sequence, labels_sequence, data)
    return rs


class WeightsWithoutEmission:
    """Base class for the weights of the CRF. Include bias and transition weights.
    The emission weights are dealt with in the subclasses DenseWeights and SparseWeights.
    """

    def __init__(self, bias=None, transition=None, nb_labels=0):
        self.bias = np.zeros([nb_labels, 3]) if bias is None else bias
        self.transition = np.zeros([nb_labels, nb_labels]) if transition is None else transition
        self.nb_labels = self.transition.shape[0]

    # BUILD THE WEIGHTS FROM DATA
    def add_datapoint(self, points_sequence, labels_sequence):
        # TODO profile to decide if should be improved
        for t, label in enumerate(labels_sequence):
            self.bias[label] += [1, t == 0, t == len(labels_sequence) - 1]
        for t in range(labels_sequence.shape[0] - 1):
            self.transition[labels_sequence[t], labels_sequence[t + 1]] += 1

    def add_centroid(self, points_sequence, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        self.bias[:, 0] += np.sum(marginals.unary, axis=0)
        self.bias[:, 1] += marginals.unary[0]
        self.bias[:, 2] += marginals.unary[-1]

        self.transition += np.sum(marginals.binary, axis=0)

    # USE THE MODEL ON DATA
    def scores(self, points_sequence):
        seq_len = points_sequence.shape[0]

        unary_scores = np.zeros([seq_len, self.nb_labels])
        unary_scores += self.bias[:, 0]  # model bias
        unary_scores[0] += self.bias[:, 1]  # beginning of word bias
        unary_scores[-1] += self.bias[:, 2]  # end of word bias

        binary_scores = (seq_len - 1) * [self.transition]

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

    def labeled_sequence_score(self, points_sequence, labels_sequence):
        """Return the score <self,F(points_sequence, labels_sequence)>."""
        ans = np.sum(self.bias[labels_sequence, 0])
        ans += self.bias[labels_sequence[0], 1]
        ans += self.bias[labels_sequence[-1], 2]

        ans += np.sum(self.transition[labels_sequence[:-1], labels_sequence[1:]])

        return ans

    # ARITHMETIC OPERATIONS
    def __imul__(self, scalar):
        self.bias *= scalar
        self.transition *= scalar

    def __add__(self, other):
        return WeightsWithoutEmission(
            bias=self.bias + other.bias,
            transition=self.transition + other.transition)

    def __sub__(self, other):
        return WeightsWithoutEmission(
            bias=self.bias - other.bias,
            transition=self.transition - other.transition)

    def squared_norm(self):
        return np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    def inner_product(self, other):
        return np.sum(self.bias * other.bias) \
               + np.sum(self.transition * other.transition)

    # MISCELLANEOUS
    def display(self):
        """Display bias and transition features as heatmaps."""
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


class DenseWeights(WeightsWithoutEmission):
    """Implement the weights of the model.

    Support all the operations necessary for the CRF and the optimization.
    """

    def __init__(self, emission=None, bias=None, transition=None,
                 nb_labels=0, nb_features=0, is_dataset_sparse=False):

        super().__init__(bias=bias, transition=transition, nb_labels=nb_labels)

        self.is_dataset_sparse = is_dataset_sparse  # only needed for weights
        self.emission = np.zeros([nb_labels, nb_features]) if emission is None else emission

    def display(self):
        super().display()
        if self.is_dataset_sparse:
            emissions = letters2wordimage(self.emission)
            plt.matshow(emissions, cmap="Greys")
            ticks_positions = np.linspace(0, emissions.shape[1],
                                          self.emission.shape[0] + 2).astype(int)[1:-1]
            plt.xticks(ticks_positions, np.arange(self.emission.shape[0]))
            plt.colorbar(fraction=0.046, pad=0.04)

    def add_datapoint(self, points_sequence, labels_sequence):
        super().add_datapoint(points_sequence, labels_sequence)

        if self.is_dataset_sparse:
            for point, label in zip(points_sequence, labels_sequence):
                self.emission[label, point[point >= 0]] += 1
        else:
            for point, label in zip(points_sequence, labels_sequence):
                self.emission[label] += point

    def add_centroid(self, points_sequence, marginals):
        super().add_centroid(points_sequence, marginals)

        if self.is_dataset_sparse:  # slow?
            for point, unimarginal in zip(points_sequence, marginals.unary):
                self.emission[:, point[point >= 0]] += unimarginal[:, np.newaxis]
        else:
            self.emission += np.dot(marginals.unary.T, points_sequence)

    def scores(self, points_sequence):
        unary_scores, binary_scores = super().scores(points_sequence)

        if self.is_dataset_sparse:  # slow?
            for t, point in enumerate(points_sequence):
                unary_scores[t] += self.emission[:, point].sum(axis=1)
        else:
            unary_scores = np.dot(points_sequence, self.emission.T)

        return unary_scores, binary_scores

    def __imul__(self, scalar):
        super().__imul__(scalar)
        self.emission *= scalar

    def __add__(self, other):
        if isinstance(other, SparseWeights):
            return other.add_to_dense(self, True)
        tmp = super().__add__(other)
        emission = self.emission + other.emission
        return DenseWeights(emission, tmp.bias, tmp.transition,
                            is_dataset_sparse=self.is_dataset_sparse)

    def __sub__(self, other):
        tmp = super().__sub__(other)
        emission = self.emission - other.emission
        return DenseWeights(emission, tmp.bias, tmp.transition,
                            is_dataset_sparse=self.is_dataset_sparse)

    def squared_norm(self):
        return super().squared_norm() + np.sum(self.emission ** 2)

    def inner_product(self, other):
        if isinstance(other, SparseWeights):
            return other.inner_product_to_dense(self)

        return np.sum(self.emission * other.emission) + super().inner_product()


class SparseWeights(WeightsWithoutEmission):
    def __init__(self, emission=None, bias=None, transition=None, active=None, nb_labels=0,
                 nb_features=0):
        super().__init__(emission=emission, bias=bias, transition=transition,
                         nb_features=nb_features, nb_labels=nb_labels)
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
            return SparseWeights(scalar * self.emission, scalar * self.bias,
                                 scalar * self.transition, self.active, self.nb_labels,
                                 self.nb_features)

    def add(self, other):
        if isinstance(other, DenseWeights):
            return self.add_to_dense(other)
        else:
            emission = self.emission + other.emission
            bias = self.bias + other.bias
            transition = self.transition + other.transition
            return SparseWeights(emission, bias, transition, self.active, self.nb_labels,
                                 self.nb_features)

    def add_to_dense(self, other, return_dense=False):
        emission = self.emission + other.emission[:, self.active]
        bias = self.bias + other.bias
        transition = self.transition + other.transition
        if return_dense:
            other.emission[:, self.active] = emission
            return DenseWeights(other.emission, bias, transition,
                                is_dataset_sparse=other.dataset_sparse)
        else:
            return SparseWeights(emission, bias, transition, self.active, self.nb_labels,
                                 self.nb_features)

    def subtract(self, other):
        if isinstance(other, DenseWeights):
            emission = self.emission - other.emission[:, self.active]
        else:
            emission = self.emission - other.emission
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        return SparseWeights(emission, bias, transition, self.active, self.nb_labels,
                             self.nb_features)

    def subtract_to_dense(self, other, return_dense=False):
        emission = self.emission - other.emission[:, self.active]
        bias = self.bias - other.bias
        transition = self.transition - other.transition
        if return_dense:
            other.emission[:, self.active] = emission
            return DenseWeights(other.emission, bias, transition,
                                is_dataset_sparse=other.dataset_sparse)
        else:
            return SparseWeights(emission, bias, transition, self.active, self.nb_labels,
                                 self.nb_features)

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
