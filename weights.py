import matplotlib.pyplot as plt
import numpy as np


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
        """

    def __init__(self, emission=None, bias=None, transition=None, nb_labels=0, nb_features=0):
        self.emission = np.zeros([nb_labels, nb_features]) if emission is None else emission
        self.bias = np.zeros([nb_labels, 3]) if bias is None else bias
        self.transition = np.zeros([nb_labels, nb_labels]) if transition is None else transition

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

    #########################################
    # Arithmetic operations
    #########################################
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

    def inner_product(self, other):
        return np.sum(self.emission * other.emission) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)

    #########################################
    # Construction operations
    #########################################
    def _add_labeled_point_emission(self, xi, yit):
        """To be completed by subclass."""
        pass

    def add_datapoint(self, xi, yi):
        for t, label in enumerate(yi):
            self.bias[label] += [1, t == 0, t == len(yi) - 1]
            self._add_labeled_point_emission(xi, label)
        for t in range(xi.shape[0] - 1):
            self.transition[yi[t], yi[t + 1]] += 1

    def _add_unary_centroid(self, xi, unary_marginals):
        # Important part. I hope it works
        for xit, mt in zip(xi, unary_marginals):
            self.emission[:, xit.indices] += mt[:, np.newaxis]

        self.bias[:, 0] += np.sum(unary_marginals, axis=0)
        self.bias[:, 1] += unary_marginals[0]
        self.bias[:, 2] += unary_marginals[-1]

    def _add_centroid_binary(self, binary_marginals):
        self.transition += np.sum(binary_marginals, axis=0)

    def add_centroid(self, xi, marginals):
        if marginals.islog:
            marginals = marginals.exp()
        self._add_unary_centroid(xi, marginals.unary)
        self._add_centroid_binary(marginals.binary)


class WeightsFromDense(Weights):

    def __init__(self):
        pass


class WeightsFromSparse(Weights):

    def __init__(self):
        pass


class Features:

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
    def _add_unary(self, xi, yit, t, emission_counts):
        if emission_counts is None:
            self.emission[yit, xi[t]] += 1
        else:
            emission_counts[yit].extend(xi[t].indices)
        self.bias[yit] += [1, t == 0, t == xi.shape[0] - 1]

    def _add_binary(self, yit, yitp):
        self.transition[yit, yitp] += 1

    def add_datapoint(self, xi, yi, emission_counts=None):
        for t, label in enumerate(yi):
            self._add_unary(xi, label, t, emission_counts)
        for t in range(xi.shape[0] - 1):
            self._add_binary(yi[t], yi[t + 1])

    def _add_unary_centroid(self, xi, unary_marginals):
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
        return oracles.sequence_viterbi(uscores, bscores)[0]
