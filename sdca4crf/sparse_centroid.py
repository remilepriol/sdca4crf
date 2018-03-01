import numpy as np

from .weights import Weights


class SparsePrimalDirection(Weights):
    def __init__(self, bias=None, transition=None, nb_labels=0, nb_features=0):
        super().__init__(bias=bias, transition=transition, nb_features=nb_features, nb_labels=nb_labels)

    def _add_centroid_emission(self, points_sequence, unary_marginals):
        self.emission = SparseCentroid(sentence=points_sequence,
                                       marginal=unary_marginals, nb_labels=self.nb_labels)

    def add_to_dense(self):
        return self.emission

    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.bias = scalar * self.bias
            self.transition = scalar * self.transition
            self.emission.centroid = scalar * self.emission.centroid
        else:
            return -1

    def inner_product(self, other):
        return np.sum(self.emission[:, self.active] * other.emission) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)

    def squared_norm(self):
        import ipdb; ipdb.set_trace()
        return np.sum(self.emission.centroid ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)


class SparseCentroid:

    def __init__(self, sentence, marginal, nb_labels):
        alphalen = nb_labels  # marginal.nb_class
        sentence_length, nb_attributes = sentence.shape

        active_attributes, inverse = np.unique(sentence, return_inverse=True)
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        # TODO reshpae inverse
        for i, inv in enumerate(inverse):
            centroid[inv] += marginal[i // nb_attributes]

        # Finally remove the zeros
        self.active = active_attributes[1:]
        self.centroid = centroid[1:]
