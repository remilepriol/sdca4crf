import numpy as np

from .weights2 import WeightsWithoutEmission


class SparsePrimalDirection(WeightsWithoutEmission):

    def __init__(self, points_sequence, marginals):
        super().__init__(nb_labels=marginals.nb_labels)
        super().add_centroid(points_sequence, marginals)

        self.sparse_emission = SparseEmission(points_sequence, marginals)

    def __imul__(self, scalar):
        super().__imul__(scalar)
        self.sparse_emission *= scalar

    def squared_norm(self):
        ans = super().squared_norm()
        return ans + self.sparse_emission.squared_norm()


class SparseEmission:

    def __init__(self, points_sequence, marginals):
        alphalen = marginals.nb_labels
        sentence_length, nb_attributes = points_sequence.shape

        active_attributes, inverse = np.unique(points_sequence, return_inverse=True)
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        # TODO reshape inverse
        for i, inv in enumerate(inverse):
            centroid[inv] += marginals[i // nb_attributes]

        # Finally remove the zeros
        self.active_set = active_attributes[1:]
        self.values = centroid[1:]

    def __imul__(self, scalar):
        self.values *= scalar

    def squared_norm(self):
        return np.sum(self.values ** 2)
