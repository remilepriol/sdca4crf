import numpy as np

from sdca4crf.parameters.weights2 import WeightsWithoutEmission


class SparsePrimalDirection(WeightsWithoutEmission):

    def __init__(self, sparse_emission=None, bias=None, transition=None,
                 nb_labels=0):
        super().__init__(bias, transition, nb_labels)
        self.sparse_emission = sparse_emission

    def __mul__(self, scalar):
        tmp = super().__mul__(scalar)
        return SparsePrimalDirection(scalar * self.sparse_emission, tmp.bias, tmp.transition)

    @classmethod
    def from_marginals(cls, points_sequence, marginals):
        if marginals.islog:
            marginals = marginals.exp()

        ans = cls(nb_labels=marginals.nb_labels)
        ans.add_centroid(points_sequence, marginals)
        ans.sparse_emission = SparseEmission.from_marginals(points_sequence, marginals)
        return ans

    def squared_norm(self):
        ans = super().squared_norm()
        return ans + self.sparse_emission.squared_norm()


class SparseEmission:

    def __init__(self, active_set, values):
        self.active_set = active_set
        self.values = values

    @classmethod
    def from_marginals(cls, points_sequence, marginals):
        alphalen = marginals.nb_labels

        active_attributes, inverse = np.unique(points_sequence, return_inverse=True)
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        inverse = inverse.reshape(points_sequence.shape)
        for inv, marg in zip(inverse, marginals.unary):
            centroid[inv] += marg

        # Finally remove the zeros
        active_set = active_attributes[1:]
        values = np.transpose(centroid[1:])
        return cls(active_set, values)

    def __mul__(self, scalar):
        return SparseEmission(self.active_set, scalar * self.values)

    def squared_norm(self):
        return np.sum(self.values ** 2)
