import numpy as np


class SparseCentroid:

    def __init__(self, sentence, marginal):
        alphalen = marginal.nb_class
        sentence_length, nb_attributes = sentence.shape

        sent = np.ravel(sentence)
        active_attributes, inverse = np.unique(sent, return_inverse=True)
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        for i, inv in enumerate(inverse):
            centroid[inv] += marginal[i // nb_attributes]

        # Finally remove the zeros
        self.active = active_attributes[1:]
        self.centroid = centroid[1:]
