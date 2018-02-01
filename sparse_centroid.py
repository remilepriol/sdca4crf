import numpy as np


class SparseCentroid:

    def __init__(self, sentence, marginal):
        alphalen = marginal.nb_class
        # sentence_length, nb_attributes = sentence.shape

        sent = np.empty_like(sentence, dtype=[('attribute', 'i4'), ('word position', 'i2')])
        # first column is the sentence array
        sent['attribute'] = sentence
        # second column is the word number
        sent['word position'] = np.arange(sentence.shape[0])[:, np.newaxis]
        sent = np.ravel(sent)
        sent.sort(order='attribute')
        active_attributes, inverse = np.unique(sent['attribute'], return_inverse=True)
        active_attributes = active_attributes[1:]  # remove zero
        centroid = np.zeros([active_attributes.shape[0], alphalen])
        for index, value in zip(inverse, sent):
            centroid[index] += marginal[sent['word position']]

        # Finally remove the zeros
        self.active = active_attributes[1:]
        self.centroid = centroid[1:]
