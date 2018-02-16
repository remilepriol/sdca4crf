import numpy as np


class LabeledSequenceData:
    # TODO include the bias in x?

    def __init__(self, points, labels):
        self.index = -1
        self.points = points
        self.labels = labels - 1  # to account for matlab format
        self.size = labels.shape[0]

        self.ends = np.where(self.labels == -1)[0]
        self.starts = np.empty_like(self.ends)
        self.starts[0] = 0
        self.starts[1:] = self.ends[:-1] + 1

        self.is_consistent()  # check that everything is correct

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index >= self.size:
            self.index = -1
            raise StopIteration
        return self.get_item(self.index)

    def get_item(self, i):
        return self.get_point(i), self.get_label(i)

    def get_point(self, i):
        return self.points[self.starts[i]:self.ends[i]]

    def get_label(self, i):
        return self.labels[self.starts[i]:self.ends[i]]

    def is_consistent(self):
        if self.size != self.points.shape[0]:
            raise ValueError(
                "Not the same number of labels (%i) and data points (%i) inside training set."
                % (self.size, self.points.shape[0]))
        return True


class VocabularySize:
    """ Represent the size of the vocabulary for attributes of sparse data.

    Each point is represented by attributes.
    Each kind of attribute has its own vocabulary, meaning that for a given attribute the
    value taken by this attribute is a number that varies between 1 and the size of the
    attribute vocabulary. It is worth 0 if that attribute is not defined for a given point.
    """

    def __init__(self, points):
        self.by_attribute = np.amax(points, axis=0)
        self.cumsum = np.cumsum(self.by_attribute)
        self.total = self.cumsum[-1]


class SparseLabeledSequenceData(LabeledSequenceData):
    """Instance of LabeledSequenceData where each row contains the index of the active features
    of a point.
    """

    def __init__(self, points, labels, vocabularies_sizes=None):
        super(SparseLabeledSequenceData, self).__init__(points, labels)

        if vocabularies_sizes is None:  # training set
            self.vocabularies_sizes = VocabularySize(points)
        else:  # test set
            self.vocabularies_sizes = vocabularies_sizes

        # total number of different attribute values
        self.vocabulary_size = vocabularies_sizes.total

        # convert to NaN non-existing attributes and attributes absent from the training set.
        points = points.astype(float)
        points[np.logical_or(points == 0, points > vocabularies_sizes.by_attribute)] = np.nan

        # shift dictionary value for each attribute
        points[:, 1:] += vocabularies_sizes.cumsum[:-1]

        # convert back to zero the absent attributes
        self.points = np.nan_to_num(points).astype(np.int32)
