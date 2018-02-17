import numpy as np


class LabeledSequenceData:
    """Store sequential data in a 2D array, with labels in a 1D array.

    Each data point is a row.
    Each sequence is represented as a contiguous block.
    Provide an iterator over the sequences.
    """
    # TODO include the bias in x?

    def __init__(self, points, labels, size=None):

        # store the data
        self.points = points
        self.labels = labels.astype(np.int8) - 1
        # -1 to account for Matlab format

        # get the start and end indices of each sequence.
        self.ends = np.where(labels == 0)[0]
        self.starts = np.empty_like(self.ends)
        self.starts[0] = 0
        self.starts[1:] = self.ends[:-1] + 1

        # trim to the desired number of sequences
        if size is not None:
            self.trim(size)

        # evaluate the important sizes.
        self.nb_sequences = self.starts.shape[0]
        self.nb_points = self.labels.shape[0]
        # important : take all the labels to evaluate the alphabet
        self.alphabet_size = labels.max()

        # initialize the iterator
        self.index = -1

        self.is_consistent()

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index >= self.nb_sequences:
            self.index = -1
            raise StopIteration
        return self.get_item(self.index)

    def __len__(self):
        return self.nb_sequences

    def get_item(self, i):
        return self.get_point(i), self.get_label(i)

    def get_point(self, i):
        return self.points[self.starts[i]:self.ends[i]]

    def get_label(self, i):
        return self.labels[self.starts[i]:self.ends[i]]

    def is_consistent(self):
        if self.nb_points != self.points.shape[0]:
            raise ValueError(
                "Not the same number of labels (%i) and data points (%i) inside training set."
                % (self.nb_points, self.points.shape[0]))
        return True

    def trim(self, size):
        self.starts = self.starts[:size]
        self.ends = self.ends[:size]
        self.points = self.points[:self.ends[-1] + 1]
        self.labels = self.labels[:self.ends[-1] + 1]


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

    def __init__(self, points, labels, size, vocabularies_sizes=None):

        if vocabularies_sizes is None:  # training set
            self.vocabularies_sizes = VocabularySize(points)
        else:  # test set
            self.vocabularies_sizes = vocabularies_sizes

        super(SparseLabeledSequenceData, self).__init__(points, labels, size)

        # total number of different attribute values
        self.vocabulary_size = vocabularies_sizes.total

        # convert to NaN non-existing attributes and attributes absent from the training set.
        points = points.astype(float)
        points[np.logical_or(points == 0, points > vocabularies_sizes.by_attribute)] = np.nan

        # shift dictionary value for each attribute
        points[:, 1:] += vocabularies_sizes.cumsum[:-1]

        # convert back to zero the absent attributes
        self.points = np.nan_to_num(points).astype(np.int32)
