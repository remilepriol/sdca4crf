import numpy as np


class LabeledSequenceData:
    # TODO include the bias in x?

    def __init__(self, points, labels):
        self.index = -1
        self.points = points
        self.labels = labels
        self.size = labels.shape[0]

        self.ends = np.where(labels == 0)[0]
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
