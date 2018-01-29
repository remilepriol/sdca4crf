class Data:
    # TODO replace data as the raw array n*(nb_attributes)
    # TODO include the bias in x?

    def __init__(self, points, labels):
        self.index = -1
        self.points = points
        self.labels = labels

        self.size = labels.shape[0]
        if self.size != points.shape[0]:
            raise ValueError(
                "Not the same number of labels (%i) and data points (%i) inside training set."
                % (self.size, points.shape[0]))

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index >= self.size:
            self.index = -1
            raise StopIteration
        return self.points[self.index], self.labels[self.index]

    def get_point(self, i):
        return self.points[i]

    def get_label(self, i):
        return self.labels[i]
