import numpy as np

from sequence import dirac


def initialize(warm_start, features_cls, data, regularization):
    if isinstance(warm_start, np.ndarray):
        # assume that init contains the marginals for a warm start.
        if warm_start.shape[0] != data.size:
            raise ValueError(
                "Not the same number of warm start marginals (%i) and data points (%i)."
                % (warm_start.shape[0], data.size))
        marginals = warm_start

    else:  # empirical initialization
        # The empirical marginals give a good value of the dual objective : 0,
        # and primal objective : average sequence length times log alphabet-size = 23
        # but the entropy has an infinite slope and curvature in the corners
        # of the simplex. Hence we take a convex combination between a lot of
        # empirical and a bit of uniform.
        # This is the recommended initialization for online exponentiated
        # gradient in appendix D of the SAG-NUS for CRF paper
        marginals = []
        for imgs, labels in data:
            marginals.append(dirac(labels, features_cls.ALPHALEN))
        marginals = np.array(marginals)

    # Initialize the weights as the centroid of the ground truth features minus the centroid
    # of the features given by the marginals.
    ground_truth_centroid = true_centroid(features_cls, data)
    weights = marginals_centroid(data, features_cls.Features, marginals)

    weights = ground_truth_centroid.subtract(weights)
    weights.multiply_scalar(1 / regularization, inplace=True)

    return marginals, weights, ground_truth_centroid


def true_centroid(features_cls, data):
    ans = features_cls()
    for point, label in data:
        ans.add_datapoint(point, label)

    ans.multiply_scalar(1 / data.size, inplace=True)
    return ans


def marginals_centroid(data, features_cls, marginals=None):
    centroid = features_cls()
    if marginals is None:  # assume uniform
        for point in data.points:
            centroid.add_centroid(point)
    else:
        for point, margs in zip(data.points, marginals):
            centroid.add_centroid(point, margs)

    centroid.multiply_scalar(1 / data.size, inplace=True)
    return centroid
