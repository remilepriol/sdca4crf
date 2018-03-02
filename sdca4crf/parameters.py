import numpy as np

from sdca4crf.dense_weights import DenseWeights
from sdca4crf.sparse_weights import SparsePrimalDirection
from .sequence import dirac


def initialize(warm_start, data, regularization):
    if isinstance(warm_start, np.ndarray):
        # assume that init contains the marginals for a warm start.
        if warm_start.shape[0] != len(data):
            raise ValueError(
                "Not the same number of warm start marginals (%i) and data points (%i)."
                % (warm_start.shape[0], len(data)))
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
            marginals.append(dirac(labels, data.nb_labels))
        marginals = np.array(marginals)

    # Initialize the weights as the centroid of the ground truth features minus the centroid
    # of the features given by the marginals.
    ground_truth_centroid = centroid(data)
    marginals_centroid = centroid(data, marginals)
    weights = ground_truth_centroid - marginals_centroid
    weights *= 1 / regularization

    return marginals, weights, ground_truth_centroid


def centroid(data, marginals=None):
    ans = DenseWeights(nb_features=data.nb_features, nb_labels=data.nb_labels,
                       is_dataset_sparse=data.is_sparse)

    if marginals is None:  # ground truth centroid
        for point, label in data:
            ans.add_datapoint(point, label)
    else:  # marginals centroid
        for (point, _), margs in zip(data, marginals):
            ans.add_centroid(point, margs)

    ans *= 1 / len(data)

    return ans


def compute_primal_direction(points_sequence, dual_direction, is_sparse, nb_samples,
                             regularization):
    primal_direction_cls = SparsePrimalDirection if is_sparse else DenseWeights
    primal_direction = primal_direction_cls.from_marginals(points_sequence, dual_direction)

    # Centroid of the corrected features in the dual direction
    # = Centroid of the real features in the opposite of the dual direction
    primal_direction *= -1 / regularization / nb_samples
