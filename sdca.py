import numpy as np
from tqdm import tqdm

import parameters
from line_search import LineSearch
from monitor import MonitorAllObjectives, MonitorDualObjective, MonitorDualityGapEstimate, \
    are_consistent, initialize_tensorboard
from sampler import Sampler


def sdca(features_cls, trainset, regularization=1, npass=5, sampler_period=None, precision=1e-5,
         sampling="uniform", non_uniformity=0, fixed_step_size=None, warm_start=None, _debug=False,
         logdir=None, testset=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the trainset points x and the labels y.

    Unless warm_start is used, the initial point is a concatenation of the smoothed empirical
    distributions.

    :param features_cls: module corresponding to the dataset, with the relevant alphabet and
    features.
    :param x: trainset points organized by rows
    :param y: labels as a one dimensional array. They should be positive.
    :param regularization: value of the l2 regularization parameter
    :param npass: maximum number of pass over the trainset
    duality gaps used in the non-uniform sampling and to get a convergence criterion.
    :param sampler_period: if not None, period to do a full batch update of the duality gaps,
    for the non-uniform sampling. Expressed as a number of epochs. This whole epoch will be
    counted in the number of pass used by sdca.
    :param precision: precision to which we wish to optimize the objective.
    :param sampling: options are "uniform" (default), "importance", "gap", "gap+"
    :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
    :param fixed_step_size: if None, SDCA will use a line search. Otherwise should be a positive
    float
    to be used as the constant step size.
    :param warm_start: if numpy array, used as marginals to start from.
    :param _debug: if true, return a detailed list of objectives
    :param logdir: if nont None, use logdir to dump values for tensorboard
    :param xtest: trainset points to test the prediction every few epochs
    :param ytest: labels to test the prediction every few epochs

    :return marginals: optimal value of the marginals
    :return weights: optimal value of the weights
    :return objectives: list of duality gaps, primal objective, dual objective and time point
    measured after each update_period epochs
    :return annex: only if _debug is true, array of useful values taken at each step of the
    algorithm
    """

    # INITIALIZE : the dual and primal variables
    marginals, weights, ground_truth_centroid = \
        parameters.initialize(warm_start, features_cls, trainset, regularization)

    # OBJECTIVES : primal objective, dual objective and duality gaps.
    use_tensorboard = initialize_tensorboard(logdir)

    monitor_all_objectives = MonitorAllObjectives(regularization, weights, marginals,
                                                  ground_truth_centroid, trainset, testset,
                                                  use_tensorboard)

    monitor_dual_objective = MonitorDualObjective(regularization, weights, marginals)

    dgaps = 100 * np.ones(trainset.size)  # fake estimate of the duality gaps
    monitor_gap_estimate = MonitorDualityGapEstimate(dgaps)

    # non-uniform sampling
    if sampling == "uniform" or sampling == "gap":
        sampler = Sampler(100 * np.ones(trainset.size))
    elif sampling == "importance" or sampling == "gap+":
        importances = 1 + features_cls.radii(trainset.points, trainset.labels) ** 2 \
                      / trainset.size / regularization
        sampler = Sampler(dgaps * importances)
    else:
        raise ValueError(" %s is not a valid argument for sampling" % str(sampling))

    ##################################################################################
    # MAIN LOOP
    ##################################################################################
    progress_bar = tqdm(range(1, trainset.size * npass + 1))
    for step in progress_bar:
        if step % 500 == 0:
            progress_bar.set_description(
                "Duality gap estimate: %e" % monitor_gap_estimate.get_value())

        # SAMPLING
        i = sampler.mixed_sample(non_uniformity)
        alpha_i = marginals[i]
        point_i = trainset.get_points(i)

        # MARGINALIZATION ORACLE
        beta_i, log_partition_i = weights.infer_probabilities(point_i)
        # ASCENT DIRECTION (primal to dual)
        dual_direction = beta_i.subtract_exp(alpha_i)

        # EXPECTATION of FEATURES (dual to primal)
        # TODO keep the primal direction sparse
        # TODO implement this method as dual_direction.features_expectation()
        primal_direction = features_cls.Features()
        primal_direction.add_centroid(point_i, dual_direction)
        # Centroid of the corrected features in the dual direction
        # = Centroid of the real features in the opposite of the dual direction
        primal_direction.multiply_scalar(-1 / regularization / trainset.size, inplace=True)

        # DUALITY GAP
        divergence_gap = alpha_i.kullback_leibler(beta_i)
        monitor_gap_estimate.update(i, divergence_gap)
        if sampling == "gap":
            sampler.update(divergence_gap, i)
        elif sampling == "gap+":
            sampler.update(divergence_gap * importances[i], i)

        # LINE SEARCH : find the optimal step size or use a fixed one
        # Update the dual objective monitor as well
        line_search = LineSearch(weights, primal_direction, dual_direction, alpha_i, beta_i,
                                 divergence_gap, regularization, trainset.size, fixed_step_size,
                                 monitor_dual_objective, i)
        newmarg, optimal_step_size = line_search.run()

        # UPDATE : the primal and dual coordinates
        marginals[i] = newmarg
        weights = weights.add(
            primal_direction.multiply_scalar(optimal_step_size))  # TODO sparsify update

        # ANNEX
        if use_tensorboard and step % 20 == 0:
            monitor_dual_objective.log_tensorboard(step)
            monitor_gap_estimate.log_tensorboard(step)
            line_search.log_tensorboard(step)

        if step % trainset.size == 0:
            # OBJECTIVES : after every epochs, compute the duality gap
            count_time = sampler_period is not None and step % (
                    sampler_period * trainset.size) == 0
            gaps_array = monitor_all_objectives.full_batch_update(
                weights, marginals, step, count_time)
            assert are_consistent(monitor_dual_objective, monitor_all_objectives)

            monitor_all_objectives.save_results(logdir)

            if count_time:
                # Non-uniform sampling full batch update:
                monitor_gap_estimate = MonitorDualityGapEstimate(gaps_array)
                if sampling == "gap":
                    sampler = Sampler(gaps_array)
                elif sampling == "gap+":
                    sampler = Sampler(gaps_array * importances)
                step += trainset.size  # count the full batch in the number of steps

        # STOP condition
        if monitor_gap_estimate.get_value() < precision:
            monitor_all_objectives.full_batch_update(weights, marginals, step, count_time=False)
            monitor_all_objectives.save_results(logdir)
            break

    return marginals, weights
