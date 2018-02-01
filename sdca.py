import numpy as np
from tqdm import tqdm

import parameters
from line_search import LineSearch
from monitor import MonitorAllObjectives, MonitorDualObjective, MonitorDualityGapEstimate, \
    are_consistent, initialize_tensorboard
from sampler_wrap import SamplerWrap


def sdca(features_cls, trainset, testset=None, regularization=1, npass=5, sampler_period=None,
         precision=1e-5, sampling_scheme="uniform", non_uniformity=0, fixed_step_size=None,
         warm_start=None, logdir=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the trainset points x and the labels y.

    Unless warm_start is used, the initial point is a concatenation of the smoothed empirical
    distributions.

    :param features_cls: module corresponding to the dataset, with the relevant alphabet and
    features.
    :param trainset: training set in the LabeledSequenceData format
    :param regularization: value of the l2 regularization parameter
    :param npass: maximum number of pass over the trainset
    duality gaps used in the non-uniform sampling and to get a convergence criterion.
    :param sampler_period: if not None, period to do a full batch update of the duality gaps,
    for the non-uniform sampling. Expressed as a number of epochs. This whole epoch will be
    counted in the number of pass used by sdca.
    :param precision: precision to which we wish to optimize the objective.
    :param sampling_scheme: options are "uniform" (default), "importance", "gap", "gap+"
    :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
    :param fixed_step_size: if None, SDCA will use a line search. Otherwise should be a positive
    float
    to be used as the constant step size.
    :param warm_start: if numpy array, used as marginals to start from.
    :param logdir: if nont None, use logdir to dump values for tensorboard
    :param testset: testing set in the LabeledSequenceData format. We test the prediction after
    every epochs.

    :return marginals: optimal value of the marginals
    :return weights: optimal value of the weights
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

    gaps_array = 100 * np.ones(trainset.size)  # fake estimate of the duality gaps
    monitor_gap_estimate = MonitorDualityGapEstimate(gaps_array)

    # non-uniform sampling
    sampler = SamplerWrap(sampling_scheme, non_uniformity,
                          gaps_array, features_cls, trainset, regularization)

    try:

        ##################################################################################
        # MAIN LOOP
        ##################################################################################
        progress_bar = tqdm(range(1, trainset.size * npass + 1))
        for step in progress_bar:
            if step % 100 == 0:
                progress_bar.set_description(
                    "Duality gap estimate: %e" % monitor_gap_estimate.get_value())

            # SAMPLING
            i = sampler.sample()
            alpha_i = marginals[i]
            point_i = trainset.get_point(i)

            # MARGINALIZATION ORACLE
            beta_i, log_partition_i = weights.infer_probabilities(point_i)
            # ASCENT DIRECTION (primal to dual)
            # TODO use a log value and scip's logsumexp with signs.
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
            sampler.update(i, divergence_gap)

            # LINE SEARCH : find the optimal step size or use a fixed one
            # Update the dual objective monitor as well
            line_search = LineSearch(weights, primal_direction, dual_direction, alpha_i, beta_i,
                                     divergence_gap, regularization, trainset.size)

            if fixed_step_size is not None:
                optimal_step_size = fixed_step_size
            else:
                optimal_step_size = line_search.run()

            # UPDATE : the primal and dual coordinates
            marginals[i] = alpha_i.convex_combination(beta_i, optimal_step_size)
            weights = weights.add(
                primal_direction.multiply_scalar(optimal_step_size))  # TODO sparsify update
            monitor_dual_objective.update(i, marginals[i].entropy(),
                                          line_search.norm_update(optimal_step_size))

            # ANNEX
            if use_tensorboard and step % 20 == 0:
                monitor_dual_objective.log_tensorboard(step)
                monitor_gap_estimate.log_tensorboard(step)
                if fixed_step_size is None:
                    line_search.log_tensorboard(step)

            if step % trainset.size == 0:
                # OBJECTIVES : after every epochs, compute the duality gap
                # Update the sampler if necessary (count_time==True)
                count_time = sampler_period is not None and step % (
                        sampler_period * trainset.size) == 0

                gaps_array = monitor_all_objectives.full_batch_update(
                    weights, marginals, step, count_time)

                assert are_consistent(monitor_dual_objective, monitor_all_objectives)

                if count_time:
                    step += trainset.size  # count the full batch in the number of steps
                    monitor_gap_estimate = MonitorDualityGapEstimate(gaps_array)
                    sampler.full_update(gaps_array)

            # STOP condition
            if monitor_gap_estimate.get_value() < precision:
                monitor_all_objectives.full_batch_update(weights, marginals, step,
                                                         count_time=False)
                break

    finally:  # save results no matter what.
        monitor_all_objectives.save_results(logdir)

    return weights, marginals
