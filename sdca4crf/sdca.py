import numpy as np
from tqdm import tqdm

import parameters
from line_search import LineSearch
from monitor import MonitorAllObjectives, MonitorDualObjective, MonitorDualityGapEstimate, \
    are_consistent, initialize_tensorboard
from sampler_wrap import SamplerWrap
from weights2 import SparseWeights, DenseWeights
import time
import tensorboard_logger as tl


def sdca(trainset, testset=None, args=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the trainset points x and the labels y.

    Unless warm_start is used, the initial point is a concatenation of the smoothed empirical
    distributions.

    :param trainset: training set in the LabeledSequenceData format
    :param testset: testing set in the LabeledSequenceData format. We test the prediction after
    every epochs.

    :return: tuple, optimal value of the (weights, marginals)
    """

    # INITIALIZE : the dual and primal variables
    marginals, weights, ground_truth_centroid = \
        parameters.initialize(args.warm_start, trainset, args.regularization)

    # OBJECTIVES : primal objective, dual objective and duality gaps.
    use_tensorboard = initialize_tensorboard(args.logdir)

    monitor_all_objectives = MonitorAllObjectives(args.regularization, weights, marginals,
                                                  ground_truth_centroid, trainset, testset,
                                                  use_tensorboard)

    monitor_dual_objective = MonitorDualObjective(args.regularization, weights, marginals)

    gaps_array = 100 * np.ones(len(trainset))  # fake estimate of the duality gaps
    monitor_gap_estimate = MonitorDualityGapEstimate(gaps_array)

    # non-uniform sampling
    sampler = SamplerWrap(args.sampling_scheme, args.non_uniformity,
                          gaps_array, trainset, args.regularization)

    try:

        ##################################################################################
        # MAIN LOOP
        ##################################################################################
        start = None
        for step in range(1, len(trainset) * args.npass + 1):
            if step % 100 == 0:
                sparsity = np.mean(weights.emission < 1e-10)
                tl.log_value('sparsity_coefficient', sparsity, step)
                tl.log_histogram('weight matrix', weights.emission.tolist(), step)
                if start is not None:
                    end = time.time()
                    tl.log_value("iteration per second", 100/(end-start), step)
                start = time.time()

            # SAMPLING
            i = sampler.sample()
            alpha_i = marginals[i]
            point_i = trainset.get_point(i)

            # MARGINALIZATION ORACLE
            beta_i, log_partition_i = weights.infer_probabilities(point_i)
            # ASCENT DIRECTION (primal to dual)
            log_dual_direction, signs_dual_direction = beta_i.logsubtractexp(alpha_i)
            dual_direction = log_dual_direction.exp().multiply(signs_dual_direction)

            # EXPECTATION of FEATURES (dual to primal)
            # TODO keep the primal direction sparse
            # TODO implement this method as dual_direction.expected_features()
            Weights_ = SparseWeights if trainset.is_sparse else DenseWeights
            primal_direction = Weights_(
                nb_features=trainset.nb_features,
                nb_labels=trainset.nb_labels,
            )
            primal_direction.add_centroid(point_i, dual_direction)
            # Centroid of the corrected features in the dual direction
            # = Centroid of the real features in the opposite of the dual direction
            primal_direction.multiply_scalar(-1 / args.regularization / len(trainset),
                                             inplace=True)

            # DUALITY GAP
            divergence_gap = alpha_i.kullback_leibler(beta_i)
            monitor_gap_estimate.update(i, divergence_gap)
            sampler.update(i, divergence_gap)

            # LINE SEARCH : find the optimal step size or use a fixed one
            # Update the dual objective monitor as well
            line_search = LineSearch(weights, primal_direction,
                                     log_dual_direction,
                                     alpha_i, beta_i, divergence_gap,
                                     args.regularization, len(trainset))

            if args.fixed_step_size is not None:
                optimal_step_size = args.fixed_step_size
            else:
                optimal_step_size = line_search.run()

            # UPDATE : the primal and dual coordinates
            marginals[i] = alpha_i.convex_combination(beta_i, optimal_step_size)
            weights = weights.add(
                primal_direction.multiply_scalar(optimal_step_size))
            # TODO sparsify update
            monitor_dual_objective.update(i, marginals[i].entropy(),
                                          line_search.norm_update(optimal_step_size))

            # ANNEX
            if use_tensorboard and step % 20 == 0:
                monitor_dual_objective.log_tensorboard(step)
                monitor_gap_estimate.log_tensorboard(step)
                if args.fixed_step_size is None:
                    line_search.log_tensorboard(step)

            if step % len(trainset) == 0:
                # OBJECTIVES : after every epochs, compute the duality gap
                # Update the sampler if necessary (count_time==True)
                count_time = args.sampler_period is not None and step % (
                        args.sampler_period * len(trainset)) == 0

                gaps_array = monitor_all_objectives.full_batch_update(
                    weights, marginals, step, count_time)

                assert are_consistent(monitor_dual_objective, monitor_all_objectives)

                if count_time:
                    step += len(trainset)  # count the full batch in the number of steps
                    monitor_gap_estimate = MonitorDualityGapEstimate(gaps_array)
                    sampler.full_update(gaps_array)

            # STOP condition
            if monitor_gap_estimate.get_value() < args.precision:
                monitor_all_objectives.full_batch_update(weights, marginals, step,
                                                         count_time=False)
                break

    finally:  # save results no matter what.
        monitor_all_objectives.save_results(args.logdir)

    return weights, marginals
