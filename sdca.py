import numpy as np
from tqdm import tqdm

import parameters
import utils
from monitor import MonitorEpoch, MonitorIteration
from sampler import Sampler


def sdca(features_cls, trainset, regu=1, npass=5, sampler_period=None, precision=1e-5,
         sampling="uniform", non_uniformity=0, step_size=None, warm_start=None, _debug=False,
         logdir=None, testset=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the trainset points x and the labels y.

    Unless warm_start is used, the initial point is a concatenation of the smoothed empirical
    distributions.

    :param features_cls: module corresponding to the dataset, with the relevant alphabet and
    features.
    :param x: trainset points organized by rows
    :param y: labels as a one dimensional array. They should be positive.
    :param regu: value of the l2 regularization parameter
    :param npass: maximum number of pass over the trainset
    duality gaps used in the non-uniform sampling and to get a convergence criterion.
    :param sampler_period: if not None, period to do a full batch update of the duality gaps,
    for the non-uniform sampling. Expressed as a number of epochs. This whole epoch will be
    counted in the number of pass used by sdca.
    :param precision: precision to which we wish to optimize the objective.
    :param sampling: options are "uniform" (default), "importance", "gap", "gap+"
    :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
    :param step_size: if None, SDCA will use a line search. Otherwise should be a positive float
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

    ##################################################################################
    # INITIALIZE : the dual and primal variables
    ##################################################################################
    marginals, weights, ground_truth_centroid = \
        parameters.initialize(warm_start, features_cls, trainset, regu)

    ##################################################################################
    # OBJECTIVES : primal objective, dual objective and duality gaps.
    # I compute every update_period epoch to monitor the evolution.
    ##################################################################################

    monitor = MonitorEpoch(regu, trainset, ground_truth_centroid, weights, marginals, npass,
                           sampler_period, testset, logdir)

    dgaps = 100 * np.ones(trainset.size)  # fake estimate of the duality gaps
    monitor_frequent = MonitorIteration(regu, trainset, weights.squared_norm(), marginals, npass,
                                        dgaps,
                                        logdir)

    # non-uniform sampling TODO move that in sampler
    if sampling == "uniform" or sampling == "gap":
        sampler = Sampler(100 * np.ones(trainset.size))
    elif sampling == "importance" or sampling == "gap+":
        importances = 1 + features_cls.radii(trainset.points, trainset.labels) ** 2 \
                      / trainset.size / regu
        sampler = Sampler(dgaps * importances)
    else:
        raise ValueError(" %s is not a valid argument for sampling" % str(sampling))

    ##################################################################################
    # MAIN LOOP
    ##################################################################################
    for t in tqdm(range(1, trainset.size * npass)):  # TODO print duality gap

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
        # TODO implement this method as a part of the sequence module
        primal_direction = features_cls.Features()
        primal_direction.add_centroid(point_i, dual_direction)
        # Centroid of the corrected features in the dual direction
        # = Centroid of the real features in the opposite of the dual direction
        primal_direction.multiply_scalar(-1 / regu / trainset.size, inplace=True)

        ##################################################################################
        # INTERESTING VALUES and NON-UNIFORM SAMPLING
        ##################################################################################
        primaldir_squared_norm = primal_direction.squared_norm()
        weights_dot_primaldir = weights.inner_product(primal_direction)

        divergence_gap = alpha_i.kullback_leibler(beta_i)
        reverse_gap = beta_i.kullback_leibler(alpha_i)

        if sampling == "gap":
            sampler.update(divergence_gap, i)
        elif sampling == "gap+":
            sampler.update(divergence_gap * importances[i], i)

        ##################################################################################
        # LINE SEARCH : find the optimal step size gammaopt or use a fixed one
        ##################################################################################
        quadratic_coeff = - regu * trainset.size / 2 * primaldir_squared_norm
        linear_coeff = - regu * trainset.size * weights_dot_primaldir

        if step_size:
            gammaopt = step_size
            subobjective = []
        else:  # TODO wrap that up into a module linesearch

            def evaluator(gamma, returnf=False):
                # line search function and its derivatives

                # new marginals
                newmargs = alpha_i.convex_combination(beta_i, gamma)
                # assert newmargs.is_density(1)
                # assert newmargs.is_consistent()

                # first derivative gf
                if gamma == 0:
                    gf = divergence_gap + reverse_gap
                elif gamma == 1:
                    gf = 2 * quadratic_coeff
                else:
                    gf = \
                        divergence_gap \
                        + beta_i.kullback_leibler(newmargs) \
                        - alpha_i.kullback_leibler(newmargs) \
                        + gamma * 2 * quadratic_coeff

                if gf == 0:
                    return gf, 0

                # first derivative divided by the second derivative
                # this is the step size used by Newton-Raphson
                log_ggf = dual_direction \
                    .absolute().log() \
                    .multiply_scalar(2) \
                    .subtract(newmargs) \
                    .log_reduce_exp(- 2 * quadratic_coeff)  # stable log sum exp
                gfdggf = np.log(np.absolute(gf)) - log_ggf  # log(absolute(gf(x)/ggf(x)))
                gfdggf = - np.sign(gf) * np.exp(gfdggf)  # gf(x)/ggf(x)
                if returnf:
                    entropy = newmargs.entropy()
                    norm = gamma ** 2 * quadratic_coeff + gamma * linear_coeff
                    return entropy, norm, gf, gfdggf
                else:
                    return gf, gfdggf

            gammaopt, subobjective = utils.find_root_decreasing(
                evaluator=evaluator, precision=1e-2)

        ##################################################################################
        # UPDATE : the primal and dual coordinates
        ##################################################################################
        marginals[i] = alpha_i.convex_combination(beta_i, gammaopt)
        weights = weights.add(primal_direction.multiply_scalar(gammaopt))

        ##################################################################################
        # ANNEX
        ##################################################################################

        monitor_frequent.frequent_update(i, marginals[i], gammaopt, weights_dot_primaldir,
                                         primaldir_squared_norm, divergence_gap)
        monitor_frequent.log_frequent_tensorboard(t)

        # TODO log all that in monitor
        if t % trainset.size == 0:
            # OBJECTIVES : after every update_period epochs, compute the duality gap
            array_gaps = monitor.full_batch_update(marginals, weights, trainset,
                                                   ground_truth_centroid)
            monitor.test_error(testset, weights)

            if sampler_period is not None and t % (sampler_period * trainset.size) == 0:
                # Non-uniform sampling full batch update:
                monitor_frequent.update_gap_estimate(array_gaps.mean())
                if sampling == "gap":
                    sampler = Sampler(array_gaps)
                elif sampling == "gap+":
                    sampler = Sampler(array_gaps * importances)
                t += trainset.size  # count the full batch in the number of steps

            monitor.log_tensorboard(t)
            monitor.append_results(t)

        if monitor_frequent.duality_gap_estimate < precision:
            break

    ##################################################################################
    # FINISH : convert the objectives to simplify the after process.
    ##################################################################################
    monitor.full_batch_update(marginals, weights, trainset, ground_truth_centroid)
    monitor.save_results()
    monitor_frequent.save()
    return marginals, weights
