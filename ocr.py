# standard imports
import time

import numpy as np
from tqdm import tqdm

import parse
import random_counters
# custom imports
import utils
from chains import Probability, LogProbability
from constant import MAX_LENGTH
from features import Features


########################################################################################################################
# INITIALIZATION
########################################################################################################################
# initialize with uniform marginals
def uniform_marginals(labels, log=False):
    nb_words = labels.shape[0]
    margs = np.empty(nb_words, dtype=np.object)
    if log:
        references = np.array([LogProbability(word_length=t) for t in range(1, MAX_LENGTH)])
    else:
        references = np.array([Probability(word_length=t) for t in range(1, MAX_LENGTH)])
    for i in range(nb_words):
        margs[i] = references[labels[i].shape[0] - 1]
    return margs


# Or marginals from the ground truth
def empirical_marginals(labels):
    nb_words = labels.shape[0]
    margs = np.empty(nb_words, dtype=np.object)
    for i, lbls in enumerate(labels):
        margs[i] = Probability.dirac(lbls)
    return margs


# Initialize the weights as the centroid of the ground truth features minus the centroid of the
# features given by the uniform marginals.
def marginals_to_features_centroid(images, labels, marginals=None, log=False):
    nb_words = labels.shape[0]

    centroid = Features()
    if marginals is None:  # assume uniform
        for image in images:
            centroid.add_centroid(image)

    elif log:
        for image, margs in zip(images, marginals):
            centroid.add_centroid(image, margs.to_probability())

    else:
        for image, margs in zip(images, marginals):
            centroid.add_centroid(image, margs)

    centroid.multiply_scalar(1 / nb_words, inplace=True)
    return centroid


def dual_score(weights, marginals, regularization_parameter):
    ans = - regularization_parameter / 2 * weights.squared_norm()
    ans += sum([margs.entropy() for margs in marginals]) / marginals.shape[0]
    return ans


def divergence_gaps(marginals, weights, images):
    ans = []
    for margs, imgs in zip(marginals, images):
        newmargs, _ = weights.infer_probabilities(imgs, log=True)
        ans.append(margs.kullback_leibler(newmargs))
    return np.array(ans)


def get_slopes(marginals, weights, images, regu):
    nb_words = marginals.shape[0]
    ans = []

    for margs, imgs in zip(marginals, images):
        newmargs, _ = weights.infer_probabilities(imgs, log=True)
        dual_direction = newmargs.to_probability().subtract(margs.to_probability())
        assert dual_direction.are_densities(integral=0)
        assert dual_direction.are_consistent()

        primal_direction = Features()
        primal_direction.add_centroid(imgs, dual_direction)
        primal_direction.multiply_scalar(-1 / regu / nb_words, inplace=True)

        slope = - dual_direction.inner_product(newmargs) \
                - regu * nb_words * weights.inner_product(primal_direction)
        ans.append(slope)

    return ans


def sdca(x, y, regu=1, npass=5, update_period=5, precision=1e-5, subprecision=1e-16, non_uniformity=0,
         step_size=None, init='uniform', _debug=False):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit the model to the
    data points x and the labels y.


    :param x: data points organized by rows
    :param y: labels as a one dimensional array. They should be positive.
    :param regu: value of the l2 regularization parameter
    :param npass: maximum number of pass over the data
    :param update_period: number of epochs before doing a full batch update of the individual duality gaps used in the
    non-uniform sampling and to get a convergence criterion
    :param precision: precision to which we wish to optimize the objective.
    :param subprecision: precision of the line search method, both on the value of the derivative and on the distance
    of the iterate to the optimum.
    :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
    :param step_size: if None, SDCA will use a line search. Otherwise should be a positive float to be used as the
    constant step size.
    :param init: if init is a numpy array, it will be used as the initial value for the marginals. If it is
    "uniform", the marginals will be initialized with uniform marginals. If it is random, the marginals will be
    initialized with random marginals, by inferring them from a random weights vector.
    :param _debug: if true, return a detailed list of objectives
    :return marginals: optimal value of the marginals
    :return weights: optimal value of the weights
    :return objectives: list of duality gaps, primal objective, dual objective and time point measured after each
    update_period epochs
    :return annex: only if _debug is true, array of useful values taken at each step of the algorithm
    """

    ##################################################################################
    # INITIALIZE : the dual and primal variables
    ##################################################################################
    nb_words = y.shape[0]
    delta_time = time.time()
    if nb_words != x.shape[0]:
        raise ValueError("Not the same number of labels (%i) and images (%i) inside training set."
                         % (nb_words, x.shape[0]))

    if isinstance(init, np.ndarray):  # assume that init contains the marginals for a warm start.
        marginals = init
        weights = marginals_to_features_centroid(x, y, marginals, log=True)
    elif init == "uniform":
        marginals = uniform_marginals(y, log=True)
        weights = marginals_to_features_centroid(x, y, marginals=None)
    elif init == "random":
        weights = Features(random=True)
        marginals = np.array([weights.infer_probabilities(imgs, log=True)[0] for imgs in x])
        weights = marginals_to_features_centroid(x, y, marginals, log=True)
    else:
        raise ValueError("Not a valid argument for init: %r" % init)

    ground_truth_centroid = Features()
    ground_truth_centroid.add_dictionary(x, y)
    ground_truth_centroid.multiply_scalar(1 / nb_words, inplace=True)

    weights = ground_truth_centroid.subtract(weights)
    weights.multiply_scalar(1 / regu, inplace=True)

    ##################################################################################
    # OBJECTIVES : primal objective, dual objective and duality gaps
    ##################################################################################
    def full_batch_update(marginals, weights, weights_squared_norm, dual_objective):
        sum_log_partitions = 0
        dgaps = []
        for margs, imgs in zip(marginals, x):
            newmargs, log_partition = weights.infer_probabilities(imgs, log=True)
            dgaps.append(margs.kullback_leibler(newmargs))
            sum_log_partitions += log_partition

        # Non-uniform sampling : create a new sampler
        sampler = random_counters.RandomCounters(dgaps)

        # update the value of the duality gap
        duality_gap = sampler.get_total() / nb_words

        # calculate the primal score
        primal_objective = \
            regu / 2 * weights_squared_norm \
            + sum_log_partitions / nb_words \
            - weights.inner_product(ground_truth_centroid)

        # check that this is coherent with my values
        # of the duality gap and the dual objective
        assert np.isclose(weights_squared_norm, weights.squared_norm())
        assert np.isclose(duality_gap, primal_objective - dual_objective), print(
            duality_gap, primal_objective, dual_objective, primal_objective - dual_objective
        )

        return duality_gap, primal_objective, sampler

    # first compute the dual objective
    entropies = np.array([margs.entropy() for margs in marginals])
    weights_squared_norm = weights.squared_norm()
    dual_objective = entropies.mean() - regu / 2 * weights_squared_norm

    # then do a whole epoch (of oracle calls)
    duality_gap, primal_objective, sampler = \
        full_batch_update(marginals, weights, weights_squared_norm, dual_objective)

    objectives = [[duality_gap, primal_objective, dual_objective, time.time() - delta_time]]

    ##################################################################################
    # ANNEX : to give insights on the algorithm
    ##################################################################################
    annex = []
    countneg = 0
    countpos = 0
    countzero = 0

    ##################################################################################
    # MAIN LOOP
    ##################################################################################
    for t in tqdm(range(nb_words * npass)):
        if duality_gap < precision:
            break

        ##################################################################################
        # SAMPLING
        ##################################################################################
        if np.random.rand() > non_uniformity:  # then sample uniformly
            i = np.random.randint(nb_words)
        else:  # sample proportionally to the duality gaps
            i = sampler.sample()
        alpha_i = marginals[i]

        ##################################################################################
        # MARGINALIZATION ORACLE and ASCENT DIRECTION (primal to dual)
        ##################################################################################
        beta_i, log_partition_i = weights.infer_probabilities(x[i], log=True)
        nbeta_i = beta_i.to_probability()
        assert nbeta_i.are_consistent()
        assert nbeta_i.are_densities(1), (parse.display_word(y[i], x[i]),
                                          beta_i.display(),
                                          weights.display())

        dual_direction = beta_i.smart_subtract(alpha_i)
        assert dual_direction.are_densities(integral=0)
        assert dual_direction.are_consistent()

        ##################################################################################
        # EXPECTATION of FEATURES (dual to primal)
        ##################################################################################
        primal_direction = Features()
        primal_direction.add_centroid(x[i], dual_direction)

        # Centroid of the corrected features in the dual direction
        # = Centroid of the real features in the opposite of the dual direction
        primal_direction.multiply_scalar(-1 / regu / nb_words, inplace=True)

        ##################################################################################
        # INTERESTING VALUES and NON-UNIFORM SAMPLING
        ##################################################################################
        primaldir_squared_norm = primal_direction.squared_norm()
        weights_dot_primaldir = weights.inner_product(primal_direction)

        divergence_gap = alpha_i.kullback_leibler(beta_i)
        reverse_gap = beta_i.kullback_leibler(alpha_i)

        sampler.update(divergence_gap, i)
        duality_gap_estimate = sampler.get_total() / nb_words

        # Compare the fenchel duality gap and the KL between alpha_i and beta_i.
        # They should be equal.
        # entropy_i = entropies[i]  # entropy of alpha_i
        # weights_i = Features()
        # weights_i.add_centroid(x[i], alpha_i.to_probability())
        # weights_i.multiply_scalar(-1, inplace=True)
        # # weights_i.add_word(x[i], y[i])
        # weights_dot_wi = weights.inner_product(weights_i)
        # fenchel_gap = log_partition_i - entropy_i + weights_dot_wi
        # assert np.isclose(divergence_gap, fenchel_gap), \
        #     print(" iteration %i \n divergence %.5e \n fenchel gap %.5e \n log_partition %f \n"
        #           " entropy %.5e \n w^T A_i alpha_i %f \n reverse divergence %f " % (
        #               t, divergence_gap, fenchel_gap, log_partition_i, entropy_i, weights_dot_wi, reverse_gap))

        ##################################################################################
        # LINE SEARCH : find the optimal step size gammaopt or use a fixed one
        ##################################################################################
        quadratic_coeff = - regu * nb_words / 2 * primaldir_squared_norm
        linear_coeff = - regu * nb_words * weights_dot_primaldir

        if step_size:
            gammaopt = step_size
            subobjective = []
        else:

            def evaluator(gamma, returnf=False):
                # line search function and its derivatives

                # new marginals
                newmargs = alpha_i.convex_combination(beta_i, gamma)
                # assert newmargs.are_densities(1)
                # assert newmargs.are_consistent()
                # assert newmargs.are_positive(), newmargs.display

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
                    .map(np.absolute) \
                    .to_logprobability() \
                    .multiply_scalar(2) \
                    .divide(newmargs) \
                    .logsumexp(- 2 * quadratic_coeff)  # stable logsumexp
                gfdggf = np.log(np.absolute(gf)) - log_ggf  # log(absolute(gf(x)/ggf(x)))
                gfdggf = - np.sign(gf) * np.exp(gfdggf)  # gf(x)/ggf(x)
                if returnf:
                    entropy = newmargs.entropy()
                    norm = gamma ** 2 * quadratic_coeff + gamma * linear_coeff
                    return entropy, norm, gf, gfdggf
                else:
                    return gf, gfdggf

            gammaopt, subobjective = utils.find_root_decreasing(evaluator=evaluator, precision=subprecision)

            ##################################################################################
            # DEBUGGING
            ##################################################################################
            # if t == 50:
            #     return evaluator, quadratic_coeff, linear_coeff, dual_direction, alpha_i, beta_i

            # Plot the line search curves
            # whentoplot = 25
            # if t == whentoplot * nb_words:
            #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            # if 0 <= t - whentoplot * nb_words < 10:
            # if t % int(npass * nb_words / 6) == 0:
            #     segment = np.linspace(0, 1, 100)
            #     funcvalues = np.array([evaluator(gam, returnf=True) for gam in segment]).T
            #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 2))
            #     axs[0].plot(segment, funcvalues[0], label="entropy")
            #     axs[0].plot(segment, funcvalues[1], label="norm")
            #     axs[0].plot(segment, funcvalues[0] + funcvalues[1], label="line search")
            #     axs[0].legend(loc='best')
            #     axs[1].plot(segment, funcvalues[2])
            #     axs[2].plot(segment, funcvalues[3])
            #     for ax in axs:
            #         ax.vlines(gammaopt, *ax.get_ylim())

            ##################################################################################
            # ANNEX
            ##################################################################################
            if _debug:
                if subobjective[-1] > subprecision:
                    countpos += 1
                elif subobjective[-1] < -subprecision:
                    countneg += 1
                else:
                    countzero += 1

        ##################################################################################
        # UPDATE : the primal and dual coordinates
        ##################################################################################
        marginals[i] = alpha_i.convex_combination(beta_i, gammaopt)
        weights = weights.add(primal_direction.multiply_scalar(gammaopt, inplace=False))

        ##################################################################################
        # ANNEX
        ##################################################################################
        # Update the weights norm, the dual objective and the entropy
        norm_update = \
            gammaopt * 2 * weights_dot_primaldir \
            + gammaopt ** 2 * primaldir_squared_norm
        weights_squared_norm += norm_update
        dual_objective += -regu / 2 * norm_update

        tmp = marginals[i].entropy()
        dual_objective += \
            (tmp - entropies[i]) / nb_words
        entropies[i] = tmp

        if _debug:
            # Append relevant variables
            annex.append([
                np.log10(primaldir_squared_norm),
                np.log10(weights_squared_norm),
                np.arccos(weights_dot_primaldir / np.sqrt(primaldir_squared_norm) / np.sqrt(
                    weights_squared_norm)),
                dual_objective,
                np.log10(duality_gap_estimate),
                divergence_gap,
                gammaopt,
                i,
                len(subobjective)
            ])

        if t % (update_period * nb_words) == 0:
            ##################################################################################
            # OBJECTIVES : after each update_period epochs, compute the duality gap
            ##################################################################################
            duality_gap, primal_objective, sampler = \
                full_batch_update(marginals, weights, weights_squared_norm, dual_objective)

            objectives.append([duality_gap, primal_objective, dual_objective, time.time() - delta_time])

    ##################################################################################
    # ANNEX
    ##################################################################################
    if _debug and not step_size:
        print("Perfect line search : %i \t Negative ls : %i \t Positive ls : %i" % (countzero, countneg, countpos))

    ##################################################################################
    # FINISH : convert the objectives to simplify the after process.
    ##################################################################################
    objectives = np.array(objectives)
    if _debug:
        annex = np.array(annex)
        return marginals, weights, objectives, annex
    else:
        return marginals, weights, objectives
