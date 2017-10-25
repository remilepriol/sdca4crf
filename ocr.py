# standard imports
import importlib
import time

import numpy as np
import tensorboard_logger as tl
from tqdm import tqdm

# custom imports
import oracles
import parse
import utils
from chains import LogProbability, Probability
from constant import ALPHABET_SIZE, MAX_LENGTH
from features import Features, radii
from random_counters import RandomCounters


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


def sdca(x, y, regu=1, npass=5, monitoring_period=5, sampler_period=None, precision=1e-5,
         subprecision=1e-2, sampling="uniform", non_uniformity=0, step_size=None, init='uniform',
         _debug=False, logdir=None, xtest=None, ytest=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the data points x and the labels y.

    :param x: data points organized by rows
    :param y: labels as a one dimensional array. They should be positive.
    :param regu: value of the l2 regularization parameter
    :param npass: maximum number of pass over the data
    :param monitoring_period: number of epochs before doing a full batch update of the individual
    duality gaps used in the non-uniform sampling and to get a convergence criterion.
    :param sampler_period: if not None, period to do a full batch update of the duality gaps,
    for the non-uniform sampling. Expressed as a number of epochs. This whole epoch will be
    counted in the number of pass used by sdca.
    :param precision: precision to which we wish to optimize the objective.
    :param subprecision: precision of the line search method, both on the value of the derivative
    and on the distance of the iterate to the optimum.
    :param sampling: options are "uniform" (default), "importance", "gap", "gap+"
    :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
    :param step_size: if None, SDCA will use a line search. Otherwise should be a positive float
    to be used as the constant step size.
    :param init: if init is a numpy array, it will be used as the initial value for the marginals.
    If it is "uniform", the marginals will be initialized with uniform marginals. If it is
    random, the marginals will be initialized with random marginals, by inferring them from a
    random weights vector.
    :param _debug: if true, return a detailed list of objectives
    :param logdir: if nont None, use logdir to dump values for tensorboard
    :param xtest: data points to test the prediction every few epochs
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
    nb_words = y.shape[0]
    delta_time = time.time()
    if nb_words != x.shape[0]:
        raise ValueError("Not the same number of labels (%i) and images (%i) inside training set."
                         % (nb_words, x.shape[0]))

    ground_truth_centroid = Features()
    ground_truth_centroid.add_dictionary(x, y)
    ground_truth_centroid.multiply_scalar(1 / nb_words, inplace=True)

    if isinstance(init, np.ndarray):  # assume that init contains the marginals for a warm start.
        marginals = init
        weights = marginals_to_features_centroid(x, y, marginals, log=True)
        weights = ground_truth_centroid.subtract(weights)
    elif init == "uniform":
        marginals = uniform_marginals(y, log=True)
        weights = marginals_to_features_centroid(x, y, marginals=None)
        weights = ground_truth_centroid.subtract(weights)
    elif init == "empirical":
        # The empirical marginals give a good value of the dual objective : 0,
        # but they entropy has an infinite slope and curvature in the corners
        # of the simplex. Hence we take a convex combination between a lot of
        # empirical and a bit of uniform.
        uniformization_value = 1e-5

        unimargs = uniform_marginals(y, log=True)
        empimargs = empirical_marginals(y)
        marginals = np.array([empi
                             .to_logprobability()
                             .convex_combination(uni, uniformization_value)
                              for uni, empi in zip(unimargs, empimargs)])
        del unimargs, empimargs

        weights = marginals_to_features_centroid(x, y, marginals=None)
        weights = ground_truth_centroid.subtract(weights)
        weights.multiply_scalar(uniformization_value, inplace=True)
    elif init == "OEG":
        # implement the recommanded initialization for improved EOG as
        # recommanded by appendix D in SAG4CRF paper
        marginals = []
        for imgs, labels in zip(x, y):
            uscores = np.zeros([imgs.shape[0], ALPHABET_SIZE])
            # import pdb
            # pdb.set_trace()
            uscores[np.arange(imgs.shape[0]), labels] = 10
            bscore = np.zeros([ALPHABET_SIZE, ALPHABET_SIZE])
            bscore[labels[:-1], labels[1:]] = 10
            bscores = (imgs.shape[0] - 1) * [bscore]
            umargs, bmargs, _ = oracles.chain_sum_product(uscores, bscores, log=True)
            marginals.append(LogProbability(unary=umargs, binary=bmargs))
        marginals = np.asarray(marginals)
        weights = marginals_to_features_centroid(x, y, marginals=marginals, log=True)
        weights = ground_truth_centroid.subtract(weights)

    elif init == "random":
        weights = Features(random=True)
        marginals = np.array([weights.infer_probabilities(imgs, log=True)[0] for imgs in x])
        weights = marginals_to_features_centroid(x, y, marginals, log=True)
        weights = ground_truth_centroid.subtract(weights)
    else:
        raise ValueError("Not a valid argument for init: %r" % init)

    weights.multiply_scalar(1 / regu, inplace=True)

    ##################################################################################
    # OBJECTIVES : primal objective, dual objective and duality gaps.
    # I compute every update_period epoch to monitor the evolution.
    ##################################################################################

    def monitor_full_batch(marginals, weights, weights_squared_norm, dual_objective):
        sum_log_partitions = 0
        dgaps = []
        for margs, imgs in zip(marginals, x):
            newmargs, log_partition = weights.infer_probabilities(imgs, log=True)
            dgaps.append(margs.kullback_leibler(newmargs))
            sum_log_partitions += log_partition

        # update the value of the duality gap
        dgaps = np.array(dgaps)
        duality_gap = np.sum(dgaps) / nb_words

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

        return duality_gap, primal_objective, dgaps

    # compute the dual objective
    entropies = np.array([margs.entropy() for margs in marginals])
    weights_squared_norm = weights.squared_norm()
    dual_objective = entropies.mean() - regu / 2 * weights_squared_norm

    # do a whole epoch (of oracle calls) to monitor
    t1 = time.time()
    duality_gap, primal_objective, _ = \
        monitor_full_batch(marginals, weights, weights_squared_norm, dual_objective)
    delta_time += time.time() - t1

    dgaps = 100 * np.ones(nb_words)  # fake estimate of the duality gaps
    duality_gap_estimate = 100

    objs = [duality_gap, primal_objective, dual_objective,
            0, time.time() - delta_time]

    # compute the test error if a test set is present:
    do_test = xtest is not None and ytest is not None
    if do_test:
        loss01, loss_hamming = weights.prediction_score(xtest, ytest)
        objs.extend([loss01, loss_hamming])

    objectives = [objs]

    # tensorboard_logger commands
    importlib.reload(tl)
    if logdir is not None:
        tl.configure(logdir=logdir, flush_secs=15)

        tl.log_value("log10 duality gap", np.log10(duality_gap), step=0)
        tl.log_value("primal objective", primal_objective, step=0)
        tl.log_value("dual objective", dual_objective, step=0)
        if do_test:
            tl.log_value("01 loss", loss01, step=0)
            tl.log_value("hamming loss", loss_hamming, step=0)

    # annex to give insights on the algorithm
    annex = []

    # non-uniform sampling
    importances = 1 + radii(x) ** 2 / nb_words / regu
    if sampling == "importance" or sampling == "gap+":
        sampler = RandomCounters(100 * importances)
    else:
        sampler = RandomCounters(100 * np.ones(nb_words))

    ##################################################################################
    # MAIN LOOP
    ##################################################################################
    for t in tqdm(range(1, nb_words * npass)):

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

        if sampling == "gap":
            sampler.update(divergence_gap, i)
        elif sampling == "gap+":
            sampler.update(divergence_gap * importances[i], i)

        duality_gap_estimate += (divergence_gap - dgaps[i]) / nb_words
        dgaps[i] = divergence_gap

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

            gammaopt, subobjective = utils.find_root_decreasing(evaluator=evaluator,
                                                                precision=subprecision)

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

        similarity = weights_dot_primaldir / np.sqrt(primaldir_squared_norm) / np.sqrt(
            weights_squared_norm)

        if logdir is not None and t % 10 == 0:
            tl.log_value("log10 primaldir_squared_norm", np.log10(primaldir_squared_norm), step=t)
            tl.log_value("weights_squared_norm", weights_squared_norm, t)
            tl.log_value("normalized weights dot primaldir", similarity, t)
            tl.log_value("dual objective", dual_objective, t)
            tl.log_value("log10 duality gap estimate", np.log10(duality_gap_estimate), t)
            tl.log_value("log10 individual gap", np.log10(divergence_gap), t)
            tl.log_value("step size", gammaopt, t)
            tl.log_value("number of line search step", len(subobjective), t)

        if _debug and t % 10 == 0:
            # Append relevant variables
            annex.append([
                np.log10(primaldir_squared_norm),
                weights_squared_norm,
                similarity,
                dual_objective,
                np.log10(duality_gap_estimate),
                np.log10(divergence_gap),
                gammaopt,
                i,
                len(subobjective),
                t
            ])

        updated = False
        if sampler_period is not None and t % (sampler_period * nb_words) == 0:
            # Non-uniform sampling full batch update:
            duality_gap, primal_objective, dgaps = monitor_full_batch(
                marginals, weights, weights_squared_norm, dual_objective)

            if sampling == "gap":
                sampler = RandomCounters(dgaps)
            elif sampling == "gap+":
                sampler = RandomCounters(dgaps * importances)

            updated = True  # avoid doing a full batch update twice just to monitor.

        if t % (monitoring_period * nb_words) == 0:
            ##################################################################################
            # OBJECTIVES : after every update_period epochs, compute the duality gap
            ##################################################################################

            if not updated:
                t1 = time.time()
                duality_gap, primal_objective, _ = monitor_full_batch(
                    marginals, weights, weights_squared_norm, dual_objective)
                delta_time += time.time() - t1

            objs = [duality_gap, primal_objective, dual_objective, t, time.time() - delta_time]

            if updated:  # if we updated the sampler
                t += nb_words  # count the full batch in the number of steps

            if do_test:
                loss01, loss_hamming = weights.prediction_score(xtest, ytest)
                objs.extend([loss01, loss_hamming])

            objectives.append(objs)

            if logdir is not None:

                tl.log_value("log10 duality gap", np.log10(duality_gap), step=t)
                tl.log_value("primal objective", primal_objective, step=t)

                if do_test:
                    tl.log_value("01 loss", loss01, step=t)
                    tl.log_value("hamming loss", loss_hamming, step=t)

            if duality_gap < precision:
                break

    ##################################################################################
    # FINISH : convert the objectives to simplify the after process.
    ##################################################################################
    objectives = np.array(objectives)
    if _debug:
        annex = np.array(annex)
        return marginals, weights, objectives, annex
    else:
        return marginals, weights, objectives
