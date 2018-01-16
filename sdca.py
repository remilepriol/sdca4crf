# standard imports
import time

import numpy as np
import tensorboard_logger as tl
from tqdm import tqdm

# custom imports
import utils
from random_counters import RandomCounters
from sequence import dirac


# Initialize the weights as the centroid of the ground truth features minus the centroid of the
# features given by the uniform marginals.
def marginals_to_features_centroid(images, labels, features_cls, marginals=None):
    nb_words = labels.shape[0]

    centroid = features_cls()
    if marginals is None:  # assume uniform
        for image in images:
            centroid.add_centroid(image)
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
        newmargs, _ = weights.infer_probabilities(imgs)
        ans.append(margs.kullback_leibler(newmargs))
    return np.array(ans)


def sdca(features_module, x, y, regu=1, npass=5, monitoring_period=5, sampler_period=None,
         precision=1e-5,
         sampling="uniform", non_uniformity=0, step_size=None,
         warm_start=None, _debug=False, logdir=None, xtest=None, ytest=None):
    """Update alpha and weights with the stochastic dual coordinate ascent algorithm to fit
    the model to the data points x and the labels y.

    Unless warm_start is used, the initial point is a concatenation of the smoothed empirical
    distributions.

    :param features_module: module corresponding to the dataset, with the relevant alphabet and
    features.
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
    :param sampling: options are "uniform" (default), "importance", "gap", "gap+"
    :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
    :param step_size: if None, SDCA will use a line search. Otherwise should be a positive float
    to be used as the constant step size.
    :param warm_start: if numpy array, used as marginals to start from.
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
        raise ValueError(
            "Not the same number of labels (%i) and data points (%i) inside training set."
            % (nb_words, x.shape[0])
        )

    if isinstance(warm_start, np.ndarray):
        # assume that init contains the marginals for a warm start.
        if warm_start.shape[0] != x.shape[0]:
            raise ValueError(
                "Not the same number of warm start marginals (%i) and data points (%i)."
                % (warm_start.shape[0], x.shape[0])
            )
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
        for imgs, labels in zip(x, y):
            marginals.append(dirac(labels, features_module.ALPHALEN))
        marginals = np.array(marginals)

    ground_truth_centroid = features_module.Features()
    ground_truth_centroid.add_dataset(x, y)
    ground_truth_centroid.multiply_scalar(1 / nb_words, inplace=True)
    weights = marginals_to_features_centroid(x, y, features_module.Features,
                                             marginals=marginals)
    weights = ground_truth_centroid.subtract(weights)
    weights.multiply_scalar(1 / regu, inplace=True)

    ##################################################################################
    # OBJECTIVES : primal objective, dual objective and duality gaps.
    # I compute every update_period epoch to monitor the evolution.
    ##################################################################################

    def monitor_full_batch(marginals, weights, weights_squared_norm, dual_objective):
        sum_log_partitions = 0
        dgaps = []
        for margs, imgs in zip(marginals, x):
            newmargs, log_partition = weights.infer_probabilities(imgs)
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
        loss01, loss_hamming = weights.prediction_loss(xtest, ytest)
        objs.extend([loss01, loss_hamming])

    objectives = [objs]

    # tensorboard_logger commands
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
    if sampling == "uniform" or sampling == "gap":
        sampler = RandomCounters(100 * np.ones(nb_words))
    elif sampling == "importance" or sampling == "gap+":
        importances = 1 + features_module.radii(x, y) ** 2 / nb_words / regu
        sampler = RandomCounters(100 * importances)
    else:
        raise ValueError(" %s is not a valid argument for sampling" % str(sampling))

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
        beta_i, log_partition_i = weights.infer_probabilities(x[i])
        nbeta_i = beta_i.exp()
        assert nbeta_i.is_consistent()
        assert nbeta_i.is_density(1)

        dual_direction = beta_i.subtract_exp(alpha_i)
        assert dual_direction.is_density(integral=0)
        assert dual_direction.is_consistent()

        ##################################################################################
        # EXPECTATION of FEATURES (dual to primal)
        ##################################################################################
        primal_direction = features_module.Features()
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

            duality_gap_estimate = duality_gap

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
                loss01, loss_hamming = weights.prediction_loss(xtest, ytest)
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
