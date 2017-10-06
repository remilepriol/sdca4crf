import time

import numpy as np
from scipy.misc import logsumexp
from tqdm import tqdm

import random_counters as rc
import utils


def conditional_probabilities(scores, log=False):
    """For each row of scores, return the probability vector that is proportional to exp(score)."""
    log_partitions = logsumexp(scores, axis=-1, keepdims=True)
    scores = scores - log_partitions
    if log:
        return scores, log_partitions
    else:
        return np.exp(scores), np.exp(log_partitions)


class MulticlassLogisticRegression:
    """Contains all the elements of a multinomial logistic classifier.
    This is the same model as neural network with zero hidden layer,
    d input neurons and k output neurons, with an exponential kernel.
    reg : regularization coefficient
    n : number of data points
    k : number of classes
    d : number of dimensions
    weights : classifier's linear parameter. shape (k,d)
    logalpha : log dual probability. shape (n,k)
    """

    def __init__(self, reg, x, y):
        self.reg = reg
        self.n, self.d = x.shape
        self.k = np.amax(y) + 1
        self.weights = np.zeros([self.k, self.d])
        self.logalpha = - np.log(self.k) * np.ones([self.n, self.k])

    ############################################################################
    # Change of variable functions
    ############################################################################

    def scores(self, x):
        """Return the score of each class for each line of x as a len(x)*k matrix."""
        return np.dot(x, self.weights.T)

    def prediction(self, x):
        """Return the most likely class of each data point in x, as an array of size len(x)."""
        return np.argmax(self.scores(x), axis=-1)

    def primal2dual(self, x):
        """Return the probability vector p(.|x_i ; self.weights) for all x_i in x.
        It is also equal to the dual variable given by the optimality condition on weights.
        """
        return conditional_probabilities(self.scores(x))[0]

    def dual2primal(self, x, y):
        """Return the vector weights given by the optimality condition for a given logalpha."""
        w = np.mean((utils.boolean_encoding(y, self.k) - np.exp(self.logalpha))[:, :, np.newaxis] * x[:, np.newaxis, :],
                    axis=0)
        w /= self.reg
        return w

    ############################################################################
    # Objective functions
    ############################################################################

    def negloglikelihood(self, x, y):
        scores = self.scores(x)
        negll = utils.logsumexp(scores)  # log partition function for each data point
        negll -= scores[np.arange(self.n), y]  # subtract the scores of the real classes
        return negll

    def entropy(self, i=None):
        if i is None:
            return -np.sum(np.exp(self.logalpha) * self.logalpha, axis=-1)
        else:
            return -np.sum(np.exp(self.logalpha[i]) * self.logalpha[i])

    def regularization(self):
        return self.reg / 2 * np.sum(self.weights ** 2)

    def primal_objective(self, x, y):
        return self.regularization() + np.mean(self.negloglikelihood(x, y))

    def dual_objective(self, x, y):
        w = self.dual2primal(x, y)
        return -self.reg / 2 * np.sum(w ** 2) + np.mean(self.entropy())

    def duality_gap(self, x, y):
        return self.primal_objective(x, y) - self.dual_objective(x, y)

    def sdca(self, x, y, alpha0=None, npass=50, update_period=5, precision=1e-15, non_uniformity=0,
             sampling_scheme='gaps', _debug=False):
        """Update self.logalpha and self.weights with the stochastic dual coordinate ascent algorithm to fit the model to the
        data points x and the labels y.

        :param x: data points organized by rows
        :param y: labels as a one dimensional array. They should be positive.
        :param alpha0: initial point for logalpha. If None, starts from the previous value of logalpha (warm start).
        :param npass: maximum number of pass over the data
        :param precision: precision to which we wish to optimize the objective.
        :param non_uniformity: between 0 and 1. probability of sampling non-uniformly.
        :param _debug: if true, return a detailed list of objectives
        :return: the list of duality gaps after each pass over the data
        :return: the time after each pass over the data
        """

        ##################################################################################
        # INITIALIZE : the dual and primal variables
        ##################################################################################
        self.n, self.d = x.shape
        if alpha0 is not None:  # cold start
            self.logalpha = np.log(alpha0)
            self.n, self.k = alpha0.shape
            self.weights = self.dual2primal(x, y)

        ##################################################################################
        # OBJECTIVES : initialize the lists to be returned
        ##################################################################################
        if _debug:
            obj = [[self.regularization(), np.mean(self.negloglikelihood(x, y)), np.mean(self.entropy()), 0]]
        else:
            obj = [self.duality_gap(x, y)]
        delta_time = time.time()
        timing = [0]

        ##################################################################################
        # PREPARE for the line search
        ##################################################################################
        squared_norm_x = np.sum(x ** 2, axis=-1)

        ##################################################################################
        # NON-UNIFORM SAMPLING : initialize the sampler
        ##################################################################################
        if non_uniformity > 0:
            sampler = rc.RandomCounters(np.ones(self.n))
        importances = np.sqrt(2 * squared_norm_x + self.n * self.reg * 2)
        duality_gap = 1

        ##################################################################################
        # COUNTERS : to give insights on the algorithm
        ##################################################################################
        countneg = 0
        countpos = 0
        countzero = 0

        ##################################################################################
        # MAIN LOOP
        ##################################################################################
        for t in tqdm(range(npass * self.n)):
            if duality_gap < precision:
                break

            ##################################################################################
            # DRAW : one sample at random.
            ##################################################################################
            if np.random.rand() > non_uniformity:  # then sample uniformly
                i = np.random.randint(self.n)
            else:  # sample proportionally to the duality gaps
                i = sampler.sample()
            logalphai = self.logalpha[i]

            ##################################################################################
            # FUNCTION ESTIMATE : for the ascent
            ##################################################################################
            scores_i = self.scores(x[i])
            logbetai, _ = conditional_probabilities(scores_i, log=True)
            max_proba = np.maximum(logalphai, logbetai)
            ascent_direction = np.exp(max_proba) * (np.exp(logbetai - max_proba) - np.exp(logalphai - max_proba))

            # stopping condition
            individual_gap = utils.log_kullback_leibler(logalphai, logbetai)
            if individual_gap < precision:
                continue
            residue = np.sqrt(np.sum(ascent_direction ** 2))

            ##################################################################################
            # NON-UNIFORM SAMPLING
            ##################################################################################
            if non_uniformity > 0:
                if sampling_scheme == 'gaps':
                    sampler.update(individual_gap, i)
                elif sampling_scheme == 'csiba':
                    sampler.update(importances[i] * residue, i)
                else:
                    raise ValueError("Not a valid sampling scheme. Choose 'csiba' or 'gaps'.")

            ##################################################################################
            # LINE SEARCH : find the optimal logalpha[i]
            ##################################################################################
            squared_ascent_norm = np.sum(ascent_direction ** 2)
            linear_coeff = - squared_norm_x[i] * squared_ascent_norm / self.reg / self.n
            # constant_coeff = np.dot(ascent_direction, scores_i)

            # check that the slope in 0 is big enough as demanded by the theory
            slope = np.dot(ascent_direction, logbetai - logalphai)
            reverse_gap = utils.log_kullback_leibler(logbetai, logalphai)
            assert np.isclose(slope, individual_gap + reverse_gap, atol=precision) \
                   and reverse_gap >= residue ** 2 / 2, print(
                "iteration : %i | data point : %i | slope : %.2e "
                "\n individual gap = %.2e | reverse gap = %.2e | sum = %.2e | residue^2/2 = %.2e" % (
                    t, i, slope, individual_gap, reverse_gap, individual_gap + reverse_gap, residue ** 2 / 2),
                "\n alpha i : ", np.exp(logalphai),
                "\n beta i : ", np.exp(logbetai),
                "\n ascent direction :", ascent_direction,
                "\n slope per class :", ascent_direction * logalphai,
                "\n scores : ", scores_i
            )

            def evaluator(gamma):
                # Evaluate the first and second derivatives of the dual objective with respect to gamma
                # We have to find a root of the first derivative
                newlogproba = logsumexp(a=[logalphai, logbetai], axis=0, b=[[1 - gamma], [gamma]])
                fgamma = np.dot(ascent_direction, logbetai - newlogproba) + gamma * linear_coeff
                gfgamma = logsumexp([logsumexp(np.log(ascent_direction ** 2) - newlogproba), linear_coeff])
                return fgamma, fgamma * np.exp(-gfgamma)

            gammaopt, subobjective = utils.find_root_decreasing(evaluator=evaluator, precision=precision / 100)

            ##################################################################################
            # COUNTERS
            ##################################################################################
            if subobjective[-1] > precision:
                countpos += 1
            elif subobjective[-1] < -precision:
                countneg += 1
            else:
                countzero += 1

            ##################################################################################
            # UPDATE : the primal and dual coordinates
            ##################################################################################
            logalphaibis = logsumexp(a=[logalphai, logbetai], axis=0, b=[[1 - gammaopt], [gammaopt]])
            self.weights += - (np.exp(logalphaibis) - np.exp(logalphai))[:, np.newaxis] * x[i] / self.reg / self.n
            self.logalpha[i] = logalphaibis

            ##################################################################################
            # OBJECTIVES : after each pass over the data, compute the duality gap
            ##################################################################################
            if t % self.n == 0:
                if _debug:  # track every score
                    obj.append(
                        [self.regularization(), np.mean(self.negloglikelihood(x, y)), np.mean(self.entropy()),
                         len(subobjective)])
                ##################################################################################
                # DUALITY GAPS: perform a batch update after every update_period epoch
                # To reduce the staleness for the non-uniform sampling
                # To monitor the objective and provide a stopping criterion
                ##################################################################################
                elif t % (update_period * self.n) == 0:
                    logprobs, _ = conditional_probabilities(self.scores(x), log=True)  # n*k array, like logalpha
                    dual_gaps = utils.log_kullback_leibler(self.logalpha, logprobs, axis=-1)
                    obj.append(np.mean(dual_gaps))
                    if non_uniformity > 0:
                        if sampling_scheme == 'gaps':
                            sampler = rc.RandomCounters(dual_gaps)
                        elif sampling_scheme == 'csiba':
                            residues = np.sqrt(np.sum((np.exp(logprobs) - np.exp(self.logalpha)) ** 2))
                            sampler = rc.RandomCounters(residues * importances)

                else:
                    t1 = time.time()
                    obj.append(self.duality_gap(x, y))
                    t2 = time.time()
                    delta_time += t2 - t1  # Don't count the time spent monitoring the function
                duality_gap = obj[-1]
                timing.append(time.time() - delta_time)

        ##################################################################################
        # COUNTERS
        ##################################################################################
        print("Perfect line search : %i \t Negative ls : %i \t Positive ls : %i" % (countzero, countneg, countpos))

        ##################################################################################
        # FINISH : convert the objectives to simplify the after process.
        ##################################################################################
        obj = np.array(obj)
        timing = np.array(timing)
        return obj, timing
