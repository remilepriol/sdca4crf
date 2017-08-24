import time

import numpy as np

import random_counters as rc


# take care to stabilize the log sum exp
def logsumexp(v):
    vmax = np.amax(v, axis=-1, keepdims=True)
    return vmax.squeeze(axis=-1) + np.log(np.sum(np.exp(v - vmax), axis=-1))


def optnewton(func, grad, init, lowerbound, upperbound, precision=1e-12, max_iter=50):
    x = init
    fx = func(x)
    obj = [fx]
    count = 0
    while np.absolute(fx) > precision and count < max_iter:  # stop condition to avoid cycling over an extremity of 0,1?
        count += 1
        gx = grad(x)
        x -= fx / gx
        # Make sure x is in (lower bound, upper bound)
        x = max(lowerbound + precision, x)
        x = min(upperbound - precision, x)
        fx = func(x)
        obj.append(fx)
    return x, obj


def optlinesearch(alphai, deltai, a, b, precision, plot=False):
    """Return the alpha maximizing the score along the ascent direction deltai, and starting from alphai.
    The function to maximize is concave, thus its derivative u/n is decreasing, thus the border conditions."""

    def u(gamma):
        return -np.sum(deltai * np.log(np.maximum(alphai + gamma * deltai, 1e-16))) - gamma * a + b

    u0 = u(0)
    if u0 <= precision:  # 0 is optimal
        return 0, [u0]

    u1 = u(1)
    if u1 >= -precision:  # 1 is optimal
        return 1, [u1]

    def gu(gamma):
        return -np.sum(deltai ** 2 / np.maximum(alphai + gamma * deltai, 1e-16)) - a

    return optnewton(u, gu, .5, 0, 1, precision=precision)


def boolean_encoding(y, k):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1  #
    return ans


def conditional_probabilities(scores):
    """For each row of scores, return the probability vector that is proportional to exp(score)."""
    smax = np.amax(scores, axis=-1, keepdims=True)  # mode of the score for each x
    prob = np.exp(scores - smax)  # stabilized exponential
    partition_functions = np.sum(prob, axis=-1, keepdims=True)
    prob /= partition_functions
    return prob, partition_functions


def kullback_leibler(p, q):
    eps = 1e-50
    return np.maximum(0, np.sum(p * np.log(np.maximum(p, eps) / np.maximum(q, eps)), axis=-1))


class MulticlassLogisticRegression:
    """Contains all the elements of a multinomial logistic classifier.
    This is the same model as neural network with zero hidden layer,
    d input neurons and k output neurons, with an exponential kernel.
    reg : regularization coefficient
    n : number of data points
    k : number of classes
    d : number of dimensions
    w : classifier's linear parameter. shape (k,d)
    alpha : dual parameter. shape (n,k)
    """

    def __init__(self, reg, x, y):
        self.reg = reg
        self.n, self.d = x.shape
        self.k = np.amax(y) + 1
        self.w = np.zeros([self.k, self.d])
        self.alpha = np.zeros([self.n, self.k])

    ############################################################################
    # Change of variable functions
    ############################################################################

    def scores(self, x):
        """Return the score of each class for each line of x as a len(x)*k matrix."""
        return np.dot(x, self.w.T)

    def prediction(self, x):
        """Return the most likely class of each data point in x, as an array of size len(x)."""
        return np.argmax(self.scores(x), axis=-1)

    def primal2dual(self, x):
        """Return the probability vector p(.|x_i ; self.w) for all x_i in x.
        It is also equal to the dual variable given by the optimality condition on w.
        """
        return conditional_probabilities(self.scores(x))[0]

    def dual2primal(self, x, y):
        """Return the vector w given by the optimality condition for a given alpha."""
        w = np.mean((boolean_encoding(y, self.k) - self.alpha)[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)
        w /= self.reg
        return w

    ############################################################################
    # Objective functions
    ############################################################################

    def negloglikelihood(self, x, y):
        scores = self.scores(x)
        negll = logsumexp(scores)  # log partition function for each data point
        negll -= scores[np.arange(self.n), y]  # subtract the scores of the real classes
        return negll

    def entropy(self, i=None):
        epsilon = 1e-15
        if i is None:
            return -np.sum(self.alpha * np.log(np.maximum(epsilon, self.alpha)), axis=-1)
        else:
            return -np.sum(self.alpha[i] * np.log(np.maximum(epsilon, self.alpha[i])))

    def regularization(self):
        return self.reg / 2 * np.sum(self.w ** 2)

    def primal_objective(self, x, y):
        return self.regularization() + np.mean(self.negloglikelihood(x, y))

    def dual_objective(self, x, y):
        w = self.dual2primal(x, y)
        return -self.reg / 2 * np.sum(w ** 2) + np.mean(self.entropy())

    def duality_gap(self, x, y):
        return self.primal_objective(x, y) - self.dual_objective(x, y)

    def sdca(self, x, y, alpha0=None, npass=50, update_period=5, precision=1e-15, non_uniformity=0, _debug=False):
        """Update self.alpha and self.w with the stochastic dual coordinate ascent algorithm to fit the model to the
        data points x and the labels y.

        :param x: data points organized by rows
        :param y: labels as a one dimensional array. They should be positive.
        :param alpha0: initial point for alpha. If None, starts from the previous value of alpha (warm start).
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
            self.alpha = alpha0.copy()
            self.n, self.k = alpha0.shape
            self.w = self.dual2primal(x, y)

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
        # PRE-COMPUTE some coefficients for the line search
        ##################################################################################
        squared_norm_x = np.sum(x ** 2, axis=-1)

        ##################################################################################
        # NON-UNIFORM SAMPLING : initialize the sampler
        ##################################################################################
        if non_uniformity > 0:
            sampler = rc.RandomCounters(np.ones(self.n))
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
        t = -1
        while t < self.n * npass and duality_gap > precision:
            t += 1

            ##################################################################################
            # DRAW : one sample at random.
            ##################################################################################
            if np.random.rand() > non_uniformity:  # then sample uniformly
                i = np.random.randint(self.n)
            else:  # sample proportionally to the duality gaps
                i = sampler.sample()

            ##################################################################################
            # FUNCTION ESTIMATE : for the ascent
            ##################################################################################
            scores_i = self.scores(x[i])
            condprob_i, _ = conditional_probabilities(scores_i)
            ascent_direction = condprob_i - self.alpha[i]

            ##################################################################################
            # DUALITY GAP ESTIMATE : for the non-uniform sampling
            ##################################################################################
            if non_uniformity > 0:
                sampler.update(kullback_leibler(self.alpha[i], condprob_i), i)

            ##################################################################################
            # LINE SEARCH : find the optimal alpha[i]
            ##################################################################################
            squared_ascent_norm = np.sum(ascent_direction ** 2)
            linear_coeff = squared_norm_x[i] * squared_ascent_norm / self.reg / self.n
            constant_coeff = np.dot(ascent_direction, np.dot(self.w, x[i]))
            gammaopt, subobjective = optlinesearch(self.alpha[i], ascent_direction,
                                                   linear_coeff, constant_coeff, precision=1e-16)
            alphai = self.alpha[i] + gammaopt * ascent_direction

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
            self.w += (self.alpha[i] - alphai)[:, np.newaxis] * x[i] / self.reg / self.n
            self.alpha[i] = alphai

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
                    cond_probs, _ = conditional_probabilities(self.scores(x))  # n*k array, like alpha
                    dual_gaps = kullback_leibler(self.alpha, cond_probs)
                    dual_gap = np.mean(dual_gaps)
                    obj.append(dual_gap)
                    if non_uniformity > 0:
                        sampler = rc.RandomCounters(dual_gaps)
                else:
                    t1 = time.time()
                    obj.append(self.duality_gap(x, y))
                    t2 = time.time()
                    delta_time += t2 - t1  # Don't count the time spent monitoring the function
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
