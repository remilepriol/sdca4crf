import time

import numpy as np


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


def boolean_encoding(y):
    """Return the n*k matrix Y whose line i is the one-hot encoding of y_i."""
    n = y.shape[0]
    k = np.unique(y).shape[0]
    ans = np.zeros([n, k])
    ans[np.arange(n), y] = 1  #
    return ans


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
        self.k = np.unique(y).shape[0]
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

    def conditional_probabilities(self, x):
        """Return the probability vector p(.|x_i ; self.w) for all x_i in x.
        It is also equal to the dual representation of w.
        """
        scores = self.scores(x)
        smax = np.amax(scores, axis=-1, keepdims=True)  # mode of the score for each x
        prob = np.exp(scores - smax)  # stabilized exponential
        partition_functions = np.sum(prob, axis=-1, keepdims=True)
        prob /= partition_functions
        return prob

    def dual2primal(self, x, y):
        """Return the vector w given by the optimality condition for a given alpha."""
        w = np.mean((boolean_encoding(y) - self.alpha)[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)
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
        return self.reg / 2 * np.sum(w ** 2) + np.mean(self.entropy())

    def duality_gap(self, x, y):
        return self.primal_objective(x, y) - self.dual_objective(x, y)

    def sdca(self, x, y, alpha0=None, npass=50, precision=1e-15, debug=False, nonuniform=False):

        ##################################################################################
        # INITIALIZE : the dual and primal variables
        ##################################################################################
        self.n, self.k = alpha0.shape
        self.n, self.d = x.shape
        if alpha0 is None:  # Warm start
            self.alpha = alpha0.copy()
            self.w = self.dual2primal(x, y)

        ##################################################################################
        # OBJECTIVES : initialize the lists to be returned
        ##################################################################################
        if debug:
            obj = [[self.regularization(), np.mean(self.negloglikelihood(x, y)), np.mean(self.entropy()), 0]]
        else:
            obj = [self.duality_gap(x, y)]
        timing = [time.time()]

        # pre-compute some coefficients for the line search
        squared_norm_x = np.sum(x ** 2, axis=-1)

        # initialize of the probability table for the non-uniform sampling
        dual_gaps = np.ones(self.n) / self.n

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
        while t < self.n * npass and (debug or obj[-1] > precision):
            t += 1

            ##################################################################################
            # DRAW : one sample at random.
            ##################################################################################
            # TODO replace nonuniform by nonuniformity
            if nonuniform:
                if np.random.rand() > 0.5:  # with probability 1/2 sample uniformly
                    i = np.random.randint(self.n)
                else:  # and with the other 1/2 sample according to the gaps
                    try:
                        i = np.random.choice(self.n, p=dual_gaps)
                    except ValueError:
                        print(dual_gaps)
                        raise
            else:
                i = np.random.randint(0, self.n)

            ##################################################################################
            # FUNCTION ESTIMATE
            ##################################################################################
            ascent_direction = self.conditional_probabilities(x[i]) - self.alpha[i]
            squared_ascent_norm = np.sum(ascent_direction ** 2)
            linear_coeff = squared_norm_x[i] * squared_ascent_norm / self.reg / self.n
            constant_coeff = np.dot(ascent_direction, np.dot(self.w, x[i]))

            # TODO add a computation of the duality gap here, and update the table.

            ##################################################################################
            # LINE SEARCH : find the optimal alpha[i]
            ##################################################################################
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

            # TODO change this computation to combine it with the stopping criterion calculation
            ##################################################################################
            # UPDATE : the table of duality gaps, after every 10 pass over the data.
            ##################################################################################
            if t % (10 * self.n) == 0:
                scores = self.scores(x)  # n*k array, like alpha
                dual_gaps = np.maximum(0, logsumexp(scores) - self.entropy() - np.sum(scores * self.alpha, axis=-1))
                dual_gaps /= dual_gaps.sum()

            ##################################################################################
            # OBJECTIVES : after each pass over the data, compute the duality gap
            ##################################################################################
            if t % self.n == 0:
                if debug:
                    obj.append(
                        [self.regularization(), np.mean(self.negloglikelihood(x, y)), np.mean(self.entropy()),
                         len(subobjective)])
                else:
                    obj.append(self.duality_gap(x, y))
                timing.append(time.time())

        ##################################################################################
        # COUNTERS
        ##################################################################################
        print("Perfect line search : %i \t Negative ls : %i \t Positive ls : %i" % (countzero, countneg, countpos))

        ##################################################################################
        # FINISH : convert the objectives to simplify the after process.
        ##################################################################################
        obj = np.array(obj)
        timing = np.array(timing)
        timing -= timing[0]
        return obj, timing
