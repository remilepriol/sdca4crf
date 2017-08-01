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

    def dual2primal(self, x, y):
        """Update w to fit alpha."""
        boolean_encoding = np.zeros([self.n, self.k])
        boolean_encoding[np.arange(self.n), y] = 1  # n*k encoding of y
        self.w = np.mean((boolean_encoding - self.alpha)[:, :, np.newaxis] * x[:, np.newaxis, :], axis=0)
        self.w /= self.reg

    def negloglikelihood(self, x, y):
        scores = np.dot(x, self.w.T)  # n*k array : score[i,j] of class j for object i
        negll = logsumexp(scores)
        negll -= scores[np.arange(self.n), y]  # subtract the scores of the real classes
        return np.mean(negll)

    def entropy(self, i=None):
        epsilon = 1e-15
        if i is None:
            return -np.sum(self.alpha * np.log(np.maximum(epsilon, self.alpha)), axis=-1)
        else:
            return -np.sum(self.alpha[i] * np.log(np.maximum(epsilon, self.alpha[i])))

    def mean_entropy(self):
        return np.mean(self.entropy())

    def regularization(self):
        return self.reg / 2 * np.sum(self.w ** 2)

    def primal_objective(self, x, y):
        return self.regularization() + self.negloglikelihood(x, y)

    def dual_objective(self, x, y):
        self.dual2primal(x, y)
        return self.regularization() + self.mean_entropy()

    def duality_gap(self, x, y):
        return 2 * self.regularization() - self.mean_entropy() + self.negloglikelihood(x, y)

    def conditional_probabilities(self, x):
        # handles x even if it is a single instance (a vector)
        scores = np.dot(x, self.w.T)  # n*k array : score[i,j] of class j for object i
        smax = np.amax(scores, axis=-1, keepdims=True)
        scores -= smax  # stabilize the exponential by subtracting the maximum score for each example x
        prob = np.exp(scores)
        partition_functions = np.sum(prob, axis=-1, keepdims=True)
        prob /= partition_functions
        return prob

    def prediction(self, x):
        # handles x even if it is a single instance (a vector)
        scores = np.dot(x, self.w.T)  # n*k array : score[i,j] of class j for object i
        ymax = np.argmax(scores, axis=-1)
        return ymax

    def sdca(self, alpha0, x, y, npass=50, precision=1e-15, debug=False, nonuniform=False):
        self.n, self.k = alpha0.shape
        self.n, self.d = x.shape

        self.alpha = alpha0.copy()
        self.dual2primal(x, y)
        # else start from the previous optimum.

        # keep track of the objective
        if debug:
            obj = [[self.regularization(), self.negloglikelihood(x, y), self.mean_entropy(), 0]]
        else:
            obj = [self.duality_gap(x, y)]
        timing = [time.time()]

        # pre-calculation of some coefficients for the line search
        linear_coeffs = np.sum(x ** 2, axis=-1) / self.reg / self.n

        dual_gaps = np.ones(self.n) / self.n

        countneg = 0
        countpos = 0
        countzero = 0

        t = 0
        while t < self.n * npass and (debug or obj[-1] > precision):
            t += 1

            if nonuniform:
                if np.random.rand() < 0.5:  # with probability 1/2 sample uniformly
                    i = np.random.randint(self.n)
                else:  # and with the other 1/2 sample according to the gaps
                    try:
                        i = np.random.choice(self.n, p=dual_gaps)
                    except ValueError:
                        print(dual_gaps)
                        raise
            else:
                i = np.random.randint(0, self.n)

            # find the optimal alpha[i]
            ascent_direction = self.conditional_probabilities(x[i]) - self.alpha[i]
            ascent_norm = np.sum(ascent_direction ** 2)
            linear_coeff = linear_coeffs[i] * ascent_norm
            constant_coeff = np.dot(ascent_direction, np.dot(self.w, x[i]))
            gammaopt, subobjective = optlinesearch(self.alpha[i], ascent_direction,
                                                   linear_coeff, constant_coeff, precision=1e-16)
            alphai = self.alpha[i] + gammaopt * ascent_direction

            if subobjective[-1] > precision:
                countpos += 1
            elif subobjective[-1] < -precision:
                countneg += 1
            else:
                countzero += 1

            # update w accordingly
            self.w += (self.alpha[i] - alphai)[:, np.newaxis] * x[i] / self.reg / self.n
            self.alpha[i] = alphai

            # update the duality gap for variable i (extra computation only because of my code)
            # problem : since it is right after the update of alpha i and w, the score is almost always 0.
            # this is not what we want.
            # Hence the full batch approach : update all the gaps once in a while
            if t % self.n == 0:
                s = np.dot(x, self.w.T)  # n*k array, like alpha
                dual_gaps = np.maximum(0, logsumexp(s) - self.entropy() - np.sum(s * self.alpha, axis=-1))
                dual_gaps /= dual_gaps.sum()
            # I do almost the same computations as in the bloc above.
            if t % self.n == 0:
                if debug:
                    obj.append(
                        [self.regularization(), self.negloglikelihood(x, y), self.mean_entropy(), len(subobjective)])
                else:
                    obj.append(self.duality_gap(x, y))
                timing.append(time.time())

        print("Perfect line search : %i \t Negative ls : %i \t Positive ls : %i" % (countzero, countneg, countpos))

        obj = np.array(obj)
        # if debug:
        #     maxlen= max([len(subobj) for subobj in obj])
        #     for subobj in obj:
        #         subobj += [0]*(maxlen-len(subobj))
        #     obj = np.array(obj)
        timing = np.array(timing)
        timing -= timing[0]
        return obj, timing

# import generator as gen
#
# n = 1000
# d = 2
# k = 6
# bias = 1
# x, y, mu, sig = gen.gengaussianmixture(n, d, k, scale=.05)
# x = gen.standardize(x)
# x = np.concatenate([x, bias * np.ones([n, 1])], axis=1)
# reg = 1
# model = MulticlassLogisticRegression(reg, x, y)
#
# alpha0 = np.random.rand(n, k)
# alpha0 /= np.sum(alpha0, axis=1, keepdims=True)
# obj_sdca, time_sdca = model.sdca(alpha0, x, y, npass=14, precision=1e-5, debug=True)
