# standard imports
import matplotlib.pyplot as plt
import numpy as np

# custom imports
import oracles
import utils


def uniform(length, alphabet_size, log=True):
    unary = np.ones([length, alphabet_size])
    binary = np.ones([length - 1, alphabet_size, alphabet_size])
    if log:
        unary *= - np.log(alphabet_size)
        binary *= -2 * np.log(alphabet_size)
    else:
        unary /= alphabet_size
        binary /= alphabet_size ** 2
    return Sequence(unary=unary, binary=binary, log=log)


def dirac(labels, alphabet_size, log=True):
    """Return an array of dirac over the observed labels.

    :param labels:
    :param alphabet_size:
    :param log: if True, return smoothed log-probabilities
    """
    length = labels.shape[0]
    constant = 10 if log else 1
    unary = np.zeros([length, alphabet_size])
    unary[np.arange(length), labels] = constant
    binary = np.zeros([length - 1, alphabet_size, alphabet_size])
    binary[np.arange(length - 1), labels[:-1], labels[1:]] = constant
    if log:
        unary, binary, _ = oracles.chain_sum_product(unary, binary)
    return Sequence(unary=binary, binary=binary, log=log)


class Sequence:
    """Represent anything that is decomposable over the nodes and edges of a sequential model.

    It can be a score, a conditional probability p(y|x) under the form of MARGINALS or
    LOG-MARGINALS (in which case self.log=True), the ascent direction, the derivative of the KL
    or the entropy."""

    def __init__(self, unary, binary, log):
        if unary.shape[0] != binary.shape[0] + 1:
            raise ValueError("Wrong length of marginals: %i vs %i"
                             % (unary.shape[0], binary.shape[0]))
        self.length = unary.shape[0]

        if unary.shape[1] != binary.shape[1] \
                or unary.shape[1] != binary.shape[2]:
            raise ValueError("Wring alphabet size: %i vs (%i, %i)"
                             % (unary.shape[1], binary.shape[1], binary.shape[2]))
        self.alphabet_size = unary.shape[1]

        self.unary = unary
        self.binary = binary
        self.log = log

    def __str__(self):
        return "unary: \n" + np.array_str(self.unary) \
               + "\n binary: \n" + np.array_str(self.binary)

    def __repr__(self):
        return "unary: \n" + np.array_repr(self.unary) \
               + "\n binary: \n" + np.array_repr(self.binary)

    def display(self, alphabet):
        alength = len(alphabet)
        plt.matshow(self.unary)
        plt.xticks(range(alength), [alphabet[x] for x in range(alength)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("unary marginals")
        plt.matshow(self.binary.sum(axis=0))
        plt.xticks(range(alength), [alphabet[x] for x in range(alength)])
        plt.yticks(range(alength), [alphabet[x] for x in range(alength)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("sum of binary marginals")

    #########################################
    # Special operations
    #########################################
    def log(self):
        return Sequence(np.log(self.unary), np.log(self.binary), log=True)

    def exp(self):
        return Sequence(np.exp(self.unary), np.exp(self.binary), log=False)

    def reduce(self):
        """Return the special summation where the marginals on the separations are
        subtracted."""
        return np.sum(self.binary) - np.sum(self.unary[1:-1])

    def inner_product(self, other):
        """Return the special inner product where the marginals on the separations are
        subtracted."""
        return self.multiply(other).reduce()

    def log_reduce_exp(self, to_add):
        themax = max(np.amax(self.unary[1:-1]), np.amax(self.binary))
        return themax + np.log(np.sum(np.exp(self.binary - themax))
                               - np.sum(np.exp(self.unary[1:-1] - themax))
                               + to_add * np.exp(-themax))

    def convex_combination(self, other, s):
        """Return (1-s)*self + s*other"""
        if s == 0:
            return self
        if s == 1:
            return other

        if self.log:
            unary = np.minimum(0, utils.logsumexp(
                np.asarray([self.unary + np.log(1 - s), other.unary + np.log(s)]), axis=0))
            binary = np.minimum(0, utils.logsumexp(
                np.asarray([self.binary + np.log(1 - s), other.binary + np.log(s)]), axis=0))
        else:
            unary = (1 - s) * self.unary + s * other.unary
            binary = (1 - s) * self.binary + s * other.binary
        return Sequence(unary=unary, binary=binary, log=self.log)

    def subtract_exp(self, other):
        """Return the ascent direction without numerical issues

        Should start from the log space.
        """
        max_unary = np.maximum(self.unary, other.unary)
        unary = np.exp(max_unary) * (np.exp(self.unary - max_unary)
                                     - np.exp(other.unary - max_unary))

        max_binary = np.maximum(self.binary, other.binary)
        binary = np.exp(max_binary) * (np.exp(self.binary - max_binary)
                                       - np.exp(other.binary - max_binary))

        return Sequence(unary=unary, binary=binary, log=False)

    #########################################
    # Typical arithmetic operations
    #########################################
    def combine(self, other, ufunc):
        unary = ufunc(self.unary, other.unary)
        binary = ufunc(self.binary, other.binary)
        return Sequence(unary, binary, self.log)

    def add(self, other):
        return self.combine(other, np.add)

    def subtract(self, other):
        return self.combine(other, np.subtract)

    def multiply(self, other):
        return self.combine(other, np.multiply)

    def map(self, ufunc):
        return Sequence(ufunc(self.unary), ufunc(self.binary), self.log)

    def absolute(self):
        return self.map(np.absolute)

    def multiply_scalar(self, scalar):
        return Sequence(scalar * self.unary, scalar * self.binary, self.log)

    #########################################
    # Assertion operations
    #########################################
    def is_density(self, integral=1):
        return np.isclose(np.sum(self.unary, axis=1), integral).all() \
               and np.isclose(np.sum(self.binary, axis=(1, 2)), integral).all()

    def is_consistent(self):
        ans = True
        from_left_binary = np.sum(self.binary, axis=1)
        from_right_binary = np.sum(self.binary, axis=2)
        if not np.isclose(from_left_binary, self.unary[1:]).all():
            ans = False
            # print("Left inconsistent with unary.")
        if not np.isclose(from_right_binary, self.unary[:-1]).all():
            ans = False
            # print("Right inconsistent with unary.")
        if not np.isclose(from_right_binary[1:], from_left_binary[:-1]).all():
            ans = False
            # print("Left inconsistent with right.")
        return ans

    #########################################
    # Information theory
    #########################################
    def entropy(self, returnlog=False):
        cliques = utils.entropy(self.binary, returnlog=True)
        separations = utils.entropy(self.unary[1:-1], returnlog=True)
        return Sequence._safe_reduce(cliques, separations, returnlog)

    def kullback_leibler(self, other, returnlog=False):
        cliques = utils.kullback_leibler(self.binary, other.binary, returnlog=True)
        separations = utils.kullback_leibler(self.unary[1:-1], other.unary[1:-1],
                                             returnlog=True)
        return Sequence._safe_reduce(cliques, separations, returnlog)

    @staticmethod
    def _safe_reduce(cliques, separations, returnlog):
        if cliques <= separations:
            return -np.inf if returnlog else 0

        try:
            ans = cliques + np.log(1 - np.exp(separations - cliques))
            if returnlog:
                return ans
            else:
                return np.exp(ans)

        except FloatingPointError:
            print("Entropy problem:", cliques, separations)
            raise
