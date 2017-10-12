# standard imports
import matplotlib.pyplot as plt
import numpy as np

# custom imports
import utils
from constant import ALPHABET, ALPHABET_SIZE


class Chain:
    """Represent anything that is decomposable over the nodes and edges of a chain."""

    def __init__(self, unary, binary):
        if unary.shape[0] != binary.shape[0] + 1:
            raise ValueError("Wrong size of marginals: %i vs %i"
                             % (unary.shape[0], binary.shape[0]))
        self.unary = unary
        self.binary = binary

    def __str__(self):
        return "unary: \n" + np.array_str(self.unary) \
               + "\n binary: \n" + np.array_str(self.binary)

    def __repr__(self):
        return "unary: \n" + np.array_repr(self.unary) \
               + "\n binary: \n" + np.array_repr(self.binary)

    def display(self):
        plt.matshow(self.unary)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("unary marginals")
        plt.matshow(self.binary.sum(axis=0))
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.yticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("sum of binary marginals")


class Probability(Chain):
    """Represent a conditional probability p(y|x) under the form of MARGINALS.
    Can also represent the ascent direction or the score.
    Ony has a finite precision on the small numbers.
    Inappropriate to handle the derivatives of the entropy or the KL.
    """

    def __init__(self, unary=None, binary=None, word_length=None):
        if unary is None or binary is None:
            self.unary = np.ones([word_length, ALPHABET_SIZE]) / ALPHABET_SIZE
            self.binary = \
                np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) \
                / ALPHABET_SIZE ** 2
        else:
            Chain.__init__(self, unary, binary)

    def are_densities(self, integral=1):
        return np.isclose(np.sum(self.unary, axis=1), integral).all() \
               and np.isclose(np.sum(self.binary, axis=(1, 2)), integral).all()

    def are_consistent(self):
        ans = True
        from_left_binary = np.sum(self.binary, axis=1)
        from_right_binary = np.sum(self.binary, axis=2)
        if not np.isclose(from_left_binary, self.unary[1:]).all():
            ans = False
            # print("Marginalisation of the left of the binary marginals is inconsistent with
            # unary marginals.")
        if not np.isclose(from_right_binary, self.unary[:-1]).all():
            ans = False
            # print("Marginalisation of the right of the binary marginals is inconsistent with
            # unary marginals.")
        if not np.isclose(from_right_binary[1:], from_left_binary[:-1]).all():
            ans = False
            # print("Marginalisation of the left and right of the binary marginals are
            # inconsistent.")
        return ans

    def subtract(self, other):
        return Probability(unary=self.unary - other.unary,
                           binary=self.binary - other.binary)

    def multiply(self, other):
        return Probability(unary=self.unary * other.unary,
                           binary=self.binary * other.binary)

    def sum(self):
        """Return the special summation where the marginals on the separations are
        subtracted."""
        return np.sum(self.binary) - np.sum(self.unary[1:-1])

    def inner_product(self, other):
        """Return the special inner product where the marginals on the separations are
        subtracted."""
        return self.multiply(other).sum()

    def map(self, func):
        return Probability(unary=func(self.unary), binary=func(self.binary))

    # def special_function(self, newmargs):
    #   """To handle the second derivative in th eline search"""
    #     unary_filter = (self.unary != 0)
    #     binary_filter = (self.binary != 0)
    #
    # def add(self, other):
    #     return Probability(unary=self.unary + other.unary,
    #                        binary=self.binary + other.binary)
    #
    # def divide(self, other):
    #     return Probability(unary=self.unary / other.unary,
    #                        binary=self.binary / other.binary)
    #
    # def multiply_scalar(self, scalar):
    #     return Probability(unary=scalar * self.unary,
    #                        binary=scalar * self.binary)

    def to_logprobability(self):
        return LogProbability(unary=np.log(self.unary),
                              binary=np.log(self.binary))

    @staticmethod
    def dirac(labels):
        word_length = labels.shape[0]
        umargs = np.zeros([word_length, ALPHABET_SIZE])
        umargs[np.arange(word_length), labels] = 1
        bmargs = np.zeros([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE])
        bmargs[np.arange(word_length - 1), labels[:-1], labels[1:]] = 1
        return Probability(unary=umargs, binary=bmargs)


class LogProbability(Chain):
    """Represent a conditional probability p(y|x) under the form of LOG-marginals."""

    def __init__(self, unary=None, binary=None, word_length=None):
        if unary is None or binary is None:  # assume uniform
            self.unary = np.ones([word_length, ALPHABET_SIZE]) * (-np.log(ALPHABET_SIZE))
            self.binary = np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) * (
                -2 * np.log(ALPHABET_SIZE))
        else:  # take what is given
            Chain.__init__(self, unary, binary)

    def entropy(self, returnlog=False):
        cliques = utils.entropy(self.binary, returnlog=True)
        separations = utils.entropy(self.unary[1:-1], returnlog=True)

        assert cliques >= separations, (cliques, separations)

        if cliques == separations:
            return 0

        else:
            try:
                ans = cliques + np.log(1 - np.exp(separations - cliques))

                if returnlog:
                    return ans
                else:
                    return np.exp(ans)

            except FloatingPointError:
                print(
                    "KL problem", cliques, separations,
                    "\n binary \n", self.binary,
                    "\n unary \n", self.unary
                )
                raise

    def kullback_leibler(self, other, returnlog=False):
        cliques = utils.kullback_leibler(self.binary, other.binary, returnlog=True)
        separations = utils.kullback_leibler(self.unary[1:-1], other.unary[1:-1], returnlog=True)
        assert cliques >= separations, (cliques, separations)

        if cliques == separations:
            return 0

        else:
            try:
                ans = cliques + np.log(1 - np.exp(separations - cliques))

                if returnlog:
                    return ans
                else:
                    return np.exp(ans)

            except FloatingPointError:
                print("KL problem", cliques, separations, "\n",
                      self.binary, other.binary,
                      self.unary, other.unary)
                raise

    def convex_combination(self, other, gamma):
        if gamma == 0:
            return self
        elif gamma == 1:
            return other
        else:
            unary = utils.logsumexp(
                np.array([self.unary + np.log(1 - gamma), other.unary + np.log(gamma)]), axis=0)
            binary = utils.logsumexp(
                np.array([self.binary + np.log(1 - gamma), other.binary + np.log(gamma)]), axis=0)
            unary = np.minimum(unary, 0)
            binary = np.minimum(binary, 0)
            return LogProbability(unary=unary, binary=binary)

    def divide(self, other):
        return LogProbability(unary=self.unary - other.unary,
                              binary=self.binary - other.binary)

    def inverse(self):
        return LogProbability(unary=-self.unary,
                              binary=-self.binary)

    def smart_subtract(self, other):
        """Gives the ascent direction without numerical issues"""

        max_unary = np.maximum(self.unary, other.unary)
        unary = np.exp(max_unary) * (np.exp(self.unary - max_unary)
                                     - np.exp(other.unary - max_unary))

        max_binary = np.maximum(self.binary, other.binary)
        binary = np.exp(max_binary) * (np.exp(self.binary - max_binary)
                                       - np.exp(other.binary - max_binary))

        return Probability(unary=unary, binary=binary)

    def logsumexp(self, to_add):
        themax = max(np.amax(self.unary[1:-1]), np.amax(self.binary))
        return themax + np.log(np.sum(np.exp(self.binary - themax))
                               - np.sum(np.exp(self.unary[1:-1] - themax))
                               + to_add * np.exp(-themax))

    def multiply_scalar(self, scalar):
        return LogProbability(unary=self.unary * scalar, binary=self.binary * scalar)

    def to_probability(self):
        return Probability(unary=np.exp(self.unary),
                           binary=np.exp(self.binary))
