# standard imports
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import oracles
import random_counters
# custom imports
import utils

########################################################################################################################
# CONSTANTS
########################################################################################################################
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_SIZE = len(ALPHABET)

# Field values to parse the csv
LETTER_ID = 0
LETTER_VALUE = 1
NEXT_ID = 2
WORD_ID = 3
POSITION = 4
FOLD = 5
FIRST_PIXEL = 6
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 8
NB_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
NB_FEATURES = ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE + 3)

MAX_LENGTH = 20


########################################################################################################################
# UTILITIES
########################################################################################################################
def letter2integer(letter):
    return int(ord(letter) - 97)


def integer2letter(i):
    if i < 0:
        return ''
    else:
        return ALPHABET[i]


def list2word(intlist):
    return ''.join([integer2letter(a) for a in intlist])


def word2list(word):
    return [letter2integer(letter) for letter in word]


def letters2wordimage(letters_images):
    word_image = np.zeros([IMAGE_HEIGHT, 2])
    spacing = np.zeros([IMAGE_HEIGHT, 2])
    for letter in letters_images:
        letter_image = letter.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        word_image = np.hstack((word_image, letter_image, spacing))
    return word_image


def display_word(letters_labels, letters_images):
    word_string = list2word(letters_labels)
    word_image = letters2wordimage(letters_images)
    plt.imshow(word_image, interpolation='nearest', cmap='Greys')
    plt.title(word_string)
    plt.axis('off')
    plt.show()


########################################################################################################################
# PARSING
########################################################################################################################
def read_lettersfile(tsv_file):
    import io
    letters = []
    with io.open(tsv_file, newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            row[1] = letter2integer(row[1])
            newrow = [int(value) for value in row[:-1]]
            letters.append(newrow)
        return np.array(letters, dtype=int)


def letters_to_labels_and_words(letters):
    """Return the labels and the letters images encoded in the 2d array letters. Encodes the labels in an array of 1d
    arrays and the images in an array of 2d arrays. For the 2d arrays, the position of the letter in the word is the
    first dimension, and the pixel position is the 2nd dimension.

    :param letters: the letters array that one gets after reading the tsv_file
    :return: labels, images
    """

    alllabels = []
    labels = []
    allimages = []
    images = []
    allfolds = []
    for letter in letters:
        labels.append(letter[LETTER_VALUE])
        images.append(letter[FIRST_PIXEL:])
        if letter[NEXT_ID] == -1:
            alllabels.append(np.array(labels))
            allimages.append(np.array(images))
            allfolds.append(letter[FOLD])
            labels = []
            images = []
    return np.array(alllabels), np.array(allimages), np.array(allfolds)


def extract_wordlengths(letters):
    nbwords = 0
    wordlengths = []
    length = 0
    for letter in letters:
        length += 1
        if letter[NEXT_ID] == -1:
            nbwords += 1
            wordlengths.append(length)
            length = 0
    print("Nombre de mots :", nbwords)
    return np.array(wordlengths)


def unique_words(words_labels):
    list_of_words = []
    for word in words_labels:
        list_of_words.append(list2word(word))
    return np.unique(list_of_words, return_counts=True)


########################################################################################################################
# RADIUS OF THE FEATURES
########################################################################################################################
def radius(word):
    # The factor 2 comes from the difference : ground truth - other label
    r = 2 * np.sum(word ** 2)  # emission
    r += 2 * word.shape[0]  # model bias
    r += 2 * 2  # beginning and end of word biases
    r += 2 * (word.shape[0] - 1)  # transitions
    return np.sqrt(r)


def radii(words):
    nb_words = words.shape[0]
    rs = np.empty(nb_words)
    for i in range(nb_words):
        rs[i] = radius(words[i])
    return rs


########################################################################################################################
# FEATURES
########################################################################################################################
class Features:
    """Features associated to a certain word. also used to store the weights of the primal model."""

    def __init__(self, emission=None, bias=None, transition=None, random=False):
        if random:
            self.emission = np.random.randn(ALPHABET_SIZE, NB_PIXELS)
            self.bias = np.random.randn(ALPHABET_SIZE, 3)
            self.transition = np.random.randn(ALPHABET_SIZE, ALPHABET_SIZE)
            return

        if emission is None:
            self.emission = np.zeros([ALPHABET_SIZE, NB_PIXELS])
        else:
            self.emission = emission

        if bias is None:
            self.bias = np.zeros([ALPHABET_SIZE, 3])
        else:
            self.bias = bias

        if transition is None:
            self.transition = np.zeros([ALPHABET_SIZE, ALPHABET_SIZE])
        else:
            self.transition = transition

    #########################################
    # Construction operations
    #########################################
    def _add_unary(self, images, label, position):
        self.emission[label] += images[position]
        self.bias[label] += [1, position == 0, position == images.shape[0] - 1]

    def _add_binary(self, label, next_label):
        self.transition[label, next_label] += 1

    def add_word(self, images, labels):
        for t in range(images.shape[0]):
            self._add_unary(images, labels[t], t)
        for t in range(images.shape[0] - 1):
            self._add_binary(labels[t], labels[t + 1])

    def add_dictionary(self, images_set, labels_set):
        for images, labels in zip(images_set, labels_set):
            word_size = labels.shape[0]
            if word_size != images.shape[0]:
                raise ValueError("Not the same number of labels (%i) and images (%i) inside word." \
                                 % (word_size, images.shape[0]))
            self.add_word(images, labels)

    def _add_unary_centroid(self, images, unary_marginals=None):
        if unary_marginals is None:  # assume uniform marginal
            self.emission += np.sum(images, axis=0) / ALPHABET_SIZE
            self.bias[:, 0] += images.shape[0] / ALPHABET_SIZE
            self.bias[:, 1:] += 1 / ALPHABET_SIZE
        else:
            self.emission += np.dot(unary_marginals.T, images)
            self.bias[:, 0] += np.sum(unary_marginals, axis=0)
            self.bias[:, 1] += unary_marginals[0]
            self.bias[:, 2] += unary_marginals[-1]

    def _add_binary_centroid(self, images, binary_marginals=None):
        if binary_marginals is None:  # assume uniform marginal
            self.transition += (images.shape[0] - 1) / ALPHABET_SIZE ** 2
        else:
            self.transition += np.sum(binary_marginals, axis=0)

    def add_centroid(self, images, marginals=None):
        if marginals is None:  # assume uniform marginal
            self._add_unary_centroid(images, None)
            self._add_binary_centroid(images, None)
        else:
            self._add_unary_centroid(images, marginals.unary)
            self._add_binary_centroid(images, marginals.binary)

    #########################################
    # From weights to probabilities
    #########################################
    def unary_scores(self, images):
        """Return the unary scores of word when self encode the weights of the model.

        :param images: T*d, each line is a letter image.
        :return: unary scores T*K, each case is a score for one image and one label.
        """
        uscores = np.dot(images, self.emission.T)
        uscores += self.bias[:, 0]  # model bias
        uscores[0] += self.bias[:, 1]  # beginning of word bias
        uscores[-1] += self.bias[:, 2]  # end of word bias
        return uscores

    def binary_scores(self, images):
        """Return the binary scores of a word when self encode the weights of the model.

        :param images: images T*d, each line is a letter image.
        :return: binary scores (T-1)*K*K, each case is the transition score between two labels for a given position.
        """
        return (images.shape[0] - 1) * [self.transition]

    def infer_probabilities(self, images, log):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        umargs, bmargs, log_partition = oracles.chain_sum_product(uscores, bscores, log=log)
        if log:
            return LogProbability(umargs, bmargs), log_partition
        else:
            return Probability(umargs, bmargs), log_partition

    def word_score(self, images, labels):
        """Return the score <self,F(images,labels)>."""
        return np.sum(images * self.emission[labels]) \
               + np.sum(self.bias[labels, np.zeros(labels.shape[0])]) \
               + np.sum(self.transition[labels[:-1], labels[1:]])

    def predict(self, images):
        uscores = self.unary_scores(images)
        bscores = self.binary_scores(images)
        return oracles.chain_viterbi(uscores, bscores)

    #########################################
    # Arithmetic operations
    #########################################
    def multiply_scalar(self, scalar, inplace=False):
        if inplace:
            self.emission *= scalar
            self.bias *= scalar
            self.transition *= scalar
        else:
            return Features(self.emission * scalar, self.bias * scalar, self.transition * scalar)

    def combine(self, other, operator):
        emission = operator(self.emission, other.emission)
        bias = operator(self.bias, other.bias)
        transition = operator(self.transition, other.transition)
        return Features(emission, bias, transition)

    def add(self, other):
        return self.combine(other, np.add)

    def subtract(self, other):
        return self.combine(other, np.subtract)

    def squared_norm(self):
        return np.sum(self.emission ** 2) + np.sum(self.bias ** 2) + np.sum(self.transition ** 2)

    def inner_product(self, other):
        return np.sum(self.emission * other.emission) + \
               np.sum(self.bias * other.bias) + \
               np.sum(self.transition * other.transition)

    def display(self):
        cmap = "Greys"
        emissions = letters2wordimage(self.emission)
        plt.matshow(emissions, cmap=cmap)
        ticks_positions = np.linspace(0, emissions.shape[1], ALPHABET_SIZE + 2).astype(int)[1:-1]
        plt.xticks(ticks_positions, list(ALPHABET))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.matshow(self.transition, cmap=cmap)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.yticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features")
        rescale_bias = np.array([1 / 7.5, 1, 1])
        plt.matshow((self.bias * rescale_bias).T, cmap=cmap)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features")


########################################################################################################################
# PROBABILITIES
########################################################################################################################
class Chain:
    """Represent anything that is decomposable over the nodes and adges of a chain."""

    def __init__(self, unary, binary):
        if unary.shape[0] != binary.shape[0] + 1:
            raise ValueError("Wrong size of marginals: %i vs %i" % (unary.shape[0], binary.shape[0]))
        self.unary = unary
        self.binary = binary

    def __str__(self):
        return "unary: \n" + np.array_str(self.unary) + "\n binary: \n" + np.array_str(self.binary)

    def __repr__(self):
        return "unary: \n" + np.array_repr(self.unary) + "\n binary: \n" + np.array_repr(self.binary)

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
            self.binary = np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) / ALPHABET_SIZE ** 2
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
            # print("Marginalisation of the left of the binary marginals is inconsistent with unary marginals.")
        if not np.isclose(from_right_binary, self.unary[:-1]).all():
            ans = False
            # print("Marginalisation of the right of the binary marginals is inconsistent with unary marginals.")
        if not np.isclose(from_right_binary[1:], from_left_binary[:-1]).all():
            ans = False
            # print("Marginalisation of the left and right of the binary marginals are inconsistent.")
        return ans

    def subtract(self, other):
        return Probability(unary=self.unary - other.unary,
                           binary=self.binary - other.binary)

    def multiply(self, other):
        return Probability(unary=self.unary * other.unary,
                           binary=self.binary * other.binary)

    def sum(self):
        """Return the special inner product where the marginals on the separations are subtracted."""
        return np.sum(self.binary) - np.sum(self.unary[1:-1])

    def inner_product(self, other):
        """Return the special inner product where the marginals on the separations are subtracted."""
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
            self.binary = np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) * (-2 * np.log(ALPHABET_SIZE))
        else:  # take what is given
            Chain.__init__(self, unary, binary)

    def entropy(self, precision=1e-10):
        ans = utils.log_entropy(self.binary) - utils.log_entropy(self.unary[1:-1])
        assert ans >= -precision, (self, ans)
        return max(ans, 0)

    def kullback_leibler(self, other, precision=1e-10):
        ans = utils.log_kullback_leibler(self.binary, other.binary) \
              - utils.log_kullback_leibler(self.unary[1:-1], other.unary[1:-1])
        assert ans >= -precision, (self, ans)
        return max(ans, 0)

    def convex_combination(self, other, gamma):
        if gamma == 0:
            return self
        elif gamma == 1:
            return other
        else:
            unary = utils.logsumexp(np.array([self.unary + np.log(1 - gamma), other.unary + np.log(gamma)]), axis=0)
            binary = utils.logsumexp(np.array([self.binary + np.log(1 - gamma), other.binary + np.log(gamma)]), axis=0)
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
def marginals_to_centroid(images, labels, marginals=None, log=False):
    nb_words = labels.shape[0]

    ground_truth_sum = Features()
    ground_truth_sum.add_dictionary(images, labels)

    marginals_sum = Features()
    if marginals is None:  # assume uniform
        for image in images:
            marginals_sum.add_centroid(image)

    elif log:
        for image, margs in zip(images, marginals):
            marginals_sum.add_centroid(image, margs.to_probability())

    else:
        for image, margs in zip(images, marginals):
            marginals_sum.add_centroid(image, margs)

    corrected_features_centroid = ground_truth_sum.subtract(marginals_sum)
    corrected_features_centroid.multiply_scalar(1 / nb_words, inplace=True)
    return corrected_features_centroid


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


# def fenchel_gaps(marginals, images):
#     ans = []
#     all_weights = np.empty(marginals.shape[0],dtype=object)
#     for margs, imgs in zip(marginals, images):
#         al
#         newmargs, logprobabity = weights.infer_probabilities(imgs, log=True)
#         ans.append(margs.kullback_leibler(newmargs))
#     return np.array(ans)


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
    nb_words = y.shape[0]
    if nb_words != x.shape[0]:
        raise ValueError("Not the same number of labels (%i) and images (%i) inside training set." \
                         % (nb_words, x.shape[0]))

    if isinstance(init, np.ndarray):  # assume that init contains the marginals for a warm start.
        marginals = init
        weights = marginals_to_centroid(x, y, marginals, log=True)
    elif init == "uniform":
        marginals = uniform_marginals(y, log=True)
        weights = marginals_to_centroid(x, y)
    elif init == "random":
        weights = Features(random=True)
        marginals = np.array([weights.infer_probabilities(imgs, log=True)[0] for imgs in x])
        weights = marginals_to_centroid(x, y, marginals, log=True)
    else:
        raise ValueError("Not a valid argument for init: %r" % init)

    weights.multiply_scalar(1 / regu, inplace=True)

    ##################################################################################
    # OBJECTIVES : dual objective and duality gaps
    ##################################################################################
    entropies = np.array([margs.entropy() for margs in marginals])
    dual_objective = entropies.mean() - regu / 2 * weights.squared_norm()

    new_marginals = [weights.infer_probabilities(imgs, log=True)[0] for imgs in x]
    dgaps = np.array([margs.kullback_leibler(newmargs) for margs, newmargs in zip(marginals, new_marginals)])
    duality_gap = dgaps.sum() / nb_words

    objective = [duality_gap]
    delta_time = time.time()
    timing = [0]

    ##################################################################################
    # NON-UNIFORM SAMPLING : initialize the sampler
    ##################################################################################
    sampler = random_counters.RandomCounters(dgaps)

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
        assert nbeta_i.are_densities(1), (display_word(y[i], x[i]),
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

        if divergence_gap < precision:
            continue

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

        sampler.update(divergence_gap, i)

        duality_gap_estimate = sampler.get_total() / nb_words

        ##################################################################################
        # LINE SEARCH : find the optimal step size gammaopt or use a fixed one
        ##################################################################################
        quadratic_coeff = - regu * nb_words / 2 * primaldir_squared_norm
        linear_coeff = - regu * nb_words * weights_dot_primaldir

        # slope of the line search function in 0
        slope = - dual_direction.inner_product(alpha_i) + linear_coeff
        assert np.isclose(slope, divergence_gap + reverse_gap), print(
            "iteration : %i \n"
            "<d, -log alpha_i> %.2e | linear coeff %.2e | slope : %.2e \n"
            "individual gap = %.2e | reverse gap = %.2e | sum = %.2e" % (
                t, - dual_direction.inner_product(alpha_i), linear_coeff, slope,
                divergence_gap, reverse_gap, divergence_gap + reverse_gap)
        )

        if step_size:
            gammaopt = step_size
            subobjective = []
        else:
            # print("dual direction")
            # print(dual_direction.square().to_logprobability())
            # print("new marginals")
            # print(beta_i)

            def evaluator(gamma, returnf=False):
                # line search function and its derivatives
                newmargs = alpha_i.convex_combination(beta_i, gamma)
                # assert newmargs.are_densities(1)
                # assert newmargs.are_consistent()
                # assert newmargs.are_positive(), newmargs.display
                gf = - dual_direction.inner_product(newmargs) + gamma * 2 * quadratic_coeff + linear_coeff
                if gf == 0:
                    return gf, 0
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

        # Plot the distribution of gradient values over the data
        if t == 100 * nb_words:
            slopes = get_slopes(marginals, weights, x, regu)
            plt.hist(slopes, 50)
            break

        ##################################################################################
        # UPDATE : the primal and dual coordinates
        ##################################################################################
        marginals[i] = alpha_i.convex_combination(beta_i, gammaopt)
        weights = weights.add(primal_direction.multiply_scalar(gammaopt, inplace=False))

        ##################################################################################
        # ANNEX
        ##################################################################################
        # Update the dual objective and the entropy
        tmp = marginals[i].entropy()
        dual_objective += \
            (tmp - entropies[i] + gammaopt ** 2 * quadratic_coeff + gammaopt * linear_coeff) / nb_words
        entropies[i] = tmp
        if _debug:
            # Append relevant variables
            annex.append([np.log10(primaldir_squared_norm),
                          weights_dot_primaldir / np.sqrt(primaldir_squared_norm),
                          gammaopt,
                          dual_objective,
                          np.log10(duality_gap_estimate),
                          divergence_gap,
                          i,
                          len(subobjective)])

        if t % (update_period * nb_words) == 0:
            ##################################################################################
            # OBJECTIVES : after each pass over the data, compute the duality gap
            ##################################################################################
            # t1 = time.time()
            dgaps = divergence_gaps(marginals, weights, x)
            sampler = random_counters.RandomCounters(dgaps)
            duality_gap = sampler.get_total() / nb_words
            objective.append(duality_gap)
            # if t % (update_period * nb_words) == 0 and non_uniformity > 0:
            #     pass
            #     ###################################################################################
            #     # DUALITY GAPS: perform a batch update after every update_period epochs
            #     # To reduce the staleness for the non-uniform sampling
            #     # To monitor the objective and provide a stopping criterion
            #     ##################################################################################
            # delta_time += time.time() - t1
            t2 = time.time()
            timing.append(t2 - delta_time)

    ##################################################################################
    # ANNEX
    ##################################################################################
    if _debug and not step_size:
        print("Perfect line search : %i \t Negative ls : %i \t Positive ls : %i" % (countzero, countneg, countpos))

    ##################################################################################
    # FINISH : convert the objectives to simplify the after process.
    ##################################################################################
    objective = np.array(objective)
    timing = np.array(timing)
    if _debug:
        annex = np.array(annex)
        return marginals, weights, objective, timing, annex
    else:
        return marginals, weights, objective, timing
