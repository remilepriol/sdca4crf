# standard imports
import csv
import time

import matplotlib.pyplot as plt
import numpy as np

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
# FEATURE SELECTION
########################################################################################################################
def select_emission(label):
    """If label is positive, return the slice of emission features corresponding to label.
    Else return the slice of emission features for all labels.

    :param label:
    :return:
    """
    if label >= 0:
        start = label * NB_PIXELS
        return slice(start, start + NB_PIXELS)
    else:
        return slice(0, ALPHABET_SIZE * NB_PIXELS)


def select_transition(label, next_label):
    """If label and next label are positive, return the coordinate of the corresponding transition feature.
    Else return the slice of all transition features.

    :param label:
    :param next_label:
    :return:
    """
    if label >= 0 and next_label >= 0:
        start = ALPHABET_SIZE * (NB_PIXELS + label) + next_label
        return slice(start, start + 1)
    else:
        start = ALPHABET_SIZE * NB_PIXELS
        return slice(start, start + ALPHABET_SIZE * ALPHABET_SIZE)


def select_bias(label):
    """If label is positive, return the slice of bias features corresponding to label.
    Else return the slice of bias features for all labels.

    :param label:
    :return:
    """
    if label >= 0:
        start = ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE) + label
        return slice(start, start + 3)
    else:  # return the whole bias range
        start = ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE)
        return slice(start, start + 3 * ALPHABET_SIZE)


########################################################################################################################
# FEATURE CREATION
########################################################################################################################
def unary_feature(images, label, position):
    feat = np.zeros(NB_FEATURES)
    # image value
    feat[select_emission(label)] = images[position]
    # bias
    feat[select_bias(label)] = [1, position == 0, position == images.shape[0] - 1]
    return feat


def binary_feature(label, next_label):
    feat = np.zeros(NB_FEATURES)
    feat[select_transition(label, next_label)] = 1
    return feat


def word_feature(images, labels):
    feat = np.zeros(NB_FEATURES)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Not the same number of letter images and labels.")
    for t in range(images.shape[0]):
        feat[select_emission(labels[t])] += images[t]
        feat[select_bias(labels[t])] += [1, t == 0, t == images.shape[0] - 1]
    for t in range(images.shape[0] - 1):
        feat[select_transition(labels[t], labels[t + 1])] += 1
    return feat


########################################################################################################################
# SCORES
########################################################################################################################
def slow_unary_scores(word, weights):
    """Return the unary scores of word given by weights. This function is defined for the sake of clarity.
    A faster version is given below.

    :param word:
    :param weights:
    :return:
    """
    uscores = np.zeros([word.shape[0], ALPHABET_SIZE])
    for t in range(word.shape[0]):
        for label in range(ALPHABET_SIZE):
            uscores[t, label] = np.dot(weights, unary_feature(word, label, t))
    return uscores


def slow_binary_scores(word, weights):
    """Return the binary scores of word given by weights. This function is defined for the sake of clarity.
    A faster version is given below.

    :param word:
    :param weights:
    :return:
    """
    bscores = np.zeros([word.shape[0] - 1, ALPHABET_SIZE, ALPHABET_SIZE])
    for t in range(word.shape[0] - 1):
        for label in range(ALPHABET_SIZE):
            for next_label in range(ALPHABET_SIZE):
                bscores[t, label, next_label] = np.dot(weights, binary_feature(label, next_label))
    return bscores


def unary_scores(images, weights):
    """Return the unary scores of word given by weights.

    :param images:
    :param weights:
    :return:
    """
    chain_length = images.shape[0]
    uscores = np.zeros([chain_length, ALPHABET_SIZE])
    for t in range(chain_length):
        bias_selector = np.array([1, t == 0, t == chain_length - 1], dtype=int)
        for label in range(ALPHABET_SIZE):
            uscores[t, label] = np.dot(weights[select_emission(label)], images[t]) \
                                + np.dot(weights[select_bias(label)], bias_selector)
    return uscores


def binary_scores(word_length, weights):
    """Return the binary scores of word given by weights.

    :param word_length:
    :param weights:
    :return:
    """
    bscores = np.empty([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE])
    bscores[:] = np.reshape(weights[select_transition(-1, -1)], (ALPHABET_SIZE, ALPHABET_SIZE))
    # the code below is more understandable but slower
    # for t in range(word.shape[0]-1):
    #    for label in range(ALPHABET_SIZE):
    #        for next_label in range(ALPHABET_SIZE):
    #            binary_scores[t,label,next_label] = select_transition(weights,label,next_label)
    return bscores


########################################################################################################################
# ORACLES
########################################################################################################################
def sum_product(uscores, bscores, log=False):
    """Apply the sum-product algorithm on a chain

    :param uscores: array T*Alphabet, scores on individual nodes
    :param bscores: array (T-1)*Alphabet, scores on the edges
    :param log: if True, return the log-marginals
    :return: marginals on nodes, marginals on edges, log-partition
    """
    # I keep track of the log messages instead of the messages, to favor stability
    chain_length = uscores.shape[0]

    # backward pass
    backward_messages = np.zeros([chain_length - 1, ALPHABET_SIZE])
    backward_messages[-1] = utils.logsumexp(bscores[-1] + uscores[-1])
    for t in range(chain_length - 3, -1, -1):
        backward_messages[t] = utils.logsumexp(bscores[t] + uscores[t + 1] + backward_messages[t + 1])

    # we compute the log-partition and include it in the forward messages
    log_partition = utils.logsumexp(backward_messages[0] + uscores[0])

    # forward pass
    forward_messages = np.zeros([chain_length - 1, ALPHABET_SIZE])
    forward_messages[0] = utils.logsumexp(bscores[0].T + uscores[0] - log_partition)
    for t in range(1, chain_length - 1):
        forward_messages[t] = utils.logsumexp(bscores[t].T + uscores[t] + forward_messages[t - 1])

    unary_marginals = np.empty([chain_length, ALPHABET_SIZE])
    unary_marginals[0] = uscores[0] + backward_messages[0] - log_partition
    unary_marginals[-1] = forward_messages[-1] + uscores[-1]
    for t in range(1, chain_length - 1):
        unary_marginals[t] = forward_messages[t - 1] + uscores[t] + backward_messages[t]

    binary_marginals = np.empty([chain_length - 1, ALPHABET_SIZE, ALPHABET_SIZE])
    binary_marginals[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] + backward_messages[1] - log_partition
    binary_marginals[-1] = forward_messages[-2, :, np.newaxis] + uscores[-2, :, np.newaxis] + bscores[-1] + uscores[-1]
    for t in range(1, chain_length - 2):
        binary_marginals[t] = forward_messages[t - 1, :, np.newaxis] + uscores[t, :, np.newaxis] + bscores[t] + uscores[
            t + 1] + backward_messages[t + 1]

    if log:
        return LogProbability(unary_marginals, binary_marginals), log_partition
    else:
        return Probability(np.exp(unary_marginals), np.exp(binary_marginals)), log_partition


def viterbi(uscores, bscores):
    # I keep track of the score instead of the potentials
    # because summation is more stable than multiplication
    chain_length = uscores.shape[0]

    # backward pass
    argmax_messages = np.empty([chain_length - 1, ALPHABET_SIZE], dtype=int)
    max_messages = np.empty([chain_length - 1, ALPHABET_SIZE], dtype=float)
    tmp = bscores[-1] + uscores[-1]
    # Find the arg max
    argmax_messages[-1] = np.argmax(tmp, axis=-1)
    # Store the max
    max_messages[-1] = tmp[np.arange(ALPHABET_SIZE), argmax_messages[-1]]
    for t in range(chain_length - 3, -1, -1):
        tmp = bscores[t] + uscores[t + 1] + max_messages[t + 1]
        argmax_messages[t] = np.argmax(tmp, axis=-1)
        max_messages[t] = tmp[np.arange(ALPHABET_SIZE), argmax_messages[t]]

    # Label to be returned
    global_argmax = np.empty(chain_length, dtype=int)

    # forward pass
    tmp = max_messages[0] + uscores[0]
    global_argmax[0] = np.argmax(tmp)
    global_max = tmp[global_argmax[0]]
    for t in range(1, chain_length):
        global_argmax[t] = argmax_messages[t - 1, global_argmax[t - 1]]

    return global_argmax, global_max


########################################################################################################################
# OBJECT ORIENTED
########################################################################################################################
class Features:
    def __init__(self, unary=None, binary=None):
        if unary is None:
            self.unary = np.zeros([ALPHABET_SIZE, NB_PIXELS + 3])
        else:
            self.unary = unary
        if binary is None:
            self.binary = np.zeros([ALPHABET_SIZE, ALPHABET_SIZE])
        else:
            self.binary = binary

    def _add_unary(self, images, label, position):
        self.unary[label, :-3] += images[position]
        self.unary[label, -3:] += [1, position == 0, position == images.shape[0] - 1]

    def _add_binary(self, label, next_label):
        self.binary[label, next_label] += 1

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
            self.unary[:, :-3] += np.sum(images, axis=0) / ALPHABET_SIZE
            self.unary[:, -3] = images.shape[0] / ALPHABET_SIZE
            self.unary[:, -2:] = 1 / ALPHABET_SIZE
        else:
            # tmp is a vertical concatenation of the images in the word and the bias terms
            tmp = np.empty([images.shape[0], NB_PIXELS + 3])
            tmp[:, :-3] = images
            tmp[:, -3] = 1
            tmp[:, -2:] = 0
            tmp[0, -2] = 1
            tmp[-1, -1] = 1
            self.unary += np.dot(unary_marginals.T, tmp)

    def _add_binary_centroid(self, binary_marginals=None):
        if binary_marginals is None:  # assume uniform marginal
            self.binary += 1 / ALPHABET_SIZE ** 2
        else:
            self.binary += np.sum(binary_marginals, axis=0)

    def add_centroid(self, images, marginals=None):
        if marginals is None:  # assume uniform marginal
            self._add_unary_centroid(images, None)
            self._add_binary_centroid(None)
        else:
            self._add_unary_centroid(images, marginals.unary)
            self._add_binary_centroid(marginals.binary)

    def to_array(self):
        feat = np.empty(NB_FEATURES)
        feat[select_emission(-1)] = self.unary[:, :-3].flatten()
        feat[select_transition(-1, -1)] = self.binary.flatten()
        feat[select_bias(-1)] = self.unary[:, -3:].flatten()
        return feat

    @staticmethod
    def from_array(feat):
        unary = np.empty([ALPHABET_SIZE, NB_PIXELS + 3])
        unary[:, :-3] = feat[select_emission(-1)].reshape([ALPHABET_SIZE, -1])
        unary[:, -3:] = feat[select_bias(-1)].reshape(ALPHABET_SIZE, -1)
        binary = feat[select_transition(-1, -1)].reshape(ALPHABET_SIZE, -1)
        return Features(unary, binary)

    def display(self):
        emissions = letters2wordimage(self.unary[:, :-3])
        plt.matshow(emissions)
        ticks_positions = np.linspace(0, emissions.shape[1], ALPHABET_SIZE + 2).astype(int)[1:-1]
        plt.xticks(ticks_positions, list(ALPHABET))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.matshow(self.binary)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.yticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Transition Features")
        rescale_bias = np.array([1 / 7.5, 1, 1])
        plt.matshow((self.unary[:, -3:] * rescale_bias).T)
        plt.xticks(range(26), [ALPHABET[x] for x in range(26)])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Bias features")


class Probability:
    """Represent a conditional probability p(y|x) under the form of MARGINALS.
    Can also represent the ascent direction or the score.
    Ony has a finite precision on the small numbers.
    Inappropriate to handle the derivatives of the entropy or the KL.
    """

    def __init__(self, unary=None, binary=None, word_length=None):
        if unary is None or binary is None:
            self.length = word_length
            self.unary = np.ones([word_length, ALPHABET_SIZE]) / ALPHABET_SIZE
            self.binary = np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) / ALPHABET_SIZE ** 2
        else:
            if unary.shape[0] != binary.shape[0] + 1:
                raise ValueError("Wrong size of marginals: %i vs %i" % (unary.shape[0], binary.shape[0]))
            self.length = unary.shape[0]
            self.unary = unary
            self.binary = binary

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

    def inner_product(self, other_marginals):
        """Return the special inner product where the marginals on the separations are subtracted."""
        return np.sum(self.binary + other_marginals.binary) - \
               np.sum(self.unary[1:-1] + other_marginals.unary[1:-1])

    def add(self, other_marginals):
        return Probability(unary=self.unary + other_marginals.unary,
                           binary=self.binary + other_marginals.binary)

    def subtract(self, other_marginals):
        return Probability(unary=self.unary - other_marginals.unary,
                           binary=self.binary - other_marginals.binary)

    def multiply(self, other_marginals):
        return Probability(unary=self.unary * other_marginals.unary,
                           binary=self.binary * other_marginals.binary)

    def divide(self, other_marginals):
        return Probability(unary=self.unary / other_marginals.unary,
                           binary=self.binary / other_marginals.binary)

    def multiply_scalar(self, scalar):
        return Probability(unary=scalar * self.unary,
                           binary=scalar * self.binary)

    def square(self):
        return Probability(unary=self.unary ** 2, binary=self.binary ** 2)

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


class LogProbability:
    """Represent a conditional probability p(y|x) under the form of LOG-marginals."""

    def __init__(self, unary=None, binary=None, word_length=None):
        if unary is None or binary is None:  # assume uniform
            self.length = word_length
            self.unary = np.ones([word_length, ALPHABET_SIZE]) * (-np.log(ALPHABET_SIZE))
            self.binary = np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) * (-2 * np.log(ALPHABET_SIZE))
        else:  # take what is given
            if unary.shape[0] != binary.shape[0] + 1:
                raise ValueError("Wrong size of marginals: %i vs %i" % (unary.shape[0], binary.shape[0]))
            self.length = unary.shape[0]
            self.unary = unary
            self.binary = binary

    def entropy(self):
        return max(0, utils.log_entropy(self.binary) - utils.log_entropy(self.unary[1:-1]))

    def kullback_leibler(self, other_marginals):
        return max(0, utils.log_kullback_leibler(self.binary, other_marginals.binary) \
                   - utils.log_kullback_leibler(self.unary[1:-1], other_marginals.unary[1:-1]))

    def inner_product(self, other):
        """Return the special inner product where the marginals on the separations are subtracted."""
        return np.sum(np.exp(self.binary + other.binary)) - \
               np.sum(np.exp(self.unary[1:-1] + other.unary[1:-1]))

    def multiply(self, other):
        return LogProbability(unary=self.unary + other.unary,
                              binary=self.binary + other.binary)

    def divide(self, other):
        return LogProbability(unary=self.unary - other.unary,
                              binary=self.binary - other.binary)

    def multiply_scalar(self, scalar):
        return LogProbability(unary=np.log(scalar) + self.unary,
                              binary=np.log(scalar) + self.binary)

    def convex_combination(self, other, gamma):
        return LogProbability(unary=np.log((1 - gamma) * np.exp(self.unary) + gamma * np.exp(other.unary)),
                              binary=np.log((1 - gamma) * np.exp(self.binary) + gamma * np.exp(other.binary)))

    def inverse(self):
        return LogProbability(unary=-self.unary,
                              binary=-self.binary)

    def to_probability(self):
        return Probability(unary=np.exp(self.unary),
                           binary=np.exp(self.binary))

    @staticmethod
    def infer_from_weights(images, weights):
        uscores = unary_scores(images, weights)
        bscores = binary_scores(images.shape[0], weights)
        return sum_product(uscores, bscores, log=True)[0]


########################################################################################################################
# NEW STUFF
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
def marginals_to_weights(images, labels, marginals=None, log=False):
    nb_words = labels.shape[0]
    ground_truth_centroid = Features()
    ground_truth_centroid.add_dictionary(images, labels)
    marginals_centroid = Features()
    if marginals is None:  # assume uniform
        for image in images:
            marginals_centroid.add_centroid(image)
    else:
        for image, margs in zip(images, marginals):
            if log:
                marginals_centroid.add_centroid(image, margs.to_probability())
            else:
                marginals_centroid.add_centroid(image, margs)
    ground_truth_centroid = ground_truth_centroid.to_array()
    marginals_centroid = marginals_centroid.to_array()
    return (ground_truth_centroid - marginals_centroid) / nb_words


def dual_score(weights, marginals, regularization_parameter):
    ans = - regularization_parameter / 2 * np.sum(weights ** 2)
    ans += sum([margs.entropy() for margs in marginals]) / marginals.shape[0]
    return ans


def duality_gaps(marginals, weights, images):
    ans = []
    for margs, imgs in zip(marginals, images):
        newmargs = LogProbability.infer_from_weights(imgs, weights)
        ans.append(margs.kullback_leibler(newmargs))
    return np.array(ans)


def sdca(x, y, regu, npass=5, update_period=5, precision=1e-5, subprecision=1e-16, non_uniformity=0,
         step_size=None, init='uniform', _debug=False):
    """Update alpha and w with the stochastic dual coordinate ascent algorithm to fit the model to the
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
    scaling = - regu * nb_words / 2
    if nb_words != x.shape[0]:
        raise ValueError("Not the same number of labels (%i) and images (%i) inside training set." \
                         % (nb_words, x.shape[0]))
    if init == "uniform":
        marginals = uniform_marginals(y, log=True)
        weights = marginals_to_weights(x, y) / regu
    elif init == "random":
        weights = np.random.randn(NB_FEATURES)
        marginals = np.array([LogProbability.infer_from_weights(imgs, weights) for imgs in x])
        weights = marginals_to_weights(x, y, marginals, log=True)
        Features.from_array(weights).display()
    else:
        raise ValueError("Not a valid argument for init: %r" % init)

    ##################################################################################
    # OBJECTIVES : dual objective and duality gaps
    ##################################################################################
    entropies = np.array([margs.entropy() for margs in marginals])
    dual_objective = entropies.mean() - regu / 2 * np.sum(weights ** 2)

    new_marginals = [LogProbability.infer_from_weights(imgs, weights) for imgs in x]
    dgaps = np.array([margs.kullback_leibler(newmargs) for margs, newmargs in zip(marginals, new_marginals)])
    duality_gap = dgaps.sum()

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
    t = 0
    while t < nb_words * npass and duality_gap > precision:
        t += 1

        ##################################################################################
        # SAMPLING
        ##################################################################################
        if np.random.rand() > non_uniformity:  # then sample uniformly
            i = np.random.randint(nb_words)
        else:  # sample proportionally to the duality gaps
            i = sampler.sample()

        ##################################################################################
        # MARGINALIZATION ORACLE
        ##################################################################################
        margs_i = LogProbability.infer_from_weights(x[i], weights)
        # assert margs_i.are_consistent()
        # assert margs_i.are_densities(1)
        # assert margs_i.are_positive(), (display_word(y[i],x[i]),
        #                                 margs_i.display(),
        #                                 Features.from_array(weights).display())

        ##################################################################################
        # ASCENT DIRECTION : and primal movement
        ##################################################################################
        dual_direction = margs_i.to_probability().subtract(marginals[i].to_probability())
        assert dual_direction.are_densities(integral=0)
        assert dual_direction.are_consistent()
        primal_direction = Features()
        primal_direction.add_centroid(x[i], dual_direction)
        # Centroid of the corrected features in the dual direction
        # = Centroid of the real features in the opposite of the dual direction
        primal_direction = - primal_direction.to_array() / regu / nb_words

        ##################################################################################
        # LINE SEARCH : find the optimal step size gammaopt or use a fixed one
        ##################################################################################
        quadratic_coeff = scaling * np.sum(primal_direction ** 2)
        linear_coeff = 2 * scaling * np.dot(weights, primal_direction)
        if step_size:
            gammaopt = step_size
        else:
            # line search function and its derivatives
            # def f(gamma):
            #    newmargs = marginals[i].add(dual_direction.multiply_scalar(gamma))
            #    return  newmargs.entropy() + gamma**2 * quadratic_coeff + gamma * linear_coeff

            def gf(gamma):
                newmargs = marginals[i].convex_combination(margs_i, gamma)
                # assert newmargs.are_densities(1)
                # assert newmargs.are_consistent()
                # assert newmargs.are_positive(), newmargs.display
                return - dual_direction.inner_product(newmargs) + gamma * 2 * quadratic_coeff + linear_coeff

            def ggf(gamma):
                newmargs = marginals[i].convex_combination(margs_i, gamma)
                return - dual_direction.square().inner_product(newmargs.inverse().to_probability()) \
                       + 2 * quadratic_coeff

            gammaopt, subobjective = utils.find_root_decreasing(gf, ggf, precision=subprecision)

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
        marginals[i] = marginals[i].convex_combination(margs_i, gammaopt)
        weights += gammaopt * primal_direction
        # assert marginals[i].are_densities(1)
        # assert marginals[i].are_consistent()
        # assert marginals[i].are_positive(), (display_word(y[i], x[i]),
        #                                      marginals[i].display(),
        #                                      Features.from_array(weights).display())

        ##################################################################################
        # DUALITY GAP ESTIMATE
        ##################################################################################
        sampler.update(marginals[i].kullback_leibler(margs_i), i)
        duality_gap = sampler.get_total()
        assert duality_gap >= 0, (
            duality_gap,
            sampler.get_score(i),
            nb_words
        )

        ##################################################################################
        # ANNEX
        ##################################################################################
        if _debug:
            # Update the dual objective and the entropy
            tmp = marginals[i].entropy()
            dual_objective += \
                (tmp - entropies[i] + gammaopt ** 2 * quadratic_coeff + gammaopt * linear_coeff) / nb_words
            entropies[i] = tmp
            # Append relevant variables
            annex.append([-quadratic_coeff, -linear_coeff, gammaopt,
                          dual_objective, duality_gap, sampler.get_score(i), i])

        if t % nb_words == 0:
            ##################################################################################
            # OBJECTIVES : after each pass over the data, compute the duality gap
            ##################################################################################
            t1 = time.time()
            dgaps = duality_gaps(marginals, weights, x)
            sampler = random_counters.RandomCounters(dgaps)
            duality_gap = sampler.get_total()
            objective.append(duality_gap)
            # if t % (update_period * nb_words) == 0 and non_uniformity > 0:
            #     pass
            #     ###################################################################################
            #     # DUALITY GAPS: perform a batch update after every update_period epochs
            #     # To reduce the staleness for the non-uniform sampling
            #     # To monitor the objective and provide a stopping criterion
            #     ##################################################################################
            # delta_time += time.time() - t1
            timing.append(time.time() - delta_time)

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
