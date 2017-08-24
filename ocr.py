# standard imports
import csv

import matplotlib.pyplot as plt
import numpy as np

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
    for letter in letters:
        labels.append(letter[LETTER_VALUE])
        images.append(letter[FIRST_PIXEL:])
        if letter[NEXT_ID] == -1:
            alllabels.append(np.array(labels))
            allimages.append(np.array(images))
            labels = []
            images = []
    return np.array(alllabels), np.array(allimages)


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


def binary_scores(images, weights):
    """Return the binary scores of word given by weights.

    :param images:
    :param weights:
    :return:
    """
    bscores = np.empty([images.shape[0] - 1, ALPHABET_SIZE, ALPHABET_SIZE])
    bscores[:] = np.reshape(weights[select_transition(-1, -1)], (ALPHABET_SIZE, ALPHABET_SIZE))
    # the code below is more understandable but slower
    # for t in range(word.shape[0]-1):
    #    for label in range(ALPHABET_SIZE):
    #        for next_label in range(ALPHABET_SIZE):
    #            binary_scores[t,label,next_label] = select_transition(weights,label,next_label)
    return bscores


########################################################################################################################
# MARGINALIZATION
########################################################################################################################
def backward_forward(unary_scores, binary_scores):
    # I keep track of the log messages instead of the messages, to favor stability
    chain_length = unary_scores.shape[0]

    # backward pass
    backward_messages = np.zeros([chain_length - 1, ALPHABET_SIZE])
    backward_messages[-1] = utils.logsumexp(binary_scores[-1] + unary_scores[-1])
    for t in range(chain_length - 3, -1, -1):
        backward_messages[t] = utils.logsumexp(binary_scores[t] + unary_scores[t + 1] + backward_messages[t + 1])

    # we compute the log-partition and include it in the forward messages
    log_partition = utils.logsumexp(backward_messages[0] + unary_scores[0])

    # forward pass
    forward_messages = np.zeros([chain_length - 1, ALPHABET_SIZE])
    forward_messages[0] = utils.logsumexp(binary_scores[0].T + unary_scores[0] - log_partition)
    for t in range(1, chain_length - 1):
        forward_messages[t] = utils.logsumexp(binary_scores[t].T + unary_scores[t] + forward_messages[t - 1])

    unary_marginals = np.empty([chain_length, ALPHABET_SIZE])
    unary_marginals[0] = np.exp(unary_scores[0] + backward_messages[0] - log_partition)
    unary_marginals[-1] = np.exp(forward_messages[-1] + unary_scores[-1])
    for t in range(1, chain_length - 1):
        unary_marginals[t] = np.exp(forward_messages[t - 1] + unary_scores[t] + backward_messages[t])

    binary_marginals = np.zeros([chain_length - 1, ALPHABET_SIZE, ALPHABET_SIZE])
    binary_marginals[0] = np.exp(unary_scores[0, :, np.newaxis] + binary_scores[0] + unary_scores[1]
                                 + backward_messages[1] - log_partition)
    binary_marginals[-1] = np.exp(forward_messages[-2, :, np.newaxis] + unary_scores[-2, :, np.newaxis]
                                  + binary_scores[-1] + unary_scores[-1])
    for t in range(1, chain_length - 2):
        binary_marginals[t] = np.exp(forward_messages[t - 1, :, np.newaxis] + unary_scores[t, :, np.newaxis]
                                     + binary_scores[t] + unary_scores[t + 1] + backward_messages[t + 1])

    return unary_marginals, binary_marginals


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
            self.add_word(images, labels)

    def _add_unary_centroid(self, images, unary_marginals=None):
        if unary_marginals is None:  # assume uniform marginal
            self.unary[:, :-3] += np.sum(images, axis=0) / ALPHABET_SIZE
            self.unary[:, -3] = images.shape[0] / ALPHABET_SIZE
            self.unary[:, -2:] = 1 / ALPHABET_SIZE
        else:
            # tmp is a vertical concatenation the images in the word and the bias terms
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


class Marginals():
    def __init__(self, unary=None, binary=None, word_length=None):
        if unary is None or binary is None:
            self.length = word_length
            self.unary = np.ones([word_length, ALPHABET_SIZE]) / ALPHABET_SIZE
            self.binary = np.ones([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE]) / (ALPHABET_SIZE ** 2)
        else:
            if unary.shape[0] != binary.shape[0] + 1:
                raise ValueError("Wrong size of marginals.")
            self.length = unary.shape[0]
            self.unary = unary
            self.binary = binary

    def _is_unary_densities(self):
        return np.isclose(self.unary.sum(axis=1), 1).all()

    def _is_binary_densities(self):
        return np.isclose(self.binary.sum(axis=(1, 2)), 1).all()

    def is_densities(self):
        return self._is_unary_densities() and self._is_binary_densities()

    def is_consistent(self):
        ans = True
        from_left_binary = np.sum(self.binary, axis=1)
        from_right_binary = np.sum(self.binary, axis=2)
        if not np.isclose(from_left_binary, self.unary[1:]).all():
            ans = False
            print("Marginalisation of the left of the binary marginals is inconsistent with unary marginals.")
        if not np.isclose(from_right_binary, self.unary[:-1]).all():
            ans = False
            print("Marginalisation of the right of the binary marginals is inconsistent with unary marginals.")
        if not np.isclose(from_right_binary[1:], from_left_binary[:-1]).all():
            ans = False
            print("Marginalisation of the left and right of the binary marginals are inconsistent.")
        return ans

    def entropy(self):
        return utils.entropy(self.binary) - utils.entropy(self.unary[1:-1])

    def kullback_leibler(self, other_marginals):
        return utils.kullback_leibler(self.binary, other_marginals.binary) \
               - utils.kullback_leibler(self.unary[1:-1], other_marginals.unary[1:-1])

    def add(self, other_marginals):
        return Marginals(unary=self.unary + other_marginals.unary,
                         binary=self.binary + other_marginals.unary)

    def subtract(self, other_marginals):
        return Marginals(unary=self.unary - other_marginals.unary,
                         binary=self.binary - other_marginals.unary)

    def multiply(self, other_marginals):
        return Marginals(unary=self.unary * other_marginals.unary,
                         binary=self.binary * other_marginals.unary)

    def divide(self, other_marginals):
        return Marginals(unary=self.unary / other_marginals.unary,
                         binary=self.binary / other_marginals.unary)

    def multiply_scalar(self, scalar):
        return Marginals(unary=scalar * self.unary,
                         binary=scalar * self.binary)

    def square(self):
        return Marginals(unary=self.unary ** 2, binary=self.binary ** 2)

    def log(self):
        return Marginals(unary=np.log(np.maximum(1e-16, self.unary)),
                         binary=np.log(np.maximum(1e-16, self.binary)))

    def inner_product(self, other_marginals):
        """ Return the special inner product where the marginals on the separations are subtracted.

        :param other_marginals:
        :return:
        """
        return np.sum(self.binary, other_marginals.binary) - \
               np.sum(self.unary[1:-1] * other_marginals.unary[1:-1])

    @staticmethod
    def infer_from_weights(images, weights):
        uscores = unary_scores(images, weights)
        bscores = binary_scores(images, weights)
        umargs, bmargs = backward_forward(uscores, bscores)
        return Marginals(unary=umargs, binary=bmargs)

    @staticmethod
    def dirac(labels):
        word_length = labels.shape[0]
        umargs = np.zeros([word_length, ALPHABET_SIZE])
        umargs[np.arange(word_length), labels] = 1
        bmargs = np.zeros([word_length - 1, ALPHABET_SIZE, ALPHABET_SIZE])
        bmargs[np.arange(word_length - 1), labels[:-1], labels[1:]] = 1
        return Marginals(unary=umargs, binary=bmargs)
