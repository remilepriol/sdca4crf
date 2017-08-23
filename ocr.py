# standard imports
import csv

import matplotlib.pyplot as plt
import numpy as np

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
# FEATURES
########################################################################################################################
def unary_feature(word, label, position):
    feat = np.zeros(ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE + 3))
    # image value
    feat[label * NB_PIXELS:(label + 1) * NB_PIXELS] = word[position]
    # bias
    feat[ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE) + label] = 1
    if position == 0:  # beginning of word
        feat[ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE + 1) + label] = 1
    elif position == word.shape[0] - 1:  # end of word
        feat[ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE + 2) + label] = 1

    return feat


def binary_feature(label, next_label):
    feat = np.zeros(NB_FEATURES)
    feat[ALPHABET_SIZE * (NB_PIXELS + label) + next_label] = 1
    return feat


def word_feature(images, labels):
    feat = np.zeros(NB_FEATURES)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("label is not the same size as this word")
    for t in range(images.shape[0]):
        feat += unary_feature(images, labels[t], t)
    for t in range(images.shape[0] - 1):
        feat += binary_feature(labels[t], labels[t + 1])
    return feat


def select_emission(features, label):
    start = label * NB_PIXELS
    return features[start:start + NB_PIXELS]


def select_transition(features, label, next_label):
    return features[ALPHABET_SIZE * (NB_PIXELS + label) + next_label]


def select_all_transitions(features):
    start = ALPHABET_SIZE * NB_PIXELS
    return features[start:start + ALPHABET_SIZE * ALPHABET_SIZE]


def select_bias(features, label):
    start = ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE) + label
    return features[start:start + 3]


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


def fast_unary_scores(word, weights):
    """Return the unary scores of word given by weights.

    :param word:
    :param weights:
    :return:
    """
    chain_length = word.shape[0]
    unary_scores = np.zeros([chain_length, ALPHABET_SIZE])
    for t in range(chain_length):
        bias_selector = np.array([1, t == 0, t == chain_length - 1], dtype=int)
        for label in range(ALPHABET_SIZE):
            unary_scores[t, label] = np.dot(select_emission(weights, label), word[t]) \
                                     + np.dot(select_bias(weights, label), bias_selector)
    return unary_scores


def fast_binary_scores(word, weights):
    """Return the binary scores of word given by weights.

    :param word:
    :param weights:
    :return:
    """
    binary_scores = np.empty([word.shape[0] - 1, ALPHABET_SIZE, ALPHABET_SIZE])
    binary_scores[:] = np.reshape(select_all_transitions(weights), (ALPHABET_SIZE, ALPHABET_SIZE))
    # the code below is more understandable but slower
    # for t in range(word.shape[0]-1):
    #    for label in range(ALPHABET_SIZE):
    #        for next_label in range(ALPHABET_SIZE):
    #            binary_scores[t,label,next_label] = select_transition(weights,label,next_label)
    return binary_scores


########################################################################################################################
# MARGINALIZATION
########################################################################################################################
def logsumexp(v):
    vmax = np.amax(v, axis=-1, keepdims=True)
    return vmax.squeeze(axis=-1) + np.log(np.sum(np.exp(v - vmax), axis=-1))


def backward_forward(unary_scores, binary_scores):
    # I keep track of the log messages instead of the messages, to favor stability
    chain_length = unary_scores.shape[0]

    # backward pass
    backward_messages = np.zeros([chain_length - 1, ALPHABET_SIZE])
    backward_messages[-1] = logsumexp(binary_scores[-1] + unary_scores[-1])
    for t in range(chain_length - 3, -1, -1):
        backward_messages[t] = logsumexp(binary_scores[t] + unary_scores[t + 1] + backward_messages[t + 1])

    # we compute the log-partition and include it in the forward messages
    log_partition = logsumexp(backward_messages[0] + unary_scores[0])

    # forward pass
    forward_messages = np.zeros([chain_length - 1, ALPHABET_SIZE])
    forward_messages[0] = logsumexp(binary_scores[0].T + unary_scores[0] - log_partition)
    for t in range(1, chain_length - 1):
        forward_messages[t] = logsumexp(binary_scores[t].T + unary_scores[t] + forward_messages[t - 1])

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
