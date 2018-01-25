"""Parse the data file containing the unary attributes created by CRF suite."""
import csv

import matplotlib.pyplot as plt
import numpy as np

ALPHABET = [
    'I-INTJ', 'I-CONJP', 'O', 'I-ADVP', 'B-VP', 'B-LST', 'I-LST', 'I-PRT', 'I-SBAR', 'I-UCP',
    'B-CONJP', 'I-PP', 'B-ADVP', 'I-NP', 'I-VP', 'B-ADJP', 'B-PP', 'B-NP', 'I-ADJP', 'B-PRT',
    'B-SBAR', 'B-UCP', 'B-INTJ'
]
ALPHALEN = len(ALPHABET)
TAG2INT = {tag: i for i, tag in enumerate(ALPHABET)}


def build_dictionary(file, min_occurence=3, nb_sentences=None):
    """Build a dictionary mapping all the possible attributes to an integer."""

    with open(file) as f:

        reader = csv.reader(f, delimiter='\t')
        attributes = []
        count_sentences = 0
        for row in reader:
            if len(row) > 0:
                attributes.extend(row[1:])
            else:
                count_sentences += 1
                if nb_sentences is not None and count_sentences >= nb_sentences:
                    break

        # remove duplicates
        uattributes, counts = np.unique(attributes, return_counts=True)
        # make dictionaries
        dattributes = {tag: i for i, tag in enumerate(uattributes[counts >= min_occurence])}

        return dattributes


class WordEmbedding:

    def __init__(self, dimension, indices):
        # each word is an array containing the indices of its attributes
        # as defined by the dictionary.
        self.dimension = dimension
        self.indices = indices

    @staticmethod
    def from_attributes(attributes, attributes_dictionary):
        dimension = len(attributes_dictionary)
        indices = []
        for att in attributes:
            if att in attributes_dictionary:
                indices.append(attributes_dictionary[att])
        indices = np.array(indices).astype(int)
        return WordEmbedding(dimension, indices)


def read_data(file, attributes_dictionary, nb_sentences=None):
    """Read the data file and output x,y tuple."""

    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')
        words = []
        labels = []
        for row in reader:
            if len(row) > 0:
                labels.append(TAG2INT[row[0]])
                words.append(WordEmbedding.from_attributes(row[1:], attributes_dictionary))
            else:
                labels.append(-1)
                words.append([])

        return aggregate_sentences(words, labels, nb_sentences)


class Attributes:
    def __init__(self, nb_attributes):
        self.counts = nb_attributes  # number of attributes for each kind
        self.cumulative = np.cumsum(nb_attributes)
        self.total = self.cumulative[-1]


def read_mat(matfile, attributes=None, nb_sentences=None):
    """Read the data file used by Mark Schmidt and output x,y tuple."""
    from scipy.io import loadmat
    dic = loadmat(matfile)
    words = dic['X'].squeeze()
    # N*19 array. Each line is one word.Each column is one kind of attribute.
    # The number (i,j) is the index of the attribute j for word i on the dictionary of attributes j
    # If a number is 0 it means it is undefined.
    labels = dic['y'].squeeze().astype(np.int16)  # N*1 array of labels.

    if attributes is None:  # this is the train set
        attributes = Attributes(np.amax(words, axis=0))

    words = words.astype(float)
    words[np.logical_or(words == 0, words > attributes.counts)] = np.nan  # remove non-existing
    # attributes and attributes absent from the training set.
    words[:, 1:] += attributes.cumulative[:-1]  # shift dictionary values
    words = np.nan_to_num(words).astype(np.int32)  # replace nan with zero again

    # We change values by minus 1 compared to matlab where matrices are indexed from 1
    words2 = [WordEmbedding(dimension=attributes.total, indices=word[word > 0] - 1)
              for word in words]
    x, y = aggregate_sentences(words2, labels - 1, nb_sentences)
    return x, y, attributes


def aggregate_sentences(words, labels, nb_sentences):
    x = []
    y = []
    xi = []
    yi = []
    count_sentences = 0
    for word, label in zip(words, labels):
        if label >= 0:
            # append to sentence
            yi.append(label)
            xi.append(word)
        else:  # end of sentence
            x.append(np.array(xi))
            y.append(np.array(yi))
            xi = []
            yi = []
            count_sentences += 1
            if nb_sentences is not None and count_sentences >= nb_sentences:
                break

    x = np.array(x)
    y = np.array(y)
    return x, y


def average_length(x):
    avg_sent = 0
    for sent in x:
        avg_sent += len(sent)
    avg_sent /= len(x)
    print(avg_sent)


# Get the list of potential tags
def chunk_tags_histogram(y):
    tags = []
    for sent in y:
        tags.extend(sent)
    plt.hist(tags, log=True)
    plt.xticks(rotation=50)
    plt.show()


def print_sentences(x, start, end):
    for sent in x[start:end]:
        words = [w[0] for w in sent]
        print(' '.join(words))
