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

    def __init__(self, attributes, attributes_dictionary):
        self.dimension = len(attributes_dictionary)
        indices = []
        for att in attributes:
            if att in attributes_dictionary:
                indices.append(attributes_dictionary[att])
        self.indices = np.array(indices).astype(int)


def read_data(file, attributes_dictionary, nb_sentences=None):
    """Read the data file and output x,y tuple."""

    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')
        x = []
        y = []
        xi = []
        yi = []
        count_sentences = 0
        for row in reader:
            if len(row) > 0:
                # append to sentence
                yi.append(TAG2INT[row[0]])

                # each sentence is represented as an array of words,
                # where each word is an array containing the indices of its attributes
                # as defined by the dictionary.
                xi.append(WordEmbedding(row[1:], attributes_dictionary))

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
