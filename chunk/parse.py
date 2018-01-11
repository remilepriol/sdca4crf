"""Parse the data file containing the unary attributes created by CRF suite."""
import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

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


def read_data(file, attributes_dictionary, nb_sentences=None):
    """Read the data file and output x,y tuple."""

    total_attributes = len(attributes_dictionary)

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

                # build a sparse binary embedding of the attributes
                xij = []
                for att in row[1:]:
                    if att in attributes_dictionary:
                        xij.append(attributes_dictionary[att])
                xi.append(xij)

            else:  # end of sentence
                # transform xi from list to sparse matrix
                # each row is the embedding of a word.
                nb_attributes = 0
                row_ind = []
                col_ind = []
                for j, xij in enumerate(xi):
                    nb_attributes += len(xij)
                    row_ind.extend([j] * len(xij))
                    col_ind.extend(xij)

                xxi = sps.csr_matrix(
                    ([1] * nb_attributes,  # values
                     (row_ind, col_ind)),  # row and column indices
                    shape=(len(xi), total_attributes))  # shape

                # append new sentence to the list
                x.append(xxi)
                y.append(yi)
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
