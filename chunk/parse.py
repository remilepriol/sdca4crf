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


def build_dictionary(file, min_occurence=3):
    """Build a dictionary mapping all the possible attributes to an integer."""

    with open(file) as f:

        reader = csv.reader(f, delimiter='\t')
        attributes = []
        for row in reader:
            if len(row) > 0:
                attributes.extend(row[1:])

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
                column_indices = []
                for att in row[1:]:
                    if att in attributes_dictionary:
                        column_indices.append(attributes_dictionary[att])

                nb_attributes = len(column_indices)
                xij = sps.csr_matrix(
                    (np.ones(nb_attributes),  # values
                     (np.zeros(nb_attributes), column_indices)),  # row and column indices
                    shape=(1, total_attributes))  # shape

                xi.append(xij)

            else:  # end of sentence
                # append new sentence to the list
                x.append(xi)
                y.append(yi)
                xi = []
                yi = []
                count_sentences += 1
                if nb_sentences is not None and count_sentences >= nb_sentences:
                    break

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
