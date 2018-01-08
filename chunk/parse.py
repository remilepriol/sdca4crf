import csv

import matplotlib.pyplot as plt
import numpy as np

from chunk.features import TAG2INT


def read_data(file):
    with open(file) as f:
        reader = csv.reader(f, delimiter=' ')
        full = []
        sentence = []
        for row in reader:
            if len(row) == 0:
                full.append(sentence)
                sentence = []
            else:
                sentence.append(row)
        full = np.array(full)
        return full


def split_points_labels(full):
    x = []
    y = []
    for sentence in full:
        xi = []
        yi = []
        for word in sentence:
            xi.append(word[0:2])
            yi.append(tag2int(word[2]))
        x.append(xi)
        y.append(yi)
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
def tags_list(y):
    tags = []
    for sent in y:
        tags.extend(sent)

    plt.hist(tags, log=True)
    plt.xticks(rotation=50)
    plt.show()

    tags = set(tags)
    return tags


def tag2int(tag):
    return TAG2INT[tag]


def print_sentences(x, start, end):
    for sent in x[start:end]:
        words = [w[0] for w in sent]
        print(' '.join(words))
