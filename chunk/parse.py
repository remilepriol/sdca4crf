import csv

import matplotlib.pyplot as plt

ALPHABET = [
    'I-INTJ', 'I-CONJP', 'O', 'I-ADVP', 'B-VP', 'B-LST', 'I-LST', 'I-PRT', 'I-SBAR', 'I-UCP',
    'B-CONJP', 'I-PP', 'B-ADVP', 'I-NP', 'I-VP', 'B-ADJP', 'B-PP', 'B-NP', 'I-ADJP', 'B-PRT',
    'B-SBAR', 'B-UCP', 'B-INTJ'
]
ALPHALEN = len(ALPHABET)
TAG2INT = {tag: i for i, tag in enumerate(ALPHABET)}
TEMPLATES = (
    ((0, -2),),
    ((0, -1),),
    ((0, 0),),
    ((0, 1),),
    ((0, 2),),
    ((0, -1), (0, 0)),
    ((0, 0), (0, 1)),
    ((1, -2),),
    ((1, -1),),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
    ((1, -2), (1, -1)),
    ((1, -1), (1, 0)),
    ((1, 0), (1, 1)),
    ((1, 1), (1, 2)),
    ((1, -2), (1, -1), (1, 0)),
    ((1, -1), (1, 0), (1, 1)),
    ((1, 0), (1, 1), (1, 2)),
)


def build_dictionaries(file):
    """Build 3 dictionaries : one for words, one for part-of-speech tags and
    one for chunk tags from the training set. Our goal is to output a chunk tag for each token.
    If a word or a tag is present in the test set but not in the training set, it will be omitted
    as it won't have any weight into the linear model anyway.
    """

    with open(file) as f:

        reader = csv.reader(f, delimiter=' ')

        # create lists of values
        words = []
        pos_tags = []
        chunk_tags = []
        for row in reader:
            if len(row) > 0:
                words.append(row[0])
                pos_tags.append(row[1])
                chunk_tags.append(row[2])

        # remove duplicates
        words = set(words)
        pos_tags = set(pos_tags)
        chunk_tags = set(chunk_tags)

        # make dictionaries
        dwords = {tag: i for i, tag in enumerate(words)}
        dpos_tags = {tag: i for i, tag in enumerate(pos_tags)}
        dchunk_tags = {tag: i for i, tag in enumerate(chunk_tags)}

        return dwords, dpos_tags, dchunk_tags


def read_data(file, nb_sentences=None):
    """Read the data file and output x,y tuple."""

    with open(file) as f:
        reader = csv.reader(f, delimiter=' ')
        x = []
        y = []
        xi = []
        yi = []
        count_sentences = 0
        for row in reader:
            if len(row) == 0:  # end of sentence
                # append new sentence to the list
                x.append(xi)
                y.append(yi)
                xi = []
                yi = []
                count_sentences += 1
                if nb_sentences is not None and count_sentences >= nb_sentences:
                    break
            else:
                # append to sentence
                xi.append(row[:2])
                yi.append(row[2])

        return x, y


def map_to_integers(x, y, dwords, dpos, dchunks):
    """Take x and y lists of strings and transform them into lists of indexes given by the
    dictionaries."""


def get_unary_attributes(sentence, t, templates=TEMPLATES):
    keys = []
    for template in templates:
        values = []
        for (field, offset) in template:
            p = t + offset
            if p < 0 or p >= len(sentence):
                values = []
                break
            values.append(sentence[p][field])
        if values:
            keys.append("%s=%s" % (str(template), '|'.join(values)))
    return keys


def embed(sentence, t):
    keys = get_unary_attributes(sentence, t)



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

