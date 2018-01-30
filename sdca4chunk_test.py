# standard imports
import os
import time

import numpy as np

import chunk.features
import sdca
from labeled_data import LabeledSequenceData

train_size = 600
test_size = 100

# # Load data
# trainfile = "../data/conll2000/train.att.txt"
# testfile = "../data/conll2000/test.att.txt"
#
# dattributes = parse.build_dictionary(trainfile, min_occurence=3, nb_sentences=train_size)
# xtrain, ytrain = parse.read_data(trainfile, dattributes, nb_sentences=train_size)
# xtest, ytest = parse.read_data(testfile, dattributes, nb_sentences=test_size)
# nb_features = len(dattributes)

# Load data.mat
trainfile = "../lib/sag4crf/data/coNLL_train.mat"
testfile = "../lib/sag4crf/data/coNLL_test.mat"

xtrain, ytrain, attributes = chunk.parse.read_mat(trainfile, nb_sentences=train_size)
xtest, ytest, _ = chunk.parse.read_mat(testfile, attributes=attributes, nb_sentences=test_size)
nb_features = attributes.total

print("Number of different features:", nb_features)
train_size = xtrain.shape[0]
print("Size of training set:", train_size)
test_size = xtest.shape[0]
print("Size of test set:", test_size)

trainset = LabeledSequenceData(xtrain, ytrain)
testset = LabeledSequenceData(xtest, ytest)

regu = 1 / train_size

parameters = {
    'regularization': regu,
    'npass': 100,
    'sampling': 'gap',
    'non_uniformity': .8,
    'sampler_period': None,
    'precision': 1e-7,
    # 'fixed_step_size': .01
}
print(parameters)

time_stamp = time.strftime("%Y%m%d_%H%M%S")

dirname = "logs/conll2000/{}_n{}_{}{}".format(
    time_stamp,
    train_size,
    parameters['sampling'],
    parameters['non_uniformity']
)

parameters['logdir'] = dirname

if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists(dirname):
    os.mkdir(dirname)

# write all parameters to text file
with open(dirname + '/parameters.txt', 'w') as file:
    message = "Time: {} \n Size of the feature map: {} \n " \
              "Train size: {} \n Test size: {} \n".format(
        time_stamp,
        nb_features,
        train_size,
        test_size
    )
    for key, value in parameters.items():
        file.write("\n {} : {}".format(key, value))

# TODO implement the initialization as a clean module
# TODO wrap SDCA with a try finally to be sure to save results and weights !

fullweights, fullmargs = \
    sdca.sdca(features_cls=chunk.features, trainset=trainset, testset=testset, **parameters)

print("End")
os.system('say "I am done."')

np.save(dirname + "/weights.npy", fullweights)
