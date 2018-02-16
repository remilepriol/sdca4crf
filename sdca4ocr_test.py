# standard imports
import os
import time

import numpy as np

# custom imports
import sdca
from ocr import features, parse

images, labels, folds = parse.letters_to_labels_and_words(
    parse.read_data('../data/ocr/letter.data.tsv'))

radii = features.radii(images, labels)
max_radius = np.amax(radii)

training_size = 60000000
testing_size = 10000000

training_folds = set(range(0, 1))
training_mask = np.array([fold in training_folds for fold in folds])
xtrain = images[training_mask][:training_size]
ytrain = labels[training_mask][:training_size]
training_size = xtrain.shape[0]
print("Training folds: ", training_folds,
      "\t Number of training words:", training_size)

testing_folds = set(range(10)) - training_folds
testing_mask = np.array([fold in testing_folds for fold in folds])
xtest = images[testing_mask][:testing_size]
ytest = labels[testing_mask][:testing_size]
testing_size = xtest.shape[0]
print("Testing folds: ", testing_folds,
      "\t Number of test words: ", testing_size)

regu = 1 / training_size

# In case of constant step size, the following size is guaranteed to yield a linear convergence
# step_size = regu * nb_words * 2 / (max_radius ** 2 + regu * nb_words * 2)
# print("step size:", step_size)

parameters = {
    'regu': regu,
    'npass': 100,
    'sampling': 'gap',
    'non_uniformity': .8,
    'monitoring_period': 5,
    'sampler_period': None,
    'precision': 1e-7,
    '_debug': True,
}
print(parameters)

time_stamp = time.strftime("%Y%m%d_%H%M%S")
dirname = "logs/" + time_stamp \
          + "_OCR_" \
          + parameters['sampling'] + "_" \
          + str(parameters['non_uniformity']) + "_" \
          + str(parameters['sampler_period'])

parameters['logdir'] = dirname

if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists(dirname):
    os.mkdir(dirname)

# write parameters to text file
with open(dirname + '/parameters.txt', 'w') as file:
    file.write(" time :" + time_stamp)
    file.write("\n training folds :" + str(training_folds))
    file.write("\n number of training words :" + str(training_size))
    file.write("\n number of testing words :" + str(testing_size))
    for key, value in parameters.items():
        file.write("\n" + key + " : " + str(value))

fullweights, fullmargs = sdca.sdca(features, xtrain, ytrain, xtest=xtest, ytest=ytest,
                                   **parameters)

os.system('say "I am done."')
