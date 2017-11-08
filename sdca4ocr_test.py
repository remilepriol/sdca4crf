# standard imports
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import features
# custom imports
import ocr
import parse

labels, images, folds = parse.letters_to_labels_and_words(
    parse.read_lettersfile('../data/ocr/letter.data.tsv'))

radii = features.radii(images,labels)
max_radius = np.amax(radii)

training_folds = set(range(1, 10))
training_mask = np.array([fold in training_folds for fold in folds])
xtrain = images[training_mask]
ytrain = labels[training_mask]
nb_words = xtrain.shape[0]
print("Training folds: ", training_folds,
      "\t Number of training words:", nb_words)

testing_folds = set(range(10)) - training_folds
testing_mask = np.array([fold in testing_folds for fold in folds])
xtest = images[testing_mask]
ytest = labels[testing_mask]
test_words = xtest.shape[0]
print("Testing folds: ", testing_folds,
      "\t Number of test words: ", test_words)

regu = 1 / nb_words

# In case of constant step size, the following size is guaranteed to yield a linear convergence
# step_size = regu * nb_words * 2 / (max_radius ** 2 + regu * nb_words * 2)
# print("step size:", step_size)

time_stamp = time.strftime("%Y%m%d_%H%M%S")
dirname = "logs/" + time_stamp + "_n" + str(nb_words)
if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists(dirname):
    os.mkdir(dirname)

parameters = {
    'regu': regu,
    'npass': 100,
    'sampling': 'importance',
    'non_uniformity': .8,
    'monitoring_period': 5,
    'sampler_period': None,
    'precision': 1e-7,
    'subprecision': 1e-2,
    'init': 'OEG',
    'logdir': dirname,
    '_debug': True,
}
print(parameters)

# write parameters to text file
with open(dirname + '/parameters.txt', 'w') as file:
    file.write(" time :" + time_stamp)
    file.write("\n training folds :" + str(training_folds))
    file.write("\n number of training words :" + str(nb_words))
    file.write("\n number of testing words :" + str(test_words))
    for key, value in parameters.items():
        file.write("\n" + key + " : " + str(value))

fullmargs, fullweights, fullobjective, fullannex = \
    ocr.sdca(xtrain, ytrain, xtest=xtest, ytest=ytest, **parameters)

os.system('say "I am done."')

np.save(dirname + "/weights.npy", fullweights)
np.save(dirname + "/objectives.npy", fullobjective)
np.save(dirname + "/annex.npy", fullannex)

plt.figure(figsize=(12, 4))
plt.suptitle("Performance of SDCA on OCR with n=%i and lambda=%.1e" % (xtrain.shape[0], regu))
plt.subplot(1, 2, 1)
plt.ylabel("log10(duality gap)")
plt.plot(np.log10(fullobjective[:, 0]))
plt.xlabel("number of pass over the data")
ticksrange = 2 * np.arange(len(fullobjective) / 2, dtype=int)
plt.xticks(ticksrange, parameters['monitoring_period'] * ticksrange)
plt.xlabel("number of pass over the data")
plt.subplot(1, 2, 2)
plt.plot(fullobjective[:, 3], np.log10(fullobjective[:, 0]))
plt.xlabel("time (s)")
plt.savefig(dirname + "/duality_gap.pdf")
