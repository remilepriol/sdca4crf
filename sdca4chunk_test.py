# standard imports
import os
import time

import matplotlib.pyplot as plt
import numpy as np

# custom imports
import sdca
from chunk import features, parse

# I want to come down to 30% of the time on the line search.

trainfile = "../data/conll2000/train.att.txt"
testfile = "../data/conll2000/test.att.txt"

train_size = 60000000
test_size = 100000000

dattributes = parse.build_dictionary(trainfile, train_size)
print("Number of different features:", len(dattributes))

xtrain, ytrain = parse.read_data(trainfile, dattributes, train_size)
train_size = xtrain.shape[0]
print("Size of training set:", train_size)

xtest, ytest = parse.read_data(testfile, dattributes, test_size)
test_size = xtest.shape[0]
print("Size of test set:", test_size)

regu = 1 / train_size

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
          + "_CONLL2000_" \
          + parameters['sampling'] + "_" \
          + str(parameters['non_uniformity']) + "_" \
          + str(parameters['sampler_period'])

parameters['logdir'] = dirname

if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists(dirname):
    os.mkdir(dirname)

# write all parameters to text file
with open(dirname + '/parameters.txt', 'w') as file:
    file.write(" time :" + time_stamp)
    file.write("\n train size :" + str(train_size))
    file.write("\n test size :" + str(test_size))
    for key, value in parameters.items():
        file.write("\n" + key + " : " + str(value))

fullmargs, fullweights, fullobjective, fullannex = \
    sdca.sdca(features, xtrain, ytrain, xtest=xtest, ytest=ytest, **parameters)

print("End")
os.system('say "I am done."')

np.save(dirname + "/weights.npy", fullweights)
np.save(dirname + "/objectives.npy", fullobjective)
np.save(dirname + "/annex.npy", fullannex)

plt.figure(figsize=(12, 4))
plt.suptitle("Performance of SDCA on CONLL2000 with n=%i and lambda=%.1e" % (train_size, regu))
plt.subplot(1, 2, 1)
plt.ylabel("log10(duality gap)")
plt.plot(np.log10(fullobjective[:, 0]))
ticksrange = 2 * np.arange(len(fullobjective) / 2, dtype=int)
plt.xticks(ticksrange, parameters['monitoring_period'] * ticksrange)
plt.xlabel("number of pass over the data")
plt.subplot(1, 2, 2)
plt.plot(fullobjective[:, 3], np.log10(fullobjective[:, 0]))
plt.xlabel("time (s)")
plt.savefig(dirname + "/duality_gap.pdf")
