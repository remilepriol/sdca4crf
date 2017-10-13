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
    parse.read_lettersfile('data/ocr/letter.data.tsv'))

radii = features.radii(images)
max_radius = np.amax(radii)

which_fold = 0
x = images[folds != which_fold]
y = labels[folds != which_fold]
nb_words = x.shape[0]
npass = 100
update_period = 5
regu = 1 / nb_words
step_size = regu * nb_words * 2 / (max_radius ** 2 + regu * nb_words * 2)
print("Number of words:", nb_words)
print("step size:", step_size)

time_stamp = time.strftime("%Y%m%d_%H%M%S")
dirname = "logs/" + time_stamp + "_n" + str(nb_words)
if not os.path.exists("logs"):
    os.mkdir("logs")
if not os.path.exists(dirname):
    os.mkdir(dirname)

parameters = {'npass': 100,
              'update_period': update_period,
              'regu': regu,
              '_debug': True,
              'precision': 1e-8,
              'subprecision': 1e-8,
              'init': 'empirical',
              'logdir': dirname}
print(parameters)

fullmargs, fullweights, fullobjective, fullannex = \
    ocr.sdca(x, y, non_uniformity=1, **parameters)

os.system('say "I am done."')

np.save(dirname + "/marginals.npy", fullmargs)
np.save(dirname + "/weights.npy", fullweights)
np.save(dirname + "/objectives.npy", fullobjective)
np.save(dirname + "/annex.npy", fullannex)

plt.figure(figsize=(12, 4))
plt.suptitle("Performance of SDCA on OCR with n=%i and lambda=%.1e" % (x.shape[0], regu))
plt.subplot(1, 2, 1)
plt.ylabel("log10(duality gap)")
plt.plot(np.log10(fullobjective[:, 0]))
plt.xlabel("number of pass over the data")
ticksrange = 2 * np.arange(len(fullobjective) / 2, dtype=int)
plt.xticks(ticksrange, update_period * ticksrange)
plt.xlabel("number of pass over the data")
plt.subplot(1, 2, 2)
plt.plot(fullobjective[:, 3], np.log10(fullobjective[:, 0]))
plt.xlabel("time (s)")
plt.savefig(dirname + "/duality_gap.pdf")
