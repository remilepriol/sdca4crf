# standard imports
import os
import time

import matplotlib.pyplot as plt
import numpy as np

# custom imports
import ocr

labels, images, folds = ocr.letters_to_labels_and_words(
    ocr.read_lettersfile('../data/ocr/letter.data.tsv'))

radii = ocr.radii(images)
max_radius = np.max(radii)

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

parameters = {'npass': 100,
              'update_period': update_period,
              'regu': regu,
              '_debug': True,
              'precision': 1e-4,
              'subprecision': 1e-4}
print(parameters)

fullmargs, fullweights, fullobjective, fullannex = \
    ocr.sdca(x, y, non_uniformity=1, **parameters)

os.system('say "I am done."')

np.save("results/" + time.strftime("%Y%m%d_%H%M%S") + "optmargs.npy", fullmargs)
np.save("results/" + time.strftime("%Y%m%d_%H%M%S") + "optweights.npy", fullweights)
np.save("results/" + time.strftime("%Y%m%d_%H%M%S") + "objective.npy", fullobjective)
np.save("results/" + time.strftime("%Y%m%d_%H%M%S") + "timing.npy", fulltiming)
np.save("results/" + time.strftime("%Y%m%d_%H%M%S") + "annex.npy", fullannex)

plt.figure(figsize=(12, 4))
plt.suptitle("Performance of SDCA on OCR with n=%i and lambda=%.1e" % (x.shape[0], regu))
plt.subplot(1, 2, 1)
plt.ylabel("log10(duality gap)")
plt.plot(np.log10(fullobjective))
plt.xlabel("number of pass over the data")
ticksrange = 2 * np.arange(len(fullobjective) / 2, dtype=int)
plt.xticks(ticksrange, update_period * ticksrange)
plt.xlabel("number of pass over the data")
plt.subplot(1, 2, 2)
plt.plot(fulltiming, np.log10(fullobjective))
plt.xlabel("time (s)")
plt.savefig("images/" + time.strftime("%Y%m%d_%H%M%S") + "_ocr_perf.pdf")
