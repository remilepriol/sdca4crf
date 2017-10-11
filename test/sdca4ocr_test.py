# standard imports
import os

import numpy as np

# custom imports
import ocr

labels, images, folds = ocr.letters_to_labels_and_words(
    ocr.read_lettersfile('../../data/ocr/letter.data.tsv'))

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

fullmargs, fullweights, fullobjective, fulltiming, fullannex = \
    ocr.sdca(x, y, non_uniformity=1, **parameters)

os.system('say "I am done."')
