import matplotlib.pyplot as plt
import numpy as np

import generator as gen
import multiclass_logistic_regression as mlr
import random_counters as randc

############################################################################
# GEN and Multiclass Logistic Regression
############################################################################
n = 1000
d = 2
k = 6
bias = 1
x, y, mu, sig = gen.gengaussianmixture(n, d, k, scale=.05)
x = gen.standardize(x)
x = np.concatenate([x, bias * np.ones([n, 1])], axis=1)

reg = 1
model = mlr.MulticlassLogisticRegression(reg, x, y)

alpha0 = 1 / k * np.ones([n, k])
obj_sdca, time_sdca = model.sdca(x, y, alpha0, npass=14, precision=1e-5)

############################################################################
# RANDOM COUNTERS
############################################################################
bins = 17
rc = randc.RandomCounters(np.ones(bins))
rc.update(.3, 6)
rc.update(.3, 7)
rc.update(.5, 1)
rc.update(.1, bins - 1)
sampling = []
for i in range(10000):
    sampling.append(rc.sample())
print(rc.score_tree)
plt.hist(sampling, bins=bins)
plt.show()
