import matplotlib.pyplot as plt

import random_counters as randc

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
