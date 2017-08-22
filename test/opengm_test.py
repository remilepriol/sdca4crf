import numpy as np
import opengm
import pygraphviz

pygraphviz.test()

number_of_labels = 26 * np.ones(5)
gm = opengm.graphicalModel(number_of_labels)
opengm.visualizeGm(gm)
